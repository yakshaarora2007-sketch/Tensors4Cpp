# Performance and Optimization Guide

High-performance numerical computing is not just about writing clean algorithms; it is about writing code that cooperates with modern computer hardware. 

This document explores the performance architecture of MiniTensor. We will examine cache locality, memory access patterns, loop ordering, cache tiling (blocking), OpenMP multithreading, and real measured benchmark results.

---

## Table of Contents

1. [The Performance Challenge in Multidimensional Computing](#1-the-performance-challenge-in-multidimensional-computing)
2. [Memory Access Patterns and Cache Locality](#2-memory-access-patterns-and-cache-locality)
   - [Why Row-Major Traversal Maximizes Bus Bandwidth](#why-row-major-traversal-maximizes-bus-bandwidth)
   - [The "Stride Penalty" of Column Traversal](#the-stride-penalty-of-column-traversal)
3. [The `i-k-j` Matrix Multiplication Optimization](#3-the-i-k-j-matrix-multiplication-optimization)
   - [The Classical `i-j-k` Loop (Cache Inefficient)](#the-classical-i-j-k-loop-cache-inefficient)
   - [The Reordered `i-k-j` Loop (Cache Optimal)](#the-reordered-i-k-j-loop-cache-optimal)
4. [Cache Tiling / Blocking (`matmul_blocked`)](#4-cache-tiling--blocking-matmul_blocked)
   - [The L1/L2 Cache Capacity Cliff](#the-l1l2-cache-capacity-cliff)
   - [How Sub-Block Tiling Keeps Data Cache-Hot](#how-sub-block-tiling-keeps-data-cache-hot)
   - [Selecting the Ideal Block Size](#selecting-the-ideal-block-size)
5. [Parallel Execution with OpenMP (`matmul_parallel`)](#5-parallel-execution-with-openmp-matmul_parallel)
   - [Workload Partitioning Across CPU Cores](#workload-partitioning-across-cpu-cores)
   - [The Overhead Trade-Off for Small Matrices](#the-overhead-trade-off-for-small-matrices)
6. [Hardware-Specific Compiler Optimizations](#6-hardware-specific-compiler-optimizations)
   - [SIMD Vectorization (`AVX2` / `-march=native`)](#simd-vectorization-avx2---marchnative)
7. [Empirical Benchmark Results and Analysis](#7-empirical-benchmark-results-and-analysis)
   - [Benchmark Methodology](#benchmark-methodology)
   - [Actual Measured Data (Windows / MSVC 19.51)](#actual-measured-data-windows--msvc-1951)
   - [Analysis of Results](#analysis-of-results)
8. [How to Run Benchmarks Yourself](#8-how-to-run-benchmarks-yourself)

---

## 1. The Performance Challenge in Multidimensional Computing

Modern computer processors can execute hundreds of billions of arithmetic instructions per second. However, their ability to feed data to those execution units is severely bottlenecked by **memory latency** and **memory bandwidth**.

- Retrieving an element from the L1 cache takes $\approx 1$ nanosecond (4 clock cycles).
- Retrieving an element from system RAM takes $\approx 60 - 100$ nanoseconds (hundreds of clock cycles).

If an algorithm accesses memory in a random or discontinuous order, the processor spends most of its time stalled, waiting for RAM. 

MiniTensor achieves high performance through three hardware-cooperative techniques:
1. **Contiguous Storage:** Storing all data in an uninterrupted 1D buffer.
2. **Sequential Loop Ordering (`i-k-j`):** Ensuring that innermost loops step through memory in unit strides.
3. **Cache Tiling and Thread Partitioning:** Reusing cached data blocks and dividing rows across multiple processor cores.

---

## 2. Memory Access Patterns and Cache Locality

### Why Row-Major Traversal Maximizes Bus Bandwidth

When data is laid out in **row-major order**, elements in the same row are physically placed side-by-side in RAM:

```text
Row 0: [ (0,0), (0,1), (0,2), (0,3) ]
Row 1: [ (1,0), (1,1), (1,2), (1,3) ]
```

When you read `(0, 0)`, the hardware memory controller fetches a full **64-byte Cache Line** from RAM into CPU cache. For a 32-bit `float` (4 bytes), a single cache line contains **16 consecutive floats**!
- If your loop immediately reads `(0, 1)`, `(0, 2)`, $\dots$, those numbers are already sitting in L1 cache! Access time is practically instantaneous.
- Furthermore, the hardware **Stream Prefetcher** detects your linear march through memory and pre-loads the next cache lines from RAM in the background.

### The "Stride Penalty" of Column Traversal

Now consider what happens if you traverse elements by column (stepping vertically down the table):
```text
Reading (0, 0) -> Fetches 64 bytes for Row 0
Reading (1, 0) -> Jumps to Row 1 (different cache line in RAM!)
Reading (2, 0) -> Jumps to Row 2 (different cache line in RAM!)
```
Every single read requires fetching an entire 64-byte cache line from RAM, using only 4 bytes of it, and discarding the rest. This wastes **93.75% of your memory bandwidth** and causes severe performance degradation!

---

## 3. The `i-k-j` Matrix Multiplication Optimization

Matrix multiplication multiplies Row $i$ of matrix $A$ by Column $j$ of matrix $B$.

### The Classical `i-j-k` Loop (Cache Inefficient)
The traditional algorithm taught in mathematics textbooks is structured as:
```cpp
// Textbook i-j-k algorithm:
for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
        T sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += A[i, k] * B[k, j]; // Notice B[k, j]!
        }
        C[i, j] = sum;
    }
}
```
In the innermost loop, $k$ increments while $j$ is constant. In row-major order, `B[k, j]` steps vertically down column $j$, jumping across memory by $N$ elements on every single iteration! For large matrices, every iteration of the $k$ loop suffers an expensive cache miss.

### The Reordered `i-k-j` Loop (Cache Optimal)
MiniTensor swaps the inner loops to `i -> k -> j`:
```cpp
// MiniTensor implementation:
for (size_t row = 0; row < M; ++row) {
    for (size_t inner = 0; inner < K; ++inner) {
        const T left = A[row * K + inner]; // Kept in a CPU register!
        for (size_t col = 0; col < N; ++col) {
            C[row * N + col] += left * B[inner * N + col];
        }
    }
}
```
Examine the innermost loop over `col`:
- `left` is constant and lives in a fast CPU register.
- `B[inner * N + col]` advances by $1$ in memory on every step!
- `C[row * N + col]` advances by $1$ in memory on every step!
- Both matrix $B$ and matrix $C$ are accessed in **strictly contiguous, unit-stride row-major order**.

This single change dramatically improves memory throughput and enables the compiler to generate SIMD vector instructions.

---

## 4. Cache Tiling / Blocking (`matmul_blocked`)

### The L1/L2 Cache Capacity Cliff
Even with the `i-k-j` loop ordering, when matrices grow to $1024 \times 1024$, matrix $B$ occupies $1024 \times 1024 \times 4 \text{ bytes} = 4 \text{ MB}$.
A typical CPU L2 cache is $512 \text{ KB}$ to $1 \text{ MB}$ per core. As the outer loop moves from Row 0 to Row 1, the data from matrix $B$ has already been evicted from cache and must be reloaded from slow RAM all over again.

### How Sub-Block Tiling Keeps Data Cache-Hot
**Tiling (blocking)** partitions the large matrices into square sub-tiles of size $B \times B$ (default 64):
```cpp
for (size_t row_block = 0; row_block < M; row_block += block_size) {
    for (size_t inner_block = 0; inner_block < K; inner_block += block_size) {
        // Compute the tile product...
    }
}
```
A $64 \times 64$ sub-tile of single-precision floats takes:
$$64 \times 64 \times 4 \text{ bytes} = 16,384 \text{ bytes } (16 \text{ KB})$$
16 KB fits completely inside the ultra-fast L1 Data Cache (typically 32 KB to 48 KB per core). The CPU can reuse the numbers in that tile repeatedly without touching RAM!

---

## 5. Parallel Execution with OpenMP (`matmul_parallel`)

### Workload Partitioning Across CPU Cores
MiniTensor provides multithreaded matrix multiplication via OpenMP:
```cpp
#pragma omp parallel for if(large_workload) schedule(static)
for (std::ptrdiff_t row = 0; row < dimensions.rows; ++row) {
    // Process row...
}
```
- The rows of the output matrix $C$ are divided evenly among all available CPU cores.
- Thread 0 computes rows $0 \dots 15$, Thread 1 computes rows $16 \dots 31$, and so on.
- Because each thread writes strictly to its own distinct memory rows, there is **zero contention, zero locking, and zero thread-synchronization overhead** during calculation.

### The Overhead Trade-Off for Small Matrices
Spawning and joining threads in the operating system has an overhead of a few microseconds. For small matrices (e.g., $32 \times 32$), running in parallel is actually *slower* than running serially!

MiniTensor uses an intelligent heuristic:
```cpp
const bool large_workload =
    dimensions.rows >= 64 && dimensions.columns >= 64 && dimensions.inner >= 64;
```
If dimensions are below 64, OpenMP executes the loop serially on the main thread, avoiding thread creation overhead.

---

## 6. Hardware-Specific Compiler Optimizations

You can enable native CPU instruction sets via CMake:
```sh
cmake -S . -B build -DMINITENSOR_ENABLE_NATIVE_OPTIMIZATIONS=ON
```
- On **MSVC**, this passes `/arch:AVX2`.
- On **GCC / Clang**, this passes `-march=native`.

This enables the compiler to use Advanced Vector Extensions (AVX2), allowing the CPU to perform 8 single-precision or 4 double-precision floating-point multiplications and additions in a single CPU clock cycle (Fused Multiply-Add, FMA).

---

## 7. Empirical Benchmark Results and Analysis

### Benchmark Methodology
The benchmark suite in `benchmarks/benchmark.cpp`:
- Tests square matrices of dimension $64 \times 64$, $256 \times 256$, $512 \times 512$, and $1024 \times 1024$.
- Executes 1 warmup iteration to prime CPU frequency and caches.
- Averages execution time over measured iterations (default 3).
- Verifies mathematical correctness by checking that blocked and parallel results match baseline within floating-point tolerance ($10^{-4}$).

### Actual Measured Data (Windows / MSVC 19.51)

The following tables show actual measured performance runs recorded in this development environment (Windows x64, MSVC 19.51, Release build `/O2 /Ob2 /DNDEBUG`, block size 64):

#### Run A: Single-Threaded Baseline (OpenMP Disabled)
```text
Compiler: MSVC 1951
Build configuration: Release
OpenMP: disabled (parallel API uses serial fallback)
Threads: 1, block size: 64
```
| Dimension | Serial (ms) | Blocked (ms) | Blocked Speedup |
| :---: | :---: | :---: | :---: |
| **64** | 0.058 ms | 0.058 ms | 1.005x |
| **256** | 4.577 ms | 4.137 ms | 1.106x |

#### Run B: Multithreaded Execution (OpenMP Enabled, 16 Threads)
```text
Compiler: MSVC 1951
Build configuration: Release
OpenMP: enabled
Threads: 16, block size: 64
```
| Dimension | Serial (ms) | Blocked (ms) | OpenMP Parallel (ms) | Parallel Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **64** | 0.226 ms | 0.185 ms | 0.041 ms | **5.49x** |
| **256** | 6.899 ms | 6.977 ms | 1.281 ms | **5.39x** |

#### Run C: Standard Historical Benchmark Reference (4 Threads)
Recorded during project calibration (4 threads, 3 iterations):

| Dimension | Serial (ms) | Blocked (ms) | OpenMP (ms) | OpenMP Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **64** | 0.057 ms | 0.067 ms | 0.050 ms | 1.130x |
| **256** | 3.865 ms | 3.839 ms | 2.381 ms | 1.623x |
| **512** | 34.627 ms | 33.118 ms | 24.499 ms | 1.413x |
| **1024** | 238.526 ms | 257.955 ms | 308.166 ms | 0.774x |

### Analysis of Results
1. **Strong Multithreaded Scaling at Medium Sizes:**
   - At $256 \times 256$, OpenMP multithreading achieved a **$5.39\times$ speedup** on 16 threads, reducing computation time from nearly 7 milliseconds down to just 1.28 milliseconds.
2. **The 1024² Memory Saturation Effect:**
   - Notice that in Run C, at $1024 \times 1024$, the OpenMP speedup regressed to $0.774\times$. Why?
   - At $1024 \times 1024$, multiple cores simultaneously compete for the system memory bus. When all cores saturate the physical RAM bus bandwidth, threads stall waiting for memory transfers, and synchronization overhead can exceed computation time.
   - **Key takeaway:** Parallel speedups are workload-, architecture-, and memory-bus-dependent.

---

## 8. How to Run Benchmarks Yourself

You can execute the benchmark binary directly from your build directory:

```sh
# Run default benchmark (dimensions 64, 256, 512, 1024; 3 iterations):
build\Release\minitensor_benchmark.exe

# Customize iterations and maximum dimension:
build\Release\minitensor_benchmark.exe --iterations 5 --max-size 512

# Customize cache block size (e.g. testing block size 32 or 128):
build\Release\minitensor_benchmark.exe --block-size 32
```

> [!IMPORTANT]
> **Always Use Release Builds for Benchmarking!**
> Never run benchmarks in Debug mode. Debug builds disable compiler optimizations, disable loop unrolling, and emit extra safety checks that make code $10\times$ to $50\times$ slower. Always configure with `-DCMAKE_BUILD_TYPE=Release`!

---

Proceed to:
- [Testing Guide](testing.md) to explore test coverage and error handling.
- [Repository Architecture](architecture.md) for build options and CMake targets.
- [Usage Tutorial](usage_tutorial.md) for code walkthroughs.

