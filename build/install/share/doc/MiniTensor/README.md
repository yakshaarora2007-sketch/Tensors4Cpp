# MiniTensor

MiniTensor is a dependency-free C++20 library for owning, contiguous,
arbitrary-rank tensors. It is implemented from scratch with standard C++
containers and algorithms; no external tensor or matrix library is used.

## Features

- Arbitrary-rank shapes and contiguous row-major `std::vector<T>` storage.
- Checked shape products, scalar and empty tensor semantics.
- Row-major strides and bounds-checked multidimensional access.
- Element-wise addition, subtraction, and multiplication (no broadcasting).
- Serial, cache-blocked, and optional OpenMP matrix multiplication.
- Copying reshape and materialized two-dimensional transpose.
- RAII ownership and ordinary value copy/move semantics.
- Header-only target, tests, examples, install/export rules, and benchmark.

Basic slicing and non-owning views are intentionally not included. No GPU,
broadcasting, matrix decompositions, or higher-rank batched matrix
multiplication is provided.

## Tensor and memory model

`Tensor<T>` owns a shape (`std::vector<std::size_t>`), row-major strides, and a
single contiguous data buffer (`std::vector<T>`). Shape and values are kept
separate; there are no per-dimension allocations. Contiguous ownership provides
simple RAII, predictable lifetime, and cache-friendly traversal. Copies own
independent storage; moves use standard vector semantics.

The number of elements is the product of the dimensions. Shape `{}` describes
a scalar with one value. A shape containing a zero dimension has no elements.
Products are checked for `size_t` overflow before allocating the data buffer.
Strides are row-major (for `{2, 3, 4}`, `{12, 4, 1}`). Empty tensors use all-zero
strides because no coordinate is valid.

## Usage

```cpp
#include <minitensor/minitensor.hpp>
#include <vector>

int main() {
    minitensor::Tensor<float> a({2, 3},
        std::vector<float>{1, 2, 3, 4, 5, 6});
    a.at({1, 2}) = 10.0F;
    a.set({0, 1}, 8.0F);
    float value = a(1, 2);

    auto reshaped = a.reshape({3, 2});
    auto transposed = a.transpose();
    auto sum = a + a;
    auto elementwise_product = a * a;

    minitensor::Tensor<float> b({3, 2}, 1.0F);
    auto matrix_product = a.matmul(b);
    (void)value;
    (void)reshaped;
    (void)transposed;
    (void)sum;
    (void)elementwise_product;
    (void)matrix_product;
}
```

`at()` and `set()` require exactly one index per dimension and throw
`std::invalid_argument` for an index-rank mismatch or `std::out_of_range` for
an invalid coordinate. Element-wise operators require identical shapes.
`A * B` is element-wise; `A.matmul(B)` is two-dimensional matrix
multiplication. No implicit broadcasting is performed.

`reshape(new_shape)` preserves row-major value order and returns an owning copy.
`transpose()` accepts only rank-two tensors and returns a new contiguous tensor.
These operations currently copy data; MiniTensor has no shared-storage view
model.

## Build and install

Requirements: CMake 3.16+ and a C++20 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
cmake --install build --prefix install
```

The exported interface target is `MiniTensor::minitensor`, with installed
headers under `include/minitensor/`. Consumers can use
`find_package(MiniTensor CONFIG REQUIRED)` and link
`MiniTensor::minitensor`, then include `<minitensor/minitensor.hpp>`.

Useful CMake options:

- `MINITENSOR_BUILD_EXAMPLES` (default `ON`)
- `MINITENSOR_BUILD_BENCHMARKS` (default `ON`)
- `BUILD_TESTING` (default `ON`)
- `MINITENSOR_USE_GTEST` (default `ON`; uses GoogleTest if found, otherwise
  uses the built-in test executable)
- `MINITENSOR_ENABLE_OPENMP` (default `OFF`; requires OpenMP)
- `MINITENSOR_ENABLE_NATIVE_OPTIMIZATIONS` (default `OFF`; enables
  `-march=native` or MSVC `/arch:AVX2`)
- `MINITENSOR_ENABLE_SANITIZERS` (default `OFF`; GCC/Clang AddressSanitizer and
  UndefinedBehaviorSanitizer, or MSVC AddressSanitizer)

For example, configure an OpenMP build with
`cmake -S . -B build-openmp -DMINITENSOR_ENABLE_OPENMP=ON`.

## Tests and examples

The tests cover construction, shape/stride correctness, indexing and error
paths, arithmetic, serial/blocked/parallel-fallback multiplication, reshape,
transpose, floating-point tolerance, copy independence, and zero/overflow
edge cases.

Build the dependency-free examples `basic_tensor` and `image_processing`. The
image example constructs a synthetic RGB tensor and demonstrates brightness,
element-wise intensity scaling, and channel swapping exclusively through the
public API.

Run them with `build\Release\basic_tensor.exe` and
`build\Release\image_processing.exe` on Windows, or their corresponding paths
on other platforms.

## Performance and benchmarks

The serial `matmul()` uses a row-major `i-k-j` loop. `matmul_blocked()` tiles
row and inner dimensions while traversing complete contiguous output rows; it
accepts a positive block size (default 64).
`matmul_parallel()` uses OpenMP row parallelism when enabled; otherwise it runs
the serial implementation. The OpenMP path keeps small workloads
serial and assigns each output row to one worker.

The benchmark validates blocked and parallel-path results against the serial
result, performs one warmup, and reports averages over three measured runs by
default for square matrices of sizes 64, 256, 512, and 1024. It reports compiler,
build configuration, OpenMP state/thread count, block size, elapsed times,
and speedups. One actual run in this development environment (Windows, MSVC
19.51, Release `/O2 /Ob2 /DNDEBUG`, OpenMP enabled, four threads, block size 64,
one warmup and three measured runs) produced:

| Size | Serial (ms) | Blocked (ms) | OpenMP (ms) | OpenMP speedup |
|---:|---:|---:|---:|---:|
| 64 | 0.057 | 0.067 | 0.050 | 1.130x |
| 256 | 3.865 | 3.839 | 2.381 | 1.623x |
| 512 | 34.627 | 33.118 | 24.499 | 1.413x |
| 1024 | 238.526 | 257.955 | 308.166 | 0.774x |

These are measurements from one run, not performance guarantees. In particular,
the OpenMP path regressed at 1024² in this environment, illustrating that
parallel speedups are workload- and machine-dependent.

```sh
build\Release\minitensor_benchmark.exe
build\Release\minitensor_benchmark.exe --iterations 5 --max-size 512
build\Release\minitensor_benchmark.exe --block-size 64
```

For meaningful performance measurements, use a Release build. Hardware-specific
optimization is optional and should only be enabled for a build intended for
the target machine.

## Limitations and future work

Current operations are synchronous and CPU-only. Matrix multiplication accepts
only rank-two tensors. Reshape and transpose materialize copies; slices and
shared-storage views have not been implemented. OpenMP is optional, and the
non-OpenMP `matmul_parallel()` API falls back to blocked serial multiplication.
Benchmark speedups are reported from actual runs and are not guaranteed.

Possible future work includes safe lifetime-managed views/slicing, batched
linear algebra, additional reductions, and improved benchmark-guided kernel
selection.
