# MiniTensor: Complete Technical Manual and Learning Guide

Welcome to **MiniTensor**! MiniTensor is a dependency-free, header-only C++20 library for owning, contiguous, arbitrary-rank multidimensional arrays (tensors).

Whether you are a student discovering tensors for the very first time, a software developer transitioning into machine learning or computer vision, or an experienced systems programmer exploring cache-friendly C++ architectures, this manual is designed to be your complete learning resource and user manual.

---

## Documentation Navigation Guide

This `README.md` provides a comprehensive overview of the library. For in-depth technical deep dives into specific topics, explore the dedicated documentation guides in [`docs/`](docs/):

| Guide | Description | Key Topics |
| :--- | :--- | :--- |
| 📖 [**Tensor Fundamentals**](docs/tensor_fundamentals.md) | Zero-to-hero primer on multidimensional data. | Scalars, vectors, matrices, rank, shape, strides, row-major order. |
| 🔬 [**Tensor Internals**](docs/tensor_internals.md) | Architectural deep dive into the `Tensor<T>` class. | Class layout, `calculate_size`, overflow checks, stride calculations. |
| 🛡️ [**Memory Management & Safety**](docs/memory_management.md) | Memory safety, RAII, and cache performance. | Contiguous buffers, deep copies, move semantics, spatial locality. |
| 🧮 [**Mathematical Operations**](docs/mathematical_operations.md) | Mathematical theory, algorithms, and complexity. | Element-wise arithmetic, `i-k-j` matrix multiplication, tiling, OpenMP. |
| 📚 [**Complete API Reference**](docs/api_reference.md) | Exhaustive signature-by-signature API manual. | All constructors, methods, operators, parameters, and exceptions. |
| 🚀 [**Progressive Usage Tutorial**](docs/usage_tutorial.md) | 15-step hands-on coding tutorial. | Step-by-step code samples building up to an image processing pipeline. |
| 🏛️ [**Repository Architecture**](docs/architecture.md) | Project directory structure and CMake setup. | Target architecture, build options, package export, linking. |
| ⚡ [**Performance & Optimization**](docs/performance.md) | Hardware co-design and empirical benchmarks. | Cache lines, memory bus saturation, AVX2, real benchmark runs. |
| 🧪 [**Testing & Reliability**](docs/testing.md) | Verification suites and edge-case handling. | Dual testing modes (GoogleTest vs fallback), sanitizers, error paths. |
| 📄 [**Technical Research Paper (Markdown)**](docs/mini_tensor_library_paper.md) | Formal academic systems research paper. | Theoretical background, architecture, IEEE references, trade-offs. |
| 📑 [**Technical Research Paper (PDF)**](docs/mini_tensor_library_paper.pdf) | Publication-formatted 16-page PDF document. | Complete paper formatted with equations, tables, and typography. |

---

## Table of Contents

1. [Section A: Introduction](#section-a-introduction)
2. [Section B: Understanding Tensors From Absolute Zero](#section-b-understanding-tensors-from-absolute-zero)
   - [1. What is Data?](#1-what-is-data)
   - [2. What is a Tensor?](#2-what-is-a-tensor)
   - [3. Tensor Terminology Demystified](#3-tensor-terminology-demystified)
   - [4. Understanding Shape](#4-understanding-shape)
   - [5. Why Tensors Instead of Ordinary Arrays?](#5-why-tensors-instead-of-ordinary-arrays)
3. [Section C: How Tensors Work Inside This Library](#section-c-how-tensors-work-inside-this-library)
   - [1. Tensor Representation](#1-tensor-representation)
   - [2. Contiguous Memory and Row-Major Order](#2-contiguous-memory-and-row-major-order)
   - [3. Memory Safety and RAII](#3-memory-safety-and-raii)
   - [4. Strides and Multidimensional Coordinate Indexing](#4-strides-and-multidimensional-coordinate-indexing)
4. [Section D: Complete Library API Reference Overview](#section-d-complete-library-api-reference-overview)
5. [Section E: Understanding Mathematical Operations](#section-e-understanding-mathematical-operations)
   - [Element-Wise Operations vs. Linear Algebra](#element-wise-operations-vs-linear-algebra)
   - [Matrix Multiplication (`matmul`) and the `i-k-j` Optimization](#matrix-multiplication-matmul-and-the-i-k-j-optimization)
   - [Cache-Blocked Matrix Multiplication (`matmul_blocked`)](#cache-blocked-matrix-multiplication-matmul_blocked)
   - [Parallel Matrix Multiplication (`matmul_parallel`)](#parallel-matrix-multiplication-matmul_parallel)
   - [Reshaping and Transposing](#reshaping-and-transposing)
   - [Complexity Summary](#complexity-summary)
6. [Section F: Getting Started: Installation and Setup](#section-f-getting-started-installation-and-setup)
   - [Prerequisites](#prerequisites)
   - [Building the Project](#building-the-project)
   - [Running the Tests](#running-the-tests)
   - [Running the Examples](#running-the-examples)
   - [Running the Benchmarks](#running-the-benchmarks)
   - [Integrating MiniTensor into Downstream CMake Projects](#integrating-minitensor-into-downstream-cmake-projects)
7. [Section G: Step-by-Step Usage Tutorial](#section-g-step-by-step-usage-tutorial)
8. [Section H: Understanding the Repository Structure](#section-h-understanding-the-repository-structure)
9. [Section I: Performance, Optimization, and Benchmarks](#section-i-performance-optimization-and-benchmarks)
10. [Section J: Testing, Edge Cases, and Reliability](#section-j-testing-edge-cases-and-reliability)
11. [Section K: Practical Applications](#section-k-practical-applications)
12. [Section L: Limitations and Future Roadmap](#section-l-limitations-and-future-roadmap)

---

## Section A: Introduction

### What is MiniTensor?
**MiniTensor** is a lightweight, dependency-free C++20 library designed for owning, contiguous, arbitrary-rank multidimensional arrays. It provides standard multidimensional indexing, element-wise arithmetic, matrix multiplication kernels (serial, cache-blocked, and multithreaded OpenMP), reshaping, and transposition.

### What Problem Does It Solve?
In scientific computing, game physics, computer vision, and machine learning, programs manipulate data organized in grids, volumes, and higher-dimensional tables.
Standard C++ does not provide a native multidimensional array container. Programmers typically resort to nested vectors (`std::vector<std::vector<float>>`), which scatter data across the heap, destroy CPU cache locality, and lack linear algebra operations. 

MiniTensor solves this by providing a unified, cache-friendly `Tensor<T>` container that stores all numbers in a single contiguous block of memory while allowing intuitive N-dimensional coordinate indexing and mathematical operations.

### Why is it Called a "Mini" Tensor Library?
Unlike massive enterprise frameworks like PyTorch, TensorFlow, or Eigen (which contain hundreds of thousands of lines of code, GPU compilers, and hundreds of dependencies), MiniTensor is **intentionally minimal, transparent, and completely self-contained**:
- **Zero Dependencies:** Written from scratch using only standard C++20.
- **Header-Only:** No complex binary compilation required; simply drop the headers into your project.
- **Readable and Maintainable:** The entire core implementation is contained in concise, documented headers that anyone can read and understand in an afternoon.

### Quick Example: What Can You Accomplish?
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    // 1. Create a 2x3 matrix filled with values
    minitensor::Tensor<float> a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // 2. Multidimensional coordinate access
    a(1, 2) = 10.0f; // Row 1, Column 2

    // 3. Reshape and Transpose
    auto reshaped = a.reshape({3, 2});
    auto transposed = a.transpose();

    // 4. Element-wise arithmetic
    auto doubled = a + a;

    // 5. High-performance matrix multiplication: (2x3) * (3x2) -> (2x2)
    minitensor::Tensor<float> b({3, 2}, 2.0f);
    auto product = a.matmul(b);

    std::cout << "Result matrix shape: {" << product.get_shape()[0] 
              << ", " << product.get_shape()[1] << "}\n";
    std::cout << "Top-left product value: " << product(0, 0) << '\n';
}
```

---

## Section B: Understanding Tensors From Absolute Zero

If you have never encountered the word **tensor** before, do not worry! Let us start from the absolute ground up.

### 1. What is Data?
In computer programs, data begins as simple numbers:
- A single number: `25.4` (a temperature measurement).
- A list of numbers: `[22.1, 23.4, 25.4, 21.0]` (hourly temperatures throughout an afternoon).
- A table of numbers: An $8 \times 8$ grid of numbers representing temperature sensors across a farm.
- A collection of tables: An hourly record of that farm grid over 24 hours.

### 2. What is a Tensor?
A **tensor** is simply the universal mathematical name for a collection of numbers arranged in any number of dimensions:

```text
Rank 0: Scalar         Rank 1: Vector         Rank 2: Matrix         Rank 3: 3D Tensor
 (A single point)       (A 1D line)            (A 2D grid/table)      (A stack of tables)

      42.0             [ 1.0, 2.0, 3.0 ]       +-----+-----+          +-----+-----+
                                               | 1.0 | 2.0 |         / 1.0 / 2.0 /|
                                               +-----+-----+        +-----+-----+ |
                                               | 3.0 | 4.0 |        | 3.0 | 4.0 |/
                                               +-----+-----+        +-----+-----+
     0 Axes                 1 Axis                 2 Axes                 3 Axes
```

- **Scalar (Rank 0):** A single standalone number.
- **Vector (Rank 1):** A one-dimensional list of numbers along a single axis.
- **Matrix (Rank 2):** A two-dimensional table with rows and columns.
- **3D Tensor (Rank 3):** A volume or stack of matrices (e.g., a color picture with Height, Width, and 3 color channels).
- **Higher-Dimensional Tensor (Rank 4+):** Data with 4 or more axes (e.g., a batch of video clips: `[Batch, Time, Height, Width, Channels]`).

### 3. Tensor Terminology Demystified

| Term | Plain-English Meaning | Concrete Example |
| :--- | :--- | :--- |
| **Element** | An individual number stored inside the tensor. | The number `5.0` at row 1, column 0. |
| **Dimension** | The length or size along one specific direction. | A table having `3` columns. |
| **Axis** | A coordinate direction (Axis 0 is outer, Axis 1 is next, etc.). | Axis 0 = Rows, Axis 1 = Columns. |
| **Rank** | The number of axes/dimensions. (Rank 2 = 2D matrix). | A tensor of shape `{2, 3}` has rank 2. |
| **Shape** | The list of dimensions describing the tensor's geometry. | Shape `{2, 3}` means 2 rows, 3 columns. |
| **Size** | The total count of all elements stored in the tensor. | For shape `{2, 3}`, size is $2 \times 3 = 6$. |
| **Index** | The multidimensional coordinate of an element. | `(1, 2)` specifies row 1, column 2. |
| **Offset** | The flat 1D memory position where the element is stored. | Element `(1, 2)` lives at offset `5` in flat memory. |
| **Stride** | How many memory steps to skip to advance by 1 in an axis. | Stride for row advance in a $2 \times 3$ matrix is `3`. |
| **Contiguous** | Storing all numbers back-to-back in an unbroken line of RAM. | `[1, 2, 3, 4, 5, 6]` in one memory buffer. |
| **Row-Major** | Storing elements row-by-row sequentially in memory. | Row 0 is stored first, followed immediately by Row 1. |

> [!NOTE]
> **Tensor Rank vs. Matrix Rank:**
> In linear algebra, the "rank of a matrix" refers to the number of linearly independent rows or columns. In tensor computing (PyTorch, NumPy, MiniTensor), **rank** simply means **the number of dimensions** (`shape.size()`).

### 4. Understanding Shape
The shape determines the organization of the tensor. Total element size is calculated by multiplying all dimensions:
$$\text{Total Size} = \text{Dimension}_0 \times \text{Dimension}_1 \times \dots \times \text{Dimension}_{k-1}$$

| Geometry | Shape (`get_shape()`) | Rank (`rank()`) | Total Size (`size()`) |
| :--- | :---: | :---: | :---: |
| **Scalar** | `{}` | 0 | 1 |
| **Vector** | `{4}` | 1 | 4 |
| **Matrix** | `{2, 3}` | 2 | $2 \times 3 = 6$ |
| **3D Volume** | `{2, 3, 4}` | 3 | $2 \times 3 \times 4 = 24$ |
| **Zero-Sized Tensor** | `{2, 0, 3}` | 3 | $2 \times 0 \times 3 = 0$ |

- **Scalar Semantics:** A rank-0 scalar has shape `{}` and holds exactly 1 element.
- **Zero-Sized Semantics:** If any dimension is 0, the total size is 0 and no buffer storage is allocated.

### 5. Why Tensors Instead of Ordinary Arrays?
Why not just use `std::vector<std::vector<float>>`?
1. **Cache Misses:** Nested vectors store pointers to separate heap allocations. Reading row by row requires "pointer chasing," causing constant CPU cache misses.
2. **Memory Fragmentation:** Every inner vector allocates its own heap control block, adding memory overhead.
3. **No Uniform Geometry:** Nested vectors can become "ragged" (e.g. Row 0 having 3 items, Row 1 having 5 items). Tensors enforce rectangular geometry.
4. **Compile-Time Limits:** In nested vectors, the rank is hardcoded in the C++ type (`vector<vector<T>>` is 2D; `vector<vector<vector<T>>>` is 3D). MiniTensor allows runtime arbitrary ranks.

For an extensive conceptual guide, read [**docs/tensor_fundamentals.md**](docs/tensor_fundamentals.md).

---

## Section C: How Tensors Work Inside This Library

### 1. Tensor Representation
Inside `include/minitensor/tensor.hpp`, the `Tensor<T>` class template encapsulates three private members:
```cpp
template <typename T>
class Tensor {
    // ...
private:
    std::vector<std::size_t> shape_;   // Dimensions: e.g. {2, 3}
    std::vector<std::size_t> strides_; // Row-major strides: e.g. {3, 1}
    std::vector<T> data_;              // Flat contiguous element buffer
};
```
- `T`: The payload data type (`float`, `double`, `int`, etc.).
- `shape_`: Vector storing the size of each axis.
- `strides_`: Precalculated jump distances for fast coordinate translation.
- `data_`: The single contiguous array holding all elements.

```text
Stack: minitensor::Tensor<float>
+-----------------------------------+
| shape_   = { 2, 3 }               |
| strides_ = { 3, 1 }               |
| data_    = pointer to heap buffer |
+-----------------------------------+
                  |
Heap:             v
+-------+-------+-------+-------+-------+-------+
|  1.0  |  2.0  |  3.0  |  4.0  |  5.0  |  6.0  |
+-------+-------+-------+-------+-------+-------+
 Offset 0  Offset 1 Offset 2 Offset 3 Offset 4 Offset 5
 (Row 0, Column 0..2)    (Row 1, Column 0..2)
```

### 2. Contiguous Memory and Row-Major Order
MiniTensor stores all elements in **Row-Major Order** (C/C++ convention).
A matrix:
$$\begin{pmatrix} 1.0 & 2.0 & 3.0 \\ 4.0 & 5.0 & 6.0 \end{pmatrix}$$
is physically stored in memory as the flat sequence:
`[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`.

Because the elements are contiguous, CPU hardware prefetchers stream consecutive floats into L1 cache at full memory bus bandwidth.

### 3. Memory Safety and RAII
MiniTensor follows strict **RAII (Resource Acquisition Is Initialization)**:
- Memory is acquired upon construction and freed automatically when the tensor goes out of scope.
- **Copying:** Generates an independent deep copy with its own buffer. Modifying a copy never affects the original.
- **Moving:** Uses C++ move semantics to transfer ownership of internal vector pointers in $O(1)$ time with zero memory allocations.
- **Strong Exception Guarantee:** In methods like `set_data()`, the copy-and-swap idiom ensures that if an error occurs, the original tensor remains 100% uncorrupted.

For more details, see [**docs/memory_management.md**](docs/memory_management.md) and [**docs/tensor_internals.md**](docs/tensor_internals.md).

### 4. Strides and Multidimensional Coordinate Indexing
To translate an N-dimensional coordinate $(i_0, i_1, \dots, i_{k-1})$ into a 1D memory offset:
$$\text{Offset} = \sum_{\text{axis}=0}^{k-1} \text{indices}[\text{axis}] \times \text{strides}[\text{axis}]$$

For shape `{2, 3}`:
- Stride for Axis 0 (rows) = `3` (jumping 1 row skips 3 columns).
- Stride for Axis 1 (columns) = `1` (adjacent in memory).
- To access `(1, 2)`: $\text{Offset} = (1 \times 3) + (2 \times 1) = 5$.
- `data_[5]` is returned!

**Bounds and Rank Checking:**
- If the number of coordinates passed does not equal `rank()`, it throws `std::invalid_argument`.
- If any coordinate is out of bounds ($\ge \text{shape}[\text{axis}]$), it throws `std::out_of_range`.

---

## Section D: Complete Library API Reference Overview

Below is a reference summary of the public API of `minitensor::Tensor<T>`. For full method signatures, parameter descriptions, exceptions, and code examples, see [**docs/api_reference.md**](docs/api_reference.md).

| Category | Method / Operator | Description |
| :--- | :--- | :--- |
| **Constructors** | `Tensor()` | Constructs a rank-0 scalar tensor with 1 value. |
| | `explicit Tensor(shape)` | Constructs a tensor with given shape, zero-initialized. |
| | `Tensor(shape, initial_value)` | Constructs a tensor filled with `initial_value`. |
| | `Tensor(shape, initial_data)` | Constructs a tensor populated from an existing `std::vector<T>`. |
| | `Tensor(const Tensor&)` | Deep copy constructor (allocates independent storage). |
| | `Tensor(Tensor&&) noexcept` | Move constructor ($O(1)$ pointer transfer). |
| **Introspection** | `size()` | Returns total element count (`std::size_t`). |
| | `rank()` | Returns number of dimensions (`std::size_t`). |
| | `get_shape()` | Returns `const std::vector<std::size_t>&`. |
| | `get_strides()` | Returns `const std::vector<std::size_t>&`. |
| | `get_data()` | Returns `const std::vector<T>&` (direct flat buffer access). |
| **Modification** | `fill(value)` | Sets every element in the tensor to `value`. |
| | `set_data(values)` | Replaces buffer (validates size, strong exception guarantee). |
| **Indexing** | `at(indices)` | Bounds-checked element access (`vector` or `initializer_list`). |
| | `set(indices, value)` | Sets element at `indices` to `value`. |
| | `operator()()` | Accesses the value of a rank-0 scalar tensor. |
| | `operator()(indices...)` | Variadic coordinate access: `tensor(row, col)`. |
| **Arithmetic** | `operator+` | Element-wise addition (identical shapes required). |
| | `operator-` | Element-wise subtraction (identical shapes required). |
| | `operator*` | Element-wise multiplication (Hadamard product). |
| **Linear Algebra**| `matmul(other)` | Serial 2D matrix multiplication (`i-k-j` cache-optimized). |
| | `matmul_blocked(other, B)` | Cache-tiled matrix multiplication (sub-block size $B$, default 64). |
| | `matmul_parallel(other, B)`| Multithreaded OpenMP matrix multiplication. |
| **Geometry** | `reshape(new_shape)` | Creates a new owning tensor with new shape (same element count). |
| | `transpose()` | Swaps rows and columns of a 2D matrix. |

---

## Section E: Understanding Mathematical Operations

### Element-Wise Operations vs. Linear Algebra
- `A * B` is **element-wise multiplication** (Hadamard product). Every number in $A$ is multiplied only by the corresponding number in $B$. Both tensors must have identical shapes.
- `A.matmul(B)` is **matrix multiplication**. Rows of $A$ are multiplied by columns of $B$.

> [!IMPORTANT]
> **No Implicit Broadcasting:** MiniTensor strictly enforces exact shape equality for element-wise operations. It does not perform implicit broadcasting.

### Matrix Multiplication (`matmul`) and the `i-k-j` Optimization
Matrix multiplication requires both tensors to be **Rank 2 (matrices)**, and the inner dimensions must match:
$$\underbrace{(M \times K)}_{A} \times \underbrace{(K \times N)}_{B} = \underbrace{(M \times N)}_{C}$$

MiniTensor orders loops as `row -> inner -> col` (`i-k-j`):
```cpp
for (size_t row = 0; row < M; ++row) {
    for (size_t inner = 0; inner < K; ++inner) {
        const T left = A[row * K + inner]; // Kept in CPU register!
        for (size_t col = 0; col < N; ++col) {
            C[row * N + col] += left * B[inner * N + col]; // Unit-stride memory streaming!
        }
    }
}
```
This avoids vertical column traversals and maximizes cache line hits.

### Cache-Blocked Matrix Multiplication (`matmul_blocked`)
For large matrices that exceed L1/L2 cache capacity, `matmul_blocked(other, block_size)` tiles the computation into sub-matrices of size $B \times B$ (default 64). Data in each tile stays cache-hot during calculation.

### Parallel Matrix Multiplication (`matmul_parallel`)
When compiled with OpenMP (`MINITENSOR_ENABLE_OPENMP=ON`), `matmul_parallel()` distributes output rows across CPU cores using `#pragma omp parallel for`. Small workloads ($< 64 \times 64 \times 64$) run serially to avoid thread spawn overhead. If OpenMP is disabled, it gracefully falls back to serial `matmul()`.

### Reshaping and Transposing
- `reshape(new_shape)`: Reorganizes dimensions while preserving row-major element order. Returns a new owning tensor.
- `transpose()`: Swaps rows and columns for rank-2 matrices ($(M \times N) \rightarrow (N \times M)$).

### Complexity Summary
| Operation | Time Complexity | Extra Space Complexity | Restrictions |
| :--- | :---: | :---: | :--- |
| `+`, `-`, `*` | $O(N)$ | $O(N)$ | Identical shapes required |
| `matmul` | $O(M \times K \times N)$ | $O(M \times N)$ | Rank 2 only; inner dimensions must match |
| `matmul_blocked` | $O(M \times K \times N)$ | $O(M \times N)$ | Rank 2 only; `block_size > 0` |
| `matmul_parallel`| $O(\frac{M \times K \times N}{P})$ | $O(M \times N)$ | Rank 2 only; $P$ threads |
| `reshape` | $O(N)$ | $O(N)$ | Element count must match |
| `transpose` | $O(M \times N)$ | $O(M \times N)$ | Rank 2 only |

For worked numerical walkthroughs and tiling diagrams, see [**docs/mathematical_operations.md**](docs/mathematical_operations.md).

---

## Section F: Getting Started: Installation and Setup

### Prerequisites
- **Compiler:** Any standard C++20 compiler:
  - GCC 10+ (Ubuntu 20.04+, Debian 11+)
  - Clang 11+
  - MSVC 19.29+ (Visual Studio 2019 version 16.11 or Visual Studio 2022)
- **Build System:** CMake 3.16 or newer.
- **Optional Dependencies:**
  - OpenMP (for multithreaded matrix multiplication).
  - GoogleTest (for expanded test discovery; fallback test runner included).

### Building the Project

```sh
# 1. Clone the repository
git clone https://github.com/your-org/minitensor.git
cd minitensor

# 2. Configure build with CMake (Release mode recommended for performance)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 3. Build examples, tests, and benchmarks
cmake --build build --config Release
```

#### Optional CMake Build Flags
| Flag | Default | Purpose |
| :--- | :---: | :--- |
| `-DMINITENSOR_ENABLE_OPENMP=ON` | `OFF` | Enables multithreaded OpenMP matrix multiplication. |
| `-DMINITENSOR_ENABLE_NATIVE_OPTIMIZATIONS=ON` | `OFF` | Enables `/arch:AVX2` (MSVC) or `-march=native` (GCC/Clang). |
| `-DMINITENSOR_ENABLE_SANITIZERS=ON` | `OFF` | Enables AddressSanitizer and UndefinedBehaviorSanitizer. |
| `-DMINITENSOR_BUILD_EXAMPLES=OFF` | `ON` | Disables building example programs. |
| `-DMINITENSOR_BUILD_BENCHMARKS=OFF` | `ON` | Disables building the benchmark executable. |
| `-DBUILD_TESTING=OFF` | `ON` | Disables building the test suite. |

### Running the Tests
```sh
# Via CTest:
ctest --test-dir build -C Release --output-on-failure

# Or run the test executable directly:
# Windows:
build\Release\test_tensor.exe
# Linux / macOS:
./build/test_tensor
```

### Running the Examples
```sh
# Basic tensor introduction:
build\Release\basic_tensor.exe      # Windows
./build/basic_tensor                # Linux

# Image processing demonstration:
build\Release\image_processing.exe  # Windows
./build/image_processing            # Linux
```

### Running the Benchmarks
```sh
build\Release\minitensor_benchmark.exe --iterations 3 --max-size 512
```

### Integrating MiniTensor into Downstream CMake Projects

#### 1. Install MiniTensor to a prefix directory:
```sh
cmake --install build --prefix install
```

#### 2. In your own project's `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.16)
project(MyAwesomeApp LANGUAGES CXX)

find_package(MiniTensor CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE MiniTensor::minitensor)
```

#### 3. In your C++ source code:
```cpp
#include <minitensor/minitensor.hpp>

int main() {
    minitensor::Tensor<float> tensor({3, 3}, 1.0f);
    return 0;
}
```

For full architecture and integration instructions, see [**docs/architecture.md**](docs/architecture.md).

---

## Section G: Step-by-Step Usage Tutorial

MiniTensor includes a progressive, 15-step tutorial taking you from zero to building an end-to-end image processing pipeline:

1. **Include the Library:** Using `<minitensor/minitensor.hpp>`.
2. **Basic Construction:** Zero-initialized, fill-initialized, and vector-initialized tensors.
3. **Scalars (Rank 0):** Managing dimensionless numbers with `operator()()`.
4. **Vectors (Rank 1):** 1D sequence manipulation.
5. **Matrices (Rank 2):** 2D tables, rows, and columns.
6. **3D Volumes (Rank 3):** Stacks of matrices and color image representations.
7. **Inspecting Metadata:** Querying `shape()`, `rank()`, `size()`, and `strides()`.
8. **Bulk Modification:** Using `fill()` and `set_data()`.
9. **Coordinate Access:** `at()`, `set()`, and variadic `tensor(i, j, k)`.
10. **Addition and Subtraction:** Element-wise arithmetic.
11. **Hadamard Multiplication:** Element-wise `operator*`.
12. **Matrix Multiplication:** Calculating row-column dot products via `matmul()`.
13. **Reshaping:** Changing tensor geometry while preserving row-major data.
14. **Transposition:** Swapping rows and columns of matrices.
15. **Full Pipeline:** Constructing a synthetic RGB image tensor, brightening pixels, scaling intensities, and swapping color channels (RGB to BGR).

To follow the complete tutorial with compilable code samples and expected outputs, open [**docs/usage_tutorial.md**](docs/usage_tutorial.md).

---

## Section H: Understanding the Repository Structure

```text
minitensor/
├── CMakeLists.txt                 # Master CMake build configuration
├── LICENSE                        # MIT License
├── README.md                      # Primary project manual (this file)
├── cmake/
│   └── MiniTensorConfig.cmake.in  # Package configuration template for find_package()
├── docs/                          # Detailed technical documentation guides
├── include/minitensor/            # Core library headers
│   ├── indexing.hpp               # Coordinate access utilities
│   ├── linalg.hpp                 # Linear algebra and matmul kernels
│   ├── minitensor.hpp             # Master umbrella header
│   ├── operations.hpp             # Element-wise operations
│   └── tensor.hpp                 # Primary Tensor<T> class template
├── examples/
│   ├── basic_tensor.cpp           # Fundamentals: construction, metadata, shapes
│   └── image_processing.cpp       # 3D RGB image manipulation pipeline
├── benchmarks/
│   └── benchmark.cpp              # Matrix multiplication microbenchmark
└── tests/
    ├── test_tensor.cpp            # Comprehensive test suite (GoogleTest / standalone)
    └── package_consumer/          # Verification project for find_package() integration
```

For file-by-file responsibilities and CMake target relationships, see [**docs/architecture.md**](docs/architecture.md).

---

## Section I: Performance, Optimization, and Benchmarks

### Cache Locality and Loop Design
MiniTensor stores data contiguously and accesses elements along row-major strides. By structuring matrix multiplication as an `i-k-j` loop, inner loops stream data sequentially across memory lines, enabling CPU cache hits and compiler auto-vectorization.

### Empirical Benchmark Measurements
Below are actual measured benchmark results from real runs executed on Windows x64 with MSVC 19.51 (Release build `/O2 /Ob2 /DNDEBUG`, block size 64):

#### Baseline Serial vs. OpenMP (16 Threads)
| Dimension | Serial (ms) | Blocked (ms) | OpenMP Parallel (ms) | Parallel Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **64** | 0.226 ms | 0.185 ms | 0.041 ms | **5.49x** |
| **256** | 6.899 ms | 6.977 ms | 1.281 ms | **5.39x** |

#### Standard Reference Benchmark (4 Threads)
| Dimension | Serial (ms) | Blocked (ms) | OpenMP (ms) | OpenMP Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **64** | 0.057 ms | 0.067 ms | 0.050 ms | 1.130x |
| **256** | 3.865 ms | 3.839 ms | 2.381 ms | 1.623x |
| **512** | 34.627 ms | 33.118 ms | 24.499 ms | 1.413x |
| **1024** | 238.526 ms | 257.955 ms | 308.166 ms | 0.774x |

*Speedups are hardware- and workload-dependent. At 1024x1024, memory bus saturation across concurrent cores can bottleneck multithreaded speedup.*

To read the in-depth hardware and memory hierarchy analysis, see [**docs/performance.md**](docs/performance.md).

---

## Section J: Testing, Edge Cases, and Reliability

MiniTensor includes tests for critical edge cases:
- **Overflow Protection:** Shape products that exceed `std::size_t` throw `std::length_error`.
- **Zero-Sized Tensors:** Tensors with a zero dimension (e.g. `{2, 0, 3}`) safely produce size 0 with zero-valued strides.
- **Index Bounds Protection:** Coordinates that exceed dimension bounds throw `std::out_of_range`.
- **Shape Mismatch:** Incompatible shapes in addition or matrix multiplication throw `std::invalid_argument`.
- **Memory Safety:** Deep copy independence and move semantics are verified to ensure zero memory leaks.

For complete test suite details and sanitizer configuration, see [**docs/testing.md**](docs/testing.md).

---

## Section K: Practical Applications

Where are tensors used in real software engineering?
1. **Computer Vision & Image Processing:**
   - Digital images are rank-3 tensors of shape `{Height, Width, Channels}` (e.g., Channels = 3 for RGB).
   - Video clips are rank-4 tensors: `{Frames, Height, Width, Channels}`.
   - MiniTensor easily supports channel manipulation, pixel adjustments, and intensity scaling (see `examples/image_processing.cpp`).
2. **Machine Learning and Neural Networks:**
   - Dense layers perform $Y = X \cdot W + B$ (matrix multiplication followed by tensor addition).
   - MiniTensor provides the core contiguous data representation and matrix multiplication building blocks.
3. **Physics Simulations and Scientific Computing:**
   - Modeling heat transfer across a 2D plate or 3D volume over time.
   - Coordinate transformation matrices (e.g. rotating 3D vectors via $3 \times 3$ matrix multiplication).

---

## Section L: Limitations and Future Roadmap

MiniTensor is intentionally minimal. To maintain technical transparency, here is what the library **currently supports** versus what is **unsupported**:

### Current Capabilities (Supported)
- Arbitrary-rank owning tensors.
- Contiguous row-major memory with checked strides.
- Bounds-checked multidimensional coordinate access.
- Value semantics (deep copy, $O(1)$ moves, RAII).
- Element-wise addition, subtraction, and multiplication (Hadamard product).
- 2D matrix multiplication (serial, cache-blocked, and OpenMP parallel).
- 2D matrix transposition and copying reshape.
- Comprehensive CMake export and install targets.

### Current Limitations (Unsupported)
- **CPU Only:** No GPU (CUDA / ROCm / Vulkan) support.
- **No Implicit Broadcasting:** Operands must have identical shapes.
- **2D-Only Matrix Multiplication:** Batched higher-rank matrix multiplication (e.g., $(B \times M \times K) \times (B \times K \times N)$) is not currently supported.
- **2D-Only Transposition:** Transposing higher-rank tensors by axis permutation is not implemented.
- **No Shared Views / Slicing:** Sub-tensor slicing and non-owning views (`std::span` style) are not implemented; `reshape()` and `transpose()` return newly allocated owning copies.
- **No Automatic Differentiation:** No autograd or computational graph engine.

### Potential Future Roadmap
- Safe, non-owning tensor views and strided slicing.
- Batched matrix multiplication for 3D/4D tensors.
- Arbitrary axis permutation / multidimensional transpose.
- Reduction operations (`sum`, `mean`, `max`, `min` along specified axes).
- Explicit broadcasting utilities.

---

## License

MiniTensor is licensed under the permissive **MIT License**. See [LICENSE](LICENSE) for details.
