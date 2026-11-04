# Testing, Reliability, and Edge Case Verification in MiniTensor

A mathematical library is only as good as its reliability. When doing scientific simulations or machine learning, an undetected off-by-one index error or a silent memory corruption bug can lead to invalid results or catastrophic crashes.

MiniTensor is backed by a comprehensive automated test suite designed to verify mathematical correctness, memory safety, exception guarantees, and boundary edge cases.

This document details the testing architecture, how each test suite operates, how edge cases are caught, and how to run tests locally with sanitizers.

---

## Table of Contents

1. [Testing Architecture](#1-testing-architecture)
   - [Dual Testing Modes: GoogleTest and Standalone Fallback](#dual-testing-modes-googletest-and-standalone-fallback)
   - [The External Package Consumer Test](#the-external-package-consumer-test)
2. [Detailed Test Suite Breakdown](#2-detailed-test-suite-breakdown)
   - [1. `CreationTypesAndShapes`](#1-creationtypesandshapes)
   - [2. `ScalarAndZeroSizedTensors`](#2-scalarandzerosizedtensors)
   - [3. `InitializationAndFill`](#3-initializationandfill)
   - [4. `InvalidDataSizes`](#4-invaliddatasizes)
   - [5. `SizeOverflow`](#5-sizeoverflow)
   - [6. `CopyAndIndependentStorage`](#6-copyandindependentstorage)
   - [7. `MoveConstructionAndAssignment`](#7-moveconstructionandassignment)
   - [8. `StridesAndIndexing`](#8-stridesandindexing)
   - [9. `ElementwiseOperations`](#9-elementwiseoperations)
   - [10. `MatrixMultiplication`](#10-matrixmultiplication)
   - [11. `ReshapeAndTranspose`](#11-reshapeandtranspose)
   - [12. `FloatingPointOperations`](#12-floatingpointoperations)
3. [Edge Cases and How They Are Handled](#3-edge-cases-and-how-they-are-handled)
   - [Exception Hierarchy and Failure Modes](#exception-hierarchy-and-failure-modes)
4. [Running the Tests](#4-running-the-tests)
   - [Running via CTest](#running-via-ctest)
   - [Running the Test Executable Directly](#running-the-test-executable-directly)
5. [Memory Safety and Sanitizer Support](#5-memory-safety-and-sanitizer-support)

---

## 1. Testing Architecture

### Dual Testing Modes: GoogleTest and Standalone Fallback
MiniTensor supports two testing modes via the CMake option `MINITENSOR_USE_GTEST` (default `ON`):
1. **GoogleTest Mode:** If GoogleTest is installed or discovered on your system (`find_package(GTest)`), the test executable compiles with `MINITENSOR_USE_GTEST` and registers individual `TEST()` cases that report rich test statistics and discoverable targets via `gtest_discover_tests`.
2. **Standalone Fallback Mode:** If GoogleTest is not installed, MiniTensor automatically compiles with its built-in, dependency-free test runner using the `require()` and `require_throws<E>()` assertion engine. It runs the exact same test suites with zero external dependencies!

### The External Package Consumer Test
Located in `tests/package_consumer/`, this sub-project tests the exported CMake package:
- It runs `find_package(MiniTensor CONFIG REQUIRED)`.
- It links against `MiniTensor::minitensor`.
- It compiles a real consumer executable to ensure downstream integration works without manual header path adjustments.

---

## 2. Detailed Test Suite Breakdown

Located in `tests/test_tensor.cpp`, the test suite covers 12 focused areas:

### 1. `CreationTypesAndShapes`
- Tests rank 1 (`{4}`), rank 2 (`{2, 3}`), and rank 3 (`{2, 3, 4}`) tensors.
- Verifies that `size()` equals the product of dimensions.
- Verifies that `rank()` equals the number of shape elements.
- Verifies across multiple data types: `Tensor<int>`, `Tensor<float>`, and `Tensor<double>`.

### 2. `ScalarAndZeroSizedTensors`
- Verifies default constructor produces rank 0, size 1, empty shape `{}`, and empty strides `{}`.
- Verifies scalar zero-index access `scalar() == 0.0`.
- Verifies zero-sized tensors (e.g., shape `{2, 0, 3}`):
  - Produces rank 3 with total size 0.
  - Strides are safely initialized to `{0, 0, 0}`.
  - `fill()` and `set_data({})` succeed without memory faults.

### 3. `InitializationAndFill`
- Verifies default value initialization (elements default to zero).
- Verifies the constant fill constructor: `Tensor({2, 2}, 1.5f)`.
- Verifies the in-place `fill()` method.
- Verifies initial vector loading constructor: `Tensor(shape, vector)`.
- Verifies full buffer replacement via `set_data()`.

### 4. `InvalidDataSizes`
- Verifies that passing an initial data vector whose size does not match `calculate_size(shape)` throws `std::invalid_argument`.
- Verifies that calling `set_data()` with mismatched elements throws `std::invalid_argument` and leaves the original tensor contents completely untouched (strong exception guarantee).

### 5. `SizeOverflow`
- Passes dimension values close to `std::numeric_limits<std::size_t>::max()`.
- Verifies that `calculate_size()` catches integer multiplication overflow and throws `std::length_error`.
- Verifies that a zero dimension with overflowing companion dimensions (e.g. `{max, max, 0}`) evaluates safely to size 0 without throwing overflow.

### 6. `CopyAndIndependentStorage`
- Verifies deep copy construction: mutating a copied tensor does not affect the original.
- Verifies deep copy assignment (`tensor_b = tensor_a`): storage remains independent.

### 7. `MoveConstructionAndAssignment`
- Verifies move construction (`Tensor b(std::move(a))`): transfers data buffer and shape metadata without memory leaks.
- Verifies move assignment (`b = std::move(a)`).

### 8. `StridesAndIndexing`
- Verifies row-major strides:
  - 1D `{4}` -> strides `{1}`
  - 2D `{2, 3}` -> strides `{3, 1}`
  - 3D `{2, 3, 4}` -> strides `{12, 4, 1}`
- Verifies multidimensional read/write via `at({i, j, k})`, `set({i, j, k}, val)`, and `(i, j, k)`.
- Verifies const-correctness of element access.
- Verifies error paths:
  - Passing too few or too many indices throws `std::invalid_argument`.
  - Out-of-bounds coordinate throws `std::out_of_range`.
  - Calling `at({0})` on a rank-0 scalar throws `std::invalid_argument`.
  - Indexing an empty dimension throws `std::out_of_range`.

### 9. `ElementwiseOperations`
- Verifies addition (`+`), subtraction (`-`), and element-wise multiplication (`*`).
- Verifies that supplying tensors with mismatched shapes throws `std::invalid_argument`.

### 10. `MatrixMultiplication`
- Verifies serial multiplication (`matmul`), cache-blocked multiplication (`matmul_blocked`), and parallel fallback (`matmul_parallel`).
- Verifies non-square matrices: $(2 \times 3) \times (3 \times 2) \rightarrow (2 \times 2)$.
- Verifies zero-sized inner dimensions: $(2 \times 0) \times (0 \times 3) \rightarrow (2 \times 3)$ filled with zeros.
- Verifies error paths:
  - Mismatched inner dimensions (e.g. $(2 \times 2) \times (3 \times 2)$) throws `std::invalid_argument`.
  - Non-matrix operands (e.g. 1D vectors) throws `std::invalid_argument`.
  - Zero block size in `matmul_blocked(other, 0)` throws `std::invalid_argument`.

### 11. `ReshapeAndTranspose`
- Verifies that `reshape()` preserves row-major order and recalculates strides.
- Verifies that reshaping with an unequal element count throws `std::invalid_argument`.
- Verifies scalar reshape to `{1}`.
- Verifies 2D matrix transposition and row/column swap.
- Verifies that calling `transpose()` on non-2D tensors throws `std::invalid_argument`.

### 12. `FloatingPointOperations`
- Multiplies floating-point matrices with fractional numbers (`0.1`, `0.2`, `0.3`, `0.4`).
- Asserts that all results match expected double-precision values within a strict tolerance of $10^{-12}$.

---

## 3. Edge Cases and How They Are Handled

MiniTensor uses standard C++ exception types to report errors clearly:

| Error Condition | Thrown Exception Type | Error Message |
| :--- | :--- | :--- |
| **Shape product overflows `size_t`** | `std::length_error` | `"Tensor shape product overflows size_t"` |
| **Stride overflows `size_t`** | `std::length_error` | `"Tensor stride overflows size_t"` |
| **Initial data size != shape size** | `std::invalid_argument` | `"Initial data size must match the tensor shape"` |
| **`set_data()` size != tensor size**| `std::invalid_argument` | `"Replacement data size must match the tensor size"` |
| **Indices count != tensor rank** | `std::invalid_argument` | `"Number of indices must match tensor rank"` |
| **Coordinate exceeds dimension** | `std::out_of_range` | `"Tensor index is out of bounds"` |
| **Element-wise shape mismatch** | `std::invalid_argument` | `"Element-wise operations require identical tensor shapes"` |
| **Matmul on non-2D tensors** | `std::invalid_argument` | `"Matrix multiplication requires two-dimensional tensors"` |
| **Matmul inner dimensions mismatch** | `std::invalid_argument` | `"Matrix multiplication inner dimensions must match"` |
| **Matmul block size is 0** | `std::invalid_argument` | `"Matrix multiplication block size must be positive"` |
| **Reshape size != current size** | `std::invalid_argument` | `"Reshaped tensor must have the same number of elements"` |
| **Transpose on non-2D tensor** | `std::invalid_argument` | `"Transpose is supported only for two-dimensional tensors"` |

---

## 4. Running the Tests

### Running via CTest
CTest is the standard test runner integrated with CMake:

```sh
# On Linux / macOS:
ctest --test-dir build --output-on-failure

# On Windows (Multi-Configuration Generators like MSVC):
ctest --test-dir build -C Release --output-on-failure
```

### Running the Test Executable Directly
You can also invoke the compiled test binary directly to view formatted test output:

```sh
# On Windows:
build\Release\test_tensor.exe

# On Linux / macOS:
./build/test_tensor
```

Sample output:
```text
All MiniTensor tests passed.
```

---

## 5. Memory Safety and Sanitizer Support

MiniTensor supports automated memory and safety sanitizers via the CMake option `MINITENSOR_ENABLE_SANITIZERS=ON`:

```sh
cmake -S . -B build-sanitizers -DMINITENSOR_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitizers
ctest --test-dir build-sanitizers --output-on-failure
```

- On **GCC / Clang**, this enables:
  - `-fsanitize=address`: AddressSanitizer detects out-of-bounds heap/stack reads/writes and use-after-free bugs.
  - `-fsanitize=undefined`: UndefinedBehaviorSanitizer detects integer overflows, null-pointer dereferences, and alignment issues.
- On **MSVC**, this enables:
  - `/fsanitize=address`: Microsoft AddressSanitizer for Windows.

Running tests with sanitizers enabled confirms that MiniTensor contains zero memory leaks and zero undefined behaviors.

---

Proceed to:
- [Repository Architecture](architecture.md) for build options and directory structures.
- [API Reference](api_reference.md) for full method documentation.
- [Performance Guide](performance.md) for cache analysis and benchmarks.

