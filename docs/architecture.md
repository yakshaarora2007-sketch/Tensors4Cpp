# MiniTensor Repository Architecture and Directory Structure

This document details the software design, directory organization, build system architecture, and module responsibilities of the MiniTensor project.

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Complete Directory Tree](#2-complete-directory-tree)
3. [Component Responsibilities and Interactions](#3-component-responsibilities-and-interactions)
   - [Public Headers (`include/minitensor/`)](#public-headers-includeminitensor)
   - [Examples (`examples/`)](#examples-examples)
   - [Benchmarks (`benchmarks/`)](#benchmarks-benchmarks)
   - [Test Suites (`tests/`)](#test-suites-tests)
   - [Build Configuration and Package Export (`cmake/` and `CMakeLists.txt`)](#build-configuration-and-package-export-cmake-and-cmakeliststxt)
4. [CMake Target Architecture and Linkage](#4-cmake-target-architecture-and-linkage)
   - [Interface Library Design](#interface-library-design)
   - [Compiler Options and Flags](#compiler-options-and-flags)
   - [Install and CMake Package Export](#install-and-cmake-package-export)
5. [Integrating MiniTensor into External Projects](#5-integrating-minitensor-into-external-projects)
   - [Method A: `find_package` (Installed Package)](#method-a-find_package-installed-package)
   - [Method B: `FetchContent` or Git Submodule](#method-b-fetchcontent-or-git-submodule)
   - [Method C: Direct Header Copying (Single-Folder Inclusion)](#method-c-direct-header-copying-single-folder-inclusion)

---

## 1. Architectural Overview

MiniTensor is designed as a **modern, dependency-free, header-only C++20 library**. 

### Key Architectural Principles
- **Header-Only:** No static or shared library binaries (`.lib`, `.a`, `.dll`, `.so`) are needed to use MiniTensor. Simply include the headers and compile with a C++20 compiler.
- **Zero External Dependencies:** Built entirely with standard C++ containers (`std::vector`, `<algorithm>`, `<limits>`, `<stdexcept>`). No BLAS, LAPACK, Eigen, or Boost dependencies are required.
- **Clean Separation of Concerns:**
  - **Storage:** Contiguous heap buffer managed via `std::vector<T>`.
  - **Metadata:** Dynamic shape and precomputed row-major strides (`std::vector<std::size_t>`).
  - **Coordinate Translation:** Mathematical stride indexing engine mapping N-dimensional coordinates to 1D buffer offsets.
  - **Linear Algebra Kernels:** Algorithmic implementations decoupled into serial, cache-blocked, and multithreaded OpenMP execution paths.

---

## 2. Complete Directory Tree

```text
minitensor/
├── CMakeLists.txt                 # Master build and package configuration
├── LICENSE                        # MIT License
├── README.md                      # Primary project manual and user guide
├── cmake/
│   └── MiniTensorConfig.cmake.in  # CMake package config template for find_package()
├── docs/                          # Detailed technical documentation guides
│   ├── api_reference.md           # Complete signature-level API reference
│   ├── architecture.md            # Repository structure and design (this file)
│   ├── mathematical_operations.md # Mathematical algorithms, tiling, OpenMP, complexity
│   ├── memory_management.md       # RAII, contiguous storage, and exception safety
│   ├── mini_tensor_library_paper.md  # Formal academic research paper (Markdown)
│   ├── mini_tensor_library_paper.pdf # Publication-quality 16-page research paper (PDF)
│   ├── performance.md             # Cache locality, benchmarks, and profiling data
│   ├── testing.md                 # Test suites, assertions, and edge case verification
│   ├── tensor_fundamentals.md     # Beginner's zero-to-hero primer on tensors
│   ├── tensor_internals.md        # Class layout, calculate_strides, calculate_offset
│   └── usage_tutorial.md          # 15-step progressive coding tutorial
├── scripts/
│   └── generate_paper_pdf.py      # Reproducible generator for mini_tensor_library_paper.pdf
├── include/
│   └── minitensor/
│       ├── indexing.hpp           # Modular header for coordinate access
│       ├── linalg.hpp             # Modular header for linear algebra / matmul
│       ├── minitensor.hpp         # Primary umbrella header (includes all modules)
│       ├── operations.hpp         # Modular header for element-wise arithmetic
│       └── tensor.hpp             # Core Tensor<T> template class definition
├── examples/
│   ├── basic_tensor.cpp           # Fundamentals: construction, sizes, fills, scalars
│   └── image_processing.cpp       # Practical 3D RGB image manipulation pipeline
├── benchmarks/
│   └── benchmark.cpp              # Matrix multiplication benchmark (serial, blocked, parallel)
└── tests/
    ├── test_tensor.cpp            # Comprehensive unit and integration test suite
    └── package_consumer/          # Verification project testing installed package integration
        ├── CMakeLists.txt
        └── main.cpp
```

---

## 3. Component Responsibilities and Interactions

### Public Headers (`include/minitensor/`)

| File | Primary Responsibility | Key Interactions |
| :--- | :--- | :--- |
| `minitensor.hpp` | The primary entry point for user applications. | Includes `indexing.hpp`, `linalg.hpp`, `operations.hpp`, and `tensor.hpp`. |
| `tensor.hpp` | The core heart of the library. Implements `minitensor::Tensor<T>`: members `shape_`, `strides_`, `data_`, constructors, bounds checking, element-wise math, `matmul`, `reshape`, and `transpose`. | Direct dependency for all other modular headers. |
| `indexing.hpp` | Modular header reserved for coordinate and indexing utilities. | Includes `tensor.hpp`. |
| `operations.hpp` | Modular header reserved for element-wise arithmetic and unary operations. | Includes `tensor.hpp`. |
| `linalg.hpp` | Modular header reserved for linear algebra algorithms. | Includes `tensor.hpp`. |

### Examples (`examples/`)

- `basic_tensor.cpp`: Demonstrates basic usage: constructing 2D and 3D tensors, inspecting shapes, printing raw data, filling tensors, creating 0-rank scalars, and instantiating 0-sized tensors.
- `image_processing.cpp`: Demonstrates a complete real-world scenario: treating a rank-3 tensor `{Height, Width, 3}` as a synthetic digital color photograph, applying brightness adjustments, dimming via Hadamard product, and swapping color channels (RGB to BGR).

### Benchmarks (`benchmarks/`)

- `benchmark.cpp`: A microbenchmark evaluating square matrix multiplication across dimensions $64 \times 64$, $256 \times 256$, $512 \times 512$, and $1024 \times 1024$.
  - Compares three implementations: Baseline serial `matmul()`, cache-blocked `matmul_blocked()`, and multithreaded `matmul_parallel()`.
  - Validates numerical accuracy against baseline with floating-point tolerance ($10^{-4}$).
  - Accepts CLI flags: `--iterations`, `--max-size`, `--block-size`.

### Test Suites (`tests/`)

- `tests/test_tensor.cpp`: The primary unit test suite.
  - Can be built with GoogleTest (`MINITENSOR_USE_GTEST=ON`) or as a standalone executable with built-in assertion macros (`require`, `require_throws`).
  - Covers every method, constructor, edge case, and error path.
- `tests/package_consumer/`: An independent CMake consumer project used to verify that the exported `MiniTensorConfig.cmake` package works seamlessly with external `find_package()` calls.

### Build Configuration and Package Export (`cmake/` and `CMakeLists.txt`)

- `CMakeLists.txt`: Root configuration file setting C++20 compliance, defining the `minitensor` interface target, configuring compiler warnings (`/W4` on MSVC, `-Wall -Wextra -Wpedantic` on GCC/Clang), and defining install rules.
- `cmake/MiniTensorConfig.cmake.in`: Input template for `configure_package_config_file`, ensuring downstream projects automatically locate OpenMP dependencies if OpenMP was enabled during installation.

---

## 4. CMake Target Architecture and Linkage

### Interface Library Design
In `CMakeLists.txt`:
```cmake
add_library(minitensor INTERFACE)
add_library(MiniTensor::minitensor ALIAS minitensor)
target_include_directories(minitensor INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(minitensor INTERFACE cxx_std_20)
```
- `INTERFACE` indicates that MiniTensor has no compiled `.cpp` files of its own; it conveys header include paths and compiler features (`cxx_std_20`) transitively to whatever executable or library links against it.
- `MiniTensor::minitensor` provides a namespaced alias matching modern CMake best practices.

### Compiler Options and Flags
MiniTensor exposes several CMake options:

| Option | Default | Description |
| :--- | :---: | :--- |
| `MINITENSOR_BUILD_EXAMPLES` | `ON` | Builds the `basic_tensor` and `image_processing` executables. |
| `MINITENSOR_BUILD_BENCHMARKS` | `ON` | Builds the `minitensor_benchmark` executable. |
| `BUILD_TESTING` | `ON` | Enables building the `test_tensor` suite. |
| `MINITENSOR_USE_GTEST` | `ON` | Uses GoogleTest if found; falls back to standalone test main otherwise. |
| `MINITENSOR_ENABLE_OPENMP` | `OFF` | Enables OpenMP parallelization for `matmul_parallel()`. |
| `MINITENSOR_ENABLE_NATIVE_OPTIMIZATIONS` | `OFF` | Enables `-march=native` (GCC/Clang) or `/arch:AVX2` (MSVC). |
| `MINITENSOR_ENABLE_SANITIZERS` | `OFF` | Enables AddressSanitizer and UndefinedBehaviorSanitizer. |

### Install and CMake Package Export
When you run `cmake --install build --prefix <install_dir>`:
1. Headers are installed to `<prefix>/include/minitensor/`.
2. Documentation and license are installed to `<prefix>/share/doc/MiniTensor/`.
3. CMake package files are installed to `<prefix>/lib/cmake/MiniTensor/`:
   - `MiniTensorConfig.cmake`
   - `MiniTensorConfigVersion.cmake`
   - `MiniTensorTargets.cmake`

---

## 5. Integrating MiniTensor into External Projects

### Method A: `find_package` (Installed Package)
If MiniTensor has been installed to your system or a local directory:
```cmake
# In your CMakeLists.txt:
cmake_minimum_required(VERSION 3.16)
project(MyProject LANGUAGES CXX)

find_package(MiniTensor CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE MiniTensor::minitensor)
```
In your C++ code:
```cpp
#include <minitensor/minitensor.hpp>

int main() {
    minitensor::Tensor<float> tensor({4, 4}, 1.0f);
    return 0;
}
```

### Method B: `FetchContent` or Git Submodule
If you want CMake to automatically download or vendor MiniTensor:
```cmake
include(FetchContent)
FetchContent_Declare(
    MiniTensor
    GIT_REPOSITORY https://github.com/your-org/minitensor.git
    GIT_TAG main
)
FetchContent_MakeAvailable(MiniTensor)

target_link_libraries(my_app PRIVATE MiniTensor::minitensor)
```

### Method C: Direct Header Copying (Single-Folder Inclusion)
Because MiniTensor is header-only and has zero dependencies, you can simply copy the `include/minitensor` folder directly into your own project's include directory!
```sh
g++ -std=c++20 -Iinclude main.cpp -o main
```

---

Proceed to:
- [Performance and Optimization](performance.md) for cache and benchmark analysis.
- [Testing Guide](testing.md) for testing procedures and safety checks.
- [API Reference](api_reference.md) for method specifications.

