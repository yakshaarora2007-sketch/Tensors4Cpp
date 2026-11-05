# MiniTensor: Download, Release, and Integration Guide

This guide explains how to package MiniTensor for the **GitHub Releases tab**, how developers can **download** the library, and how to **use** it in any C++20 project.

---

## Table of Contents

1. [Distribution Overview](#1-distribution-overview)
2. [How to Publish MiniTensor on GitHub Releases Tab](#2-how-to-publish-minitensor-on-github-releases-tab)
   - [Step 1: Generate Release Assets](#step-1-generate-release-assets)
   - [Step 2: Upload to GitHub Releases](#step-2-upload-to-github-releases)
3. [How to Download and Integrate MiniTensor](#3-how-to-download-and-integrate-minitensor)
   - [Method 1: Single-Header Download (Simplest, 1-Click)](#method-1-single-header-download-simplest-1-click)
   - [Method 2: CMake `FetchContent` (Automated Dependency Management)](#method-2-cmake-fetchcontent-automated-dependency-management)
   - [Method 3: Download Release Zip Archive](#method-3-download-release-zip-archive)
   - [Method 4: System Installation via CMake `find_package`](#method-4-system-installation-via-cmake-find_package)
   - [Method 5: Git Submodule](#method-5-git-submodule)
4. [Using the Library: Code Examples Across All Ranks](#4-using-the-library-code-examples-across-all-ranks)
   - [1. Rank 1 Vector Dot Product](#1-rank-1-vector-dot-product)
   - [2. Rank 2 Matrix Multiplication](#2-rank-2-matrix-multiplication)
   - [3. Rank 3 Batched Matrix Multiplication](#3-rank-3-batched-matrix-multiplication)
   - [4. Rank 4 Multi-Batch Matrix Multiplication](#4-rank-4-multi-batch-matrix-multiplication)
   - [5. Batch Broadcasting (3D Batch x 2D Matrix)](#5-batch-broadcasting-3d-batch-x-2d-matrix)
   - [6. Arbitrary-Axis Tensor Contraction (`tensordot`)](#6-arbitrary-axis-tensor-contraction-tensordot)
   - [7. High-Performance Blocked & OpenMP Parallel Multiplication](#7-high-performance-blocked--openmp-parallel-multiplication)
5. [Compilation Commands](#5-compilation-commands)

---

## 1. Distribution Overview

MiniTensor is a **dependency-free, header-only C++20 library**. Because it requires no pre-compiled binaries, it can be distributed in two formats under your GitHub Release tab:

| Distribution Asset | Target Use Case | How to Use |
| :--- | :--- | :--- |
| **`minitensor.hpp`** *(Single-Header Amalgamation)* | Rapid prototyping, single-file inclusion, no CMake required. | Download 1 file, put in your project, `#include "minitensor.hpp"`. |
| **`minitensor-v1.1.0.zip`** *(Full Release Archive)* | Multi-file modular projects, CMake `add_subdirectory`, standard package installs. | Unzip, add `include/` to include paths or link `MiniTensor::minitensor`. |

---

## 2. How to Publish MiniTensor on GitHub Releases Tab

If you are the repository maintainer publishing a new version to GitHub, follow these simple steps:

### Step 1: Generate Release Assets

Run the included automated release bundler from the root of the repository:

```sh
python scripts/create_release_bundle.py
```

This generates two files in the `release/` directory:
- `release/minitensor.hpp` (the standalone, self-contained single-header distribution)
- `release/minitensor-v1.1.0.zip` (the complete release archive)

### Step 2: Upload to GitHub Releases

#### Option A: Using the GitHub Web Interface
1. Push your latest code and tag to GitHub:
   ```sh
   git tag v1.1.0
   git push origin v1.1.0
   ```
2. Navigate to your repository on GitHub: `https://github.com/<your-username>/<your-repo>`.
3. In the right-hand sidebar, click **"Releases"** (or go to `https://github.com/<your-username>/<your-repo>/releases`).
4. Click **"Draft a new release"** (or **"Create a new release"**).
5. Choose your tag (`v1.1.0`).
6. Set the Release title (e.g., `MiniTensor v1.1.0: Arbitrary-Rank Tensor Multiplication & Single-Header Release`).
7. In the section labeled **"Attach binaries by dropping them here or selecting them"**:
   - Drag and drop `release/minitensor.hpp`
   - Drag and drop `release/minitensor-v1.1.0.zip`
8. Click **"Publish release"**.

#### Option B: Using GitHub CLI (`gh`)
If you have the GitHub CLI installed:
```sh
gh release create v1.1.0 release/minitensor.hpp release/minitensor-v1.1.0.zip \
  --title "MiniTensor v1.1.0" \
  --notes "Release v1.1.0 featuring arbitrary-rank tensor multiplication and single-header distribution."
```

---

## 3. How to Download and Integrate MiniTensor

Here is how any developer in the world can download and use MiniTensor in their own C++ project:

---

### Method 1: Single-Header Download (Simplest, 1-Click)

This is the easiest method. You download **one single file** and you are ready to code!

#### Download via Terminal:
```sh
# On Linux / macOS:
curl -LO https://github.com/<your-username>/<your-repo>/releases/download/v1.1.0/minitensor.hpp

# On Windows PowerShell:
Invoke-WebRequest -Uri "https://github.com/<your-username>/<your-repo>/releases/download/v1.1.0/minitensor.hpp" -OutFile "minitensor.hpp"
```

#### In Your C++ File (`main.cpp`):
```cpp
#include "minitensor.hpp"
#include <iostream>

int main() {
    minitensor::Tensor<float> a({2, 3}, 1.5f);
    minitensor::Tensor<float> b({3, 2}, 2.0f);
    auto c = a.matmul(b);
    std::cout << "Result matrix: " << c(0, 0) << '\n';
    return 0;
}
```

#### Compile with Any C++20 Compiler:
```sh
# GCC / Clang:
g++ -std=c++20 -O3 main.cpp -o app

# MSVC (Windows):
cl /std:c++20 /O2 /EHsc main.cpp
```

---

### Method 2: CMake `FetchContent` (Automated Dependency Management)

With modern CMake, users don't even have to manually download anything! CMake will automatically clone, configure, and link MiniTensor during the build.

#### In Your Project's `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.16)
project(MyAwesomeApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    MiniTensor
    GIT_REPOSITORY https://github.com/<your-username>/<your-repo>.git
    GIT_TAG        v1.1.0
)
FetchContent_MakeAvailable(MiniTensor)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE MiniTensor::minitensor)
```

In your C++ code:
```cpp
#include <minitensor/minitensor.hpp>
```

---

### Method 3: Download Release Zip Archive

1. Download `minitensor-v1.1.0.zip` from the GitHub Releases tab.
2. Unzip into your project directory (e.g. `external/minitensor/`).
3. Compile with `-I external/minitensor/include`:
   ```sh
   g++ -std=c++20 -I external/minitensor/include main.cpp -o app
   ```

---

### Method 4: System Installation via CMake `find_package`

If you prefer to install MiniTensor once to your system or a central prefix:

```sh
# Clone and install:
git clone https://github.com/<your-username>/<your-repo>.git
cd minitensor
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --install build --prefix /usr/local
```

Then in any downstream project's `CMakeLists.txt`:
```cmake
find_package(MiniTensor CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE MiniTensor::minitensor)
```

---

### Method 5: Git Submodule

```sh
git submodule add https://github.com/<your-username>/<your-repo>.git external/minitensor
```
In your `CMakeLists.txt`:
```cmake
add_subdirectory(external/minitensor)
target_link_libraries(my_app PRIVATE MiniTensor::minitensor)
```

---

## 4. Using the Library: Code Examples Across All Ranks

MiniTensor now supports **arbitrary-rank tensor multiplication**! Below are complete, copy-pasteable examples for every rank.

### 1. Rank 1 Vector Dot Product
Multiplying two 1D vectors $(K) \times (K)$ yields a **Rank 0 scalar**:

```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<int> v1({3}, {1, 2, 3});
    minitensor::Tensor<int> v2({3}, {4, 5, 6});

    auto dot = v1.matmul(v2);

    std::cout << "Dot product rank: " << dot.rank() << '\n'; // 0 (Scalar)
    std::cout << "Dot product value: " << dot() << '\n';     // 32
}
```

### 2. Rank 2 Matrix Multiplication
Standard linear algebra multiplication $(M \times K) \times (K \times N) \to (M \times N)$:

```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<float> a({2, 3}, {1, 2, 3, 4, 5, 6});
    minitensor::Tensor<float> b({3, 2}, {7, 8, 9, 10, 11, 12});

    auto c = a.matmul(b); // Shape: {2, 2}
    std::cout << "Result: " << c(0, 0) << " " << c(0, 1) << '\n'; // 58 64
}
```

### 3. Rank 3 Batched Matrix Multiplication
In deep learning and computer vision, datasets are grouped in batches $(B \times M \times K) \times (B \times K \times N) \to (B \times M \times N)$:

```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    // 2 batches of (2x2) matrices
    minitensor::Tensor<int> a({2, 2, 2}, {1, 2, 3, 4,  5, 6, 7, 8});
    minitensor::Tensor<int> b({2, 2, 2}, {2, 0, 1, 2,  1, 3, 2, 4});

    auto res = a.matmul(b); // Result shape: {2, 2, 2}

    std::cout << "Batch 0, top-left: " << res(0, 0, 0) << '\n'; // 4
    std::cout << "Batch 1, top-left: " << res(1, 0, 0) << '\n'; // 17
}
```

### 4. Rank 4 Multi-Batch Matrix Multiplication
MiniTensor supports any number of leading batch dimensions $(B_1 \times B_2 \times M \times K) \times (B_1 \times B_2 \times K \times N)$:

```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    // Rank 4: [2, 3, 4, 5] x [2, 3, 5, 6] -> [2, 3, 4, 6]
    minitensor::Tensor<float> a({2, 3, 4, 5}, 1.0f);
    minitensor::Tensor<float> b({2, 3, 5, 6}, 2.0f);

    auto res = a.matmul(b);

    std::cout << "Rank: " << res.rank() << '\n'; // 4
    std::cout << "Shape: {" << res.get_shape()[0] << ", "
                            << res.get_shape()[1] << ", "
                            << res.get_shape()[2] << ", "
                            << res.get_shape()[3] << "}\n"; // {2, 3, 4, 6}
}
```

### 5. Batch Broadcasting (3D Batch x 2D Matrix)
Multiply a batched 3D tensor $(B \times M \times K)$ by a single shared 2D weight matrix $(K \times N)$ without duplicating the weight matrix:

```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<float> batched_inputs({10, 4, 8}, 1.0f); // 10 batches of 4x8
    minitensor::Tensor<float> weights({8, 3}, 0.5f);            // Shared 8x3 weights

    // Automatically broadcasts weights across all 10 batches!
    auto outputs = batched_inputs.matmul(weights); // Shape: {10, 4, 3}

    std::cout << "Output shape: {" << outputs.get_shape()[0] << ", "
                                   << outputs.get_shape()[1] << ", "
                                   << outputs.get_shape()[2] << "}\n"; // {10, 4, 3}
}
```

### 6. Arbitrary-Axis Tensor Contraction (`tensordot`)
Contract tensors along any custom axes, exactly like `numpy.tensordot`:

```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<int> a({2, 3, 4}, 1);
    minitensor::Tensor<int> b({4, 3, 5}, 2);

    // Contract axis 2 of a (length 4) with axis 0 of b (length 4)
    // and axis 1 of a (length 3) with axis 1 of b (length 3):
    auto contracted = a.tensordot(b, {2, 1}, {0, 1}); // Result shape: {2, 5}

    std::cout << "Contracted shape: {" << contracted.get_shape()[0] << ", "
                                       << contracted.get_shape()[1] << "}\n"; // {2, 5}
}
```

### 7. High-Performance Blocked & OpenMP Parallel Multiplication
```cpp
#include <minitensor/minitensor.hpp>

int main() {
    minitensor::Tensor<float> a({16, 256, 256}, 1.0f);
    minitensor::Tensor<float> b({16, 256, 256}, 2.0f);

    // Cache-tiled sub-block multiplication (L1 cache resident):
    auto c_blocked = a.matmul_blocked(b, 64);

    // Multithreaded OpenMP execution across CPU cores:
    auto c_parallel = a.matmul_parallel(b, 64);
}
```

---

## 5. Compilation Commands

### GCC (Linux / MinGW)
```sh
g++ -std=c++20 -O3 -Wall -Wextra main.cpp -o app
# With OpenMP:
g++ -std=c++20 -O3 -fopenmp main.cpp -o app
```

### Clang (macOS / Linux)
```sh
clang++ -std=c++20 -O3 -Wall -Wextra main.cpp -o app
```

### Microsoft Visual C++ (MSVC on Windows)
```sh
cl /std:c++20 /O2 /EHsc /W4 main.cpp
# With OpenMP:
cl /std:c++20 /O2 /EHsc /openmp main.cpp
```

