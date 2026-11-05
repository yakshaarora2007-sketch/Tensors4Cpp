# How to Download, Import, and Use MiniTensor Header Files

This comprehensive guide explains how to **download**, **import**, and **use** the MiniTensor C++ header files in your own projects.

---

## Table of Contents

1. [Where to Download the Header Files](#1-where-to-download-the-header-files)
2. [Download Methods](#2-download-methods)
   - [Method 1: Direct Single-Header Download (Recommended)](#method-1-direct-single-header-download-recommended)
   - [Method 2: Full Release Zip Archive](#method-2-full-release-zip-archive)
   - [Method 3: Zero-Download via CMake `FetchContent`](#method-3-zero-download-via-cmake-fetchcontent)
   - [Method 4: Git Clone / Submodule](#method-4-git-clone--submodule)
3. [How to Import and Include the Headers in Your C++ Project](#3-how-to-import-and-include-the-headers-in-your-c-project)
   - [Approach A: Using the Single Header (`minitensor.hpp`)](#approach-a-using-the-single-header-minitensorhpp)
   - [Approach B: Using the Modular Headers (`include/minitensor/`)](#approach-b-using-the-modular-headers-includeminitensor)
4. [Setting Up Include Paths in Popular IDEs](#4-setting-up-include-paths-in-popular-ides)
   - [Visual Studio (Windows)](#visual-studio-windows)
   - [Visual Studio Code (VS Code)](#visual-studio-code-vs-code)
   - [CLion / CMake](#clion--cmake)
5. [Complete Code Examples Across All Ranks](#5-complete-code-examples-across-all-ranks)
   - [Example 1: Basic Tensor Creation & Element Access](#example-1-basic-tensor-creation--element-access)
   - [Example 2: Rank 1 Vector Dot Product](#example-2-rank-1-vector-dot-product)
   - [Example 3: Rank 2 Matrix Multiplication](#example-3-rank-2-matrix-multiplication)
   - [Example 4: Rank 3 Batched Matrix Multiplication](#example-4-rank-3-batched-matrix-multiplication)
   - [Example 5: Rank 4 Multi-Batch Matrix Multiplication](#example-5-rank-4-multi-batch-matrix-multiplication)
   - [Example 6: Batch Dimension Broadcasting (3D Batch x 2D Matrix)](#example-6-batch-dimension-broadcasting-3d-batch-x-2d-matrix)
   - [Example 7: Arbitrary-Axis Tensor Contraction (`tensordot`)](#example-7-arbitrary-axis-tensor-contraction-tensordot)
6. [Compilation Commands for GCC, Clang, and MSVC](#6-compilation-commands-for-gcc-clang-and-msvc)
7. [Troubleshooting & FAQ](#7-troubleshooting--faq)

---

## 1. Where to Download the Header Files

All official header files and release packages are published under the **Releases** tab of the GitHub repository:

🔗 **GitHub Releases Page:**  
👉 [https://github.com/yakshaarora2007-sketch/Tensors4Cpp/releases](https://github.com/yakshaarora2007-sketch/Tensors4Cpp/releases)

Under the latest release (e.g., `v1.1.0`), you will find the following assets:

| Release Asset | Description | When to Choose |
| :--- | :--- | :--- |
| **`minitensor.hpp`** | The complete library amalgamated into **one single header file** (~30 KB). Zero dependencies, zero build steps. | **Recommended for 90% of users.** Just download 1 file, drop it into your project folder, and write code! |
| **`minitensor-v1.1.0.zip`** | Full library archive containing modular headers (`include/minitensor/*.hpp`), CMake build files, documentation, and LICENSE. | Best if you prefer organized subfolders, modular headers, or CMake integration. |
| **Source Code (`.zip` / `.tar.gz`)** | Complete repository snapshot including tests, examples, benchmarks, and research paper. | Best if you want to run the tests, benchmarks, or contribute to the library. |

---

## 2. Download Methods

### Method 1: Direct Single-Header Download (Recommended)

You can download `minitensor.hpp` directly using your web browser or terminal:

#### Via Terminal / Command Line:

**On Linux / macOS:**
```sh
curl -LO https://github.com/yakshaarora2007-sketch/Tensors4Cpp/releases/download/v1.1.0/minitensor.hpp
```
or with `wget`:
```sh
wget https://github.com/yakshaarora2007-sketch/Tensors4Cpp/releases/download/v1.1.0/minitensor.hpp
```

**On Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://github.com/yakshaarora2007-sketch/Tensors4Cpp/releases/download/v1.1.0/minitensor.hpp" -OutFile "minitensor.hpp"
```

---

### Method 2: Full Release Zip Archive

1. Go to [https://github.com/yakshaarora2007-sketch/Tensors4Cpp/releases](https://github.com/yakshaarora2007-sketch/Tensors4Cpp/releases).
2. Click on **`minitensor-v1.1.0.zip`** to download.
3. Extract the contents:
   ```sh
   unzip minitensor-v1.1.0.zip
   ```
4. Copy the `include/minitensor` folder into your project's `include/` directory.

---

### Method 3: Zero-Download via CMake `FetchContent`

If your project uses CMake, you do **not** even have to download files manually! CMake can download and configure MiniTensor automatically during the build:

In your `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.16)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    MiniTensor
    GIT_REPOSITORY https://github.com/yakshaarora2007-sketch/Tensors4Cpp.git
    GIT_TAG        v1.1.0
)
FetchContent_MakeAvailable(MiniTensor)

add_executable(my_program main.cpp)
target_link_libraries(my_program PRIVATE MiniTensor::minitensor)
```

---

### Method 4: Git Clone / Submodule

```sh
# Clone directly:
git clone https://github.com/yakshaarora2007-sketch/Tensors4Cpp.git

# Or add as a git submodule in your existing repository:
git submodule add https://github.com/yakshaarora2007-sketch/Tensors4Cpp.git external/minitensor
```

---

## 3. How to Import and Include the Headers in Your C++ Project

### Approach A: Using the Single Header (`minitensor.hpp`)

1. Place `minitensor.hpp` in the same directory as your source code (or in an `include/` directory).
2. In your C++ file:
   ```cpp
   #include "minitensor.hpp"

   int main() {
       minitensor::Tensor<float> tensor({2, 3}, 1.0f);
       return 0;
   }
   ```
3. Compile with C++20 enabled:
   ```sh
   g++ -std=c++20 main.cpp -o main
   ```

### Approach B: Using the Modular Headers (`include/minitensor/`)

If you downloaded the full archive or cloned the repo:
```text
my_project/
├── include/
│   └── minitensor/
│       ├── minitensor.hpp
│       ├── tensor.hpp
│       ├── indexing.hpp
│       ├── operations.hpp
│       └── linalg.hpp
└── main.cpp
```
In your C++ file:
```cpp
#include <minitensor/minitensor.hpp>

int main() {
    minitensor::Tensor<double> matrix({3, 3}, 0.0);
    return 0;
}
```
Compile specifying the include folder:
```sh
g++ -std=c++20 -I include main.cpp -o main
```

---

## 4. Setting Up Include Paths in Popular IDEs

### Visual Studio (Windows)
1. Right-click your Project in **Solution Explorer** $\to$ **Properties**.
2. Set **Configuration** to **All Configurations**.
3. Navigate to **Configuration Properties** $\to$ **General** $\to$ **C++ Language Standard** $\to$ Select **ISO C++20 Standard (`/std:c++20`)**.
4. Navigate to **C/C++** $\to$ **General** $\to$ **Additional Include Directories**.
5. Add the folder containing `minitensor.hpp` (or the folder containing `minitensor/`).
6. Click **Apply** $\to$ **OK**.

### Visual Studio Code (VS Code)
1. Open `.vscode/c_cpp_properties.json` (or press `Ctrl+Shift+P` $\to$ **C/C++: Edit Configurations (JSON)**).
2. Set `"cppStandard": "c++20"`.
3. In `"includePath"`, add the path to your header:
   ```json
   {
       "configurations": [
           {
               "name": "Win32",
               "includePath": [
                   "${workspaceFolder}/**",
                   "${workspaceFolder}/include"
               ],
               "cppStandard": "c++20"
           }
       ],
       "version": 4
   }
   ```

### CLion / CMake
Simply add `target_include_directories` in your `CMakeLists.txt`:
```cmake
target_include_directories(my_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

---

## 5. Complete Code Examples Across All Ranks

MiniTensor supports tensor operations across **any rank** (Rank 1 vectors, Rank 2 matrices, Rank 3 volumes, Rank 4 batches, etc.).

### Example 1: Basic Tensor Creation & Element Access

```cpp
#include "minitensor.hpp"
#include <iostream>

int main() {
    // Create a 2x3 matrix filled with values
    minitensor::Tensor<float> matrix({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Access elements via (row, col)
    matrix(0, 1) = 99.0f;
    std::cout << "Value at (0, 1): " << matrix(0, 1) << '\n';

    // Query metadata
    std::cout << "Rank: " << matrix.rank() << '\n'; // 2
    std::cout << "Size: " << matrix.size() << '\n'; // 6
    return 0;
}
```

### Example 2: Rank 1 Vector Dot Product
Multiplying two 1D vectors $(K) \times (K)$ computes the inner dot product and returns a **Rank 0 scalar**:

```cpp
#include "minitensor.hpp"
#include <iostream>

int main() {
    minitensor::Tensor<int> v1({3}, {1, 2, 3});
    minitensor::Tensor<int> v2({3}, {4, 5, 6});

    auto dot = v1.matmul(v2);

    std::cout << "Dot product rank: " << dot.rank() << '\n'; // 0 (Scalar)
    std::cout << "Dot product value: " << dot() << '\n';     // 32: (1*4 + 2*5 + 3*6)
    return 0;
}
```

### Example 3: Rank 2 Matrix Multiplication
Standard $(M \times K) \times (K \times N) \to (M \times N)$:

```cpp
#include "minitensor.hpp"
#include <iostream>

int main() {
    minitensor::Tensor<double> a({2, 3}, {1, 2, 3, 4, 5, 6});
    minitensor::Tensor<double> b({3, 2}, {7, 8, 9, 10, 11, 12});

    auto c = a.matmul(b); // Result shape: {2, 2}

    std::cout << "C(0,0): " << c(0, 0) << ", C(0,1): " << c(0, 1) << '\n'; // 58, 64
    std::cout << "C(1,0): " << c(1, 0) << ", C(1,1): " << c(1, 1) << '\n'; // 139, 154
    return 0;
}
```

### Example 4: Rank 3 Batched Matrix Multiplication
In computer vision and deep learning, matrices are processed in batches:
$$(B \times M \times K) \times (B \times K \times N) \to (B \times M \times N)$$

```cpp
#include "minitensor.hpp"
#include <iostream>

int main() {
    // 2 batches of 2x2 matrices
    minitensor::Tensor<int> a({2, 2, 2}, {1, 2, 3, 4,  5, 6, 7, 8});
    minitensor::Tensor<int> b({2, 2, 2}, {2, 0, 1, 2,  1, 3, 2, 4});

    auto res = a.matmul(b); // Result shape: {2, 2, 2}

    std::cout << "Batch 0 top-left: " << res(0, 0, 0) << '\n'; // 4
    std::cout << "Batch 1 top-left: " << res(1, 0, 0) << '\n'; // 17
    return 0;
}
```

### Example 5: Rank 4 Multi-Batch Matrix Multiplication
MiniTensor supports any number of leading batch dimensions:
$$(B_1 \times B_2 \times M \times K) \times (B_1 \times B_2 \times K \times N) \to (B_1 \times B_2 \times M \times N)$$

```cpp
#include "minitensor.hpp"
#include <iostream>

int main() {
    // Shape: [2, 3, 4, 5] x [2, 3, 5, 6] -> [2, 3, 4, 6]
    minitensor::Tensor<float> a({2, 3, 4, 5}, 1.0f);
    minitensor::Tensor<float> b({2, 3, 5, 6}, 2.0f);

    auto res = a.matmul(b);

    std::cout << "Rank: " << res.rank() << '\n'; // 4
    std::cout << "Shape: {" << res.get_shape()[0] << ", "
                            << res.get_shape()[1] << ", "
                            << res.get_shape()[2] << ", "
                            << res.get_shape()[3] << "}\n"; // {2, 3, 4, 6}
    return 0;
}
```

### Example 6: Batch Dimension Broadcasting (3D Batch x 2D Matrix)
Multiply a batched 3D tensor by a single shared 2D weight matrix without copying the weight matrix:

```cpp
#include "minitensor.hpp"
#include <iostream>

int main() {
    minitensor::Tensor<float> batched_inputs({10, 4, 8}, 1.0f); // 10 batches of 4x8
    minitensor::Tensor<float> weights({8, 3}, 0.5f);            // 2D matrix 8x3

    // Automatically broadcasts weights across all 10 batches!
    auto outputs = batched_inputs.matmul(weights); // Shape: {10, 4, 3}

    std::cout << "Outputs shape: {" << outputs.get_shape()[0] << ", "
                                    << outputs.get_shape()[1] << ", "
                                    << outputs.get_shape()[2] << "}\n"; // {10, 4, 3}
    return 0;
}
```

### Example 7: Arbitrary-Axis Tensor Contraction (`tensordot`)
Contract tensors along any custom pairs of axes (identical to `numpy.tensordot`):

```cpp
#include "minitensor.hpp"
#include <iostream>

int main() {
    minitensor::Tensor<int> a({2, 3, 4}, 1);
    minitensor::Tensor<int> b({4, 3, 5}, 2);

    // Contract axis 2 of a (length 4) with axis 0 of b (length 4)
    // and axis 1 of a (length 3) with axis 1 of b (length 3):
    auto result = a.tensordot(b, {2, 1}, {0, 1}); // Result shape: {2, 5}

    std::cout << "Result shape: {" << result.get_shape()[0] << ", "
                                   << result.get_shape()[1] << "}\n"; // {2, 5}
    return 0;
}
```

---

## 6. Compilation Commands for GCC, Clang, and MSVC

MiniTensor requires a standard **C++20** compiler:

### GCC (Linux / MinGW)
```sh
g++ -std=c++20 -O3 -Wall -Wextra main.cpp -o app

# With OpenMP multithreading enabled:
g++ -std=c++20 -O3 -fopenmp main.cpp -o app
```

### Clang (macOS / Linux)
```sh
clang++ -std=c++20 -O3 -Wall -Wextra main.cpp -o app
```

### Microsoft Visual C++ (MSVC on Windows)
```sh
cl /std:c++20 /O2 /EHsc /W4 main.cpp

# With OpenMP multithreading enabled:
cl /std:c++20 /O2 /EHsc /openmp main.cpp
```

---

## 7. Troubleshooting & FAQ

### Q: Why do I get a compiler error: `'requires' does not name a type`?
**A:** MiniTensor uses C++20 concepts and variadic requirements. Make sure your compiler is configured with `-std=c++20` (GCC/Clang) or `/std:c++20` (MSVC).

### Q: Why do I get `fatal error: minitensor.hpp: No such file or directory`?
**A:** If `minitensor.hpp` is in a subfolder (such as `include/`), compile with `-I include` or pass the relative path: `#include "include/minitensor.hpp"`.

### Q: Does MiniTensor require linking any `.lib`, `.so`, or `.dll` files?
**A:** No! MiniTensor is 100% header-only and dependency-free. There are no binary libraries to build or link.

### Q: How do I enable OpenMP acceleration?
**A:** Pass `-fopenmp` (GCC/Clang) or `/openmp` (MSVC) when compiling, and call `tensor.matmul_parallel(other)`. If OpenMP is not enabled, `matmul_parallel()` automatically runs serial multiplication.
