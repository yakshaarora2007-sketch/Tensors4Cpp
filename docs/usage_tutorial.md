# MiniTensor Progressive Usage Tutorial

This tutorial takes you on a step-by-step journey from writing your first line of code with MiniTensor to building a complete multidimensional image processing pipeline.

Every example is a complete, self-contained C++ program. We explain every key line of code, describe what happens inside the library, and show the exact output you will see on your screen.

---

## Prerequisites and Setup

To compile these examples, ensure you have:
- A C++20 compliant compiler (GCC 10+, Clang 11+, or MSVC 2019/2022).
- MiniTensor headers in your include path (e.g. `-I<path_to_minitensor>/include`).

Compile any example from the repository root:
```sh
# On Linux / macOS (g++ or clang++):
g++ -std=c++20 -Iinclude my_program.cpp -o my_program

# On Windows (MSVC cl):
cl /std:c++20 /Iinclude my_program.cpp
```

---

## Tutorial Steps Overview

1. [Step 1: Including the Library](#step-1-including-the-library)
2. [Step 2: Basic Tensor Construction](#step-2-basic-tensor-construction)
3. [Step 3: Creating and Using a Scalar (Rank 0)](#step-3-creating-and-using-a-scalar-rank-0)
4. [Step 4: Creating a 1D Vector (Rank 1)](#step-4-creating-a-1d-vector-rank-1)
5. [Step 5: Creating a 2D Matrix (Rank 2)](#step-5-creating-a-2d-matrix-rank-2)
6. [Step 6: Creating a 3D Tensor (Volume)](#step-6-creating-a-3d-tensor-volume)
7. [Step 7: Inspecting Shape, Rank, Size, and Strides](#step-7-inspecting-shape-rank-size-and-strides)
8. [Step 8: Filling and Bulk Updating Tensors](#step-8-filling-and-bulk-updating-tensors)
9. [Step 9: Accessing and Modifying Elements](#step-9-accessing-and-modifying-elements)
10. [Step 10: Tensor Addition and Subtraction](#step-10-tensor-addition-and-subtraction)
11. [Step 11: Element-Wise Multiplication (Hadamard Product)](#step-11-element-wise-multiplication-hadamard-product)
12. [Step 12: Matrix Multiplication (`matmul`)](#step-12-matrix-multiplication-matmul)
13. [Step 13: Reshaping Tensors](#step-13-reshaping-tensors)
14. [Step 14: Transposing Matrices](#step-14-transposing-matrices)
15. [Step 15: Putting It All Together: A Synthetic Image Processing Pipeline](#step-15-putting-it-all-together-a-synthetic-image-processing-pipeline)

---

## Step 1: Including the Library

The easiest way to use MiniTensor is through its single umbrella header:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    std::cout << "MiniTensor is ready!\n";
    return 0;
}
```
**Explanation:**
- `<minitensor/minitensor.hpp>` automatically pulls in `tensor.hpp`, `indexing.hpp`, `operations.hpp`, and `linalg.hpp`.
- All functionality lives inside the `minitensor::` namespace.

---

## Step 2: Basic Tensor Construction

Let us explore the four primary ways to construct a tensor:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>
#include <vector>

int main() {
    // 1. Default constructor: creates a scalar
    minitensor::Tensor<float> t1;

    // 2. Shape-only: 2x3 matrix, value-initialized (filled with 0.0f)
    minitensor::Tensor<float> t2({2, 3});

    // 3. Shape + constant fill value: 2x3 matrix filled with 7.5f
    minitensor::Tensor<float> t3({2, 3}, 7.5f);

    // 4. Shape + existing vector data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    minitensor::Tensor<float> t4({2, 3}, std::move(data));

    std::cout << "t1 size: " << t1.size() << '\n';
    std::cout << "t2 size: " << t2.size() << '\n';
    std::cout << "t3 size: " << t3.size() << '\n';
    std::cout << "t4 size: " << t4.size() << '\n';
}
```
**Expected Output:**
```text
t1 size: 1
t2 size: 6
t3 size: 6
t4 size: 6
```
**What happens internally:**
- `t2` computes $2 \times 3 = 6$, validates overflow, and allocates `std::vector<float>(6, 0.0f)`.
- `t3` allocates `std::vector<float>(6, 7.5f)`.
- `t4` verifies `data.size() == 6` and moves the buffer into `data_` with zero copying.

---

## Step 3: Creating and Using a Scalar (Rank 0)

A scalar tensor represents a single point in data space:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<double> scalar;

    // Access or assign the scalar value using operator()()
    scalar() = 42.5;

    std::cout << "Scalar rank: " << scalar.rank() << '\n';
    std::cout << "Scalar size: " << scalar.size() << '\n';
    std::cout << "Scalar value: " << scalar() << '\n';
}
```
**Expected Output:**
```text
Scalar rank: 0
Scalar size: 1
Scalar value: 42.5
```
**What happens internally:**
- The shape is `{}` (0 dimensions, rank 0).
- `operator()()` invokes `at({})`, which maps to offset 0 in the 1-element buffer.

---

## Step 4: Creating a 1D Vector (Rank 1)

A 1D tensor represents a list of numbers along a single axis:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<int> vec({4}, {10, 20, 30, 40});

    std::cout << "Vector rank: " << vec.rank() << '\n';
    std::cout << "Vector size: " << vec.size() << '\n';

    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::cout << "vec(" << i << ") = " << vec(i) << '\n';
    }
}
```
**Expected Output:**
```text
Vector rank: 1
Vector size: 4
vec(0) = 10
vec(1) = 20
vec(2) = 30
vec(3) = 40
```

---

## Step 5: Creating a 2D Matrix (Rank 2)

A 2D tensor has rows and columns:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<int> matrix({2, 3}, {1, 2, 3, 4, 5, 6});

    std::cout << "Matrix (2 rows, 3 columns):\n";
    for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            std::cout << matrix(r, c) << ' ';
        }
        std::cout << '\n';
    }
}
```
**Expected Output:**
```text
Matrix (2 rows, 3 columns):
1 2 3 
4 5 6 
```

---

## Step 6: Creating a 3D Tensor (Volume)

A 3D tensor can be thought of as a book of matrices (Depth $\times$ Rows $\times$ Columns):
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    // 2 pages, 2 rows per page, 3 columns per row
    minitensor::Tensor<int> volume({2, 2, 3});
    
    // Fill with values
    int counter = 1;
    for (std::size_t d = 0; d < 2; ++d) {
        for (std::size_t r = 0; r < 2; ++r) {
            for (std::size_t c = 0; c < 3; ++c) {
                volume(d, r, c) = counter++;
            }
        }
    }

    std::cout << "3D Volume Rank: " << volume.rank() << ", Size: " << volume.size() << '\n';
    std::cout << "Value at page 1, row 0, col 2: " << volume(1, 0, 2) << '\n';
}
```
**Expected Output:**
```text
3D Volume Rank: 3, Size: 12
Value at page 1, row 0, col 2: 9
```

---

## Step 7: Inspecting Shape, Rank, Size, and Strides

Let us inspect the metadata of any tensor:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

template <typename T>
void inspect(const minitensor::Tensor<T>& t, const std::string& name) {
    std::cout << "=== Tensor: " << name << " ===\n";
    std::cout << "Rank: " << t.rank() << '\n';
    std::cout << "Size: " << t.size() << " elements\n";
    
    std::cout << "Shape:   { ";
    for (auto d : t.get_shape()) std::cout << d << ' ';
    std::cout << "}\n";

    std::cout << "Strides: { ";
    for (auto s : t.get_strides()) std::cout << s << ' ';
    std::cout << "}\n\n";
}

int main() {
    inspect(minitensor::Tensor<float>(), "Scalar");
    inspect(minitensor::Tensor<float>({4}), "1D Vector");
    inspect(minitensor::Tensor<float>({2, 3}), "2D Matrix");
    inspect(minitensor::Tensor<float>({2, 3, 4}), "3D Volume");
}
```
**Expected Output:**
```text
=== Tensor: Scalar ===
Rank: 0
Size: 1 elements
Shape:   { }
Strides: { }

=== Tensor: 1D Vector ===
Rank: 1
Size: 4 elements
Shape:   { 4 }
Strides: { 1 }

=== Tensor: 2D Matrix ===
Rank: 2
Size: 6 elements
Shape:   { 2 3 }
Strides: { 3 1 }

=== Tensor: 3D Volume ===
Rank: 3
Size: 24 elements
Shape:   { 2 3 4 }
Strides: { 12 4 1 }
```

---

## Step 8: Filling and Bulk Updating Tensors

You can overwrite an entire tensor with a constant value using `fill()` or replace its data with `set_data()`:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<int> t({2, 2}, 0);

    // Overwrite all 4 elements with 99:
    t.fill(99);
    std::cout << "After fill(99): " << t(0, 0) << " " << t(1, 1) << '\n';

    // Replace data buffer with new vector:
    t.set_data({5, 10, 15, 20});
    std::cout << "After set_data: " << t(0, 0) << " " << t(0, 1) << " "
              << t(1, 0) << " " << t(1, 1) << '\n';
}
```
**Expected Output:**
```text
After fill(99): 99 99
After set_data: 5 10 15 20
```

---

## Step 9: Accessing and Modifying Elements

MiniTensor provides three ways to access elements:
1. `tensor(i, j, ...)`: variadic call operator.
2. `tensor.at({i, j, ...})`: vector / initializer-list method (returns mutable reference).
3. `tensor.set({i, j, ...}, value)`: explicit setter.

```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<int> matrix({2, 2}, 0);

    // Using operator()
    matrix(0, 0) = 10;

    // Using at()
    matrix.at({0, 1}) = 20;

    // Using set()
    matrix.set({1, 0}, 30);
    matrix.set({1, 1}, 40);

    // Out-of-bounds error handling:
    try {
        matrix(5, 5) = 100; // Invalid row 5, col 5!
    } catch (const std::out_of_range& e) {
        std::cout << "Caught expected error: " << e.what() << '\n';
    }
}
```
**Expected Output:**
```text
Caught expected error: Tensor index is out of bounds
```

---

## Step 10: Tensor Addition and Subtraction

Element-wise arithmetic requires matching shapes:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<int> a({2, 2}, {1, 2, 3, 4});
    minitensor::Tensor<int> b({2, 2}, {10, 20, 30, 40});

    minitensor::Tensor<int> sum = a + b;
    minitensor::Tensor<int> diff = b - a;

    std::cout << "Sum:  " << sum(0, 0) << ", " << sum(0, 1) << ", "
              << sum(1, 0) << ", " << sum(1, 1) << '\n';
    std::cout << "Diff: " << diff(0, 0) << ", " << diff(0, 1) << ", "
              << diff(1, 0) << ", " << diff(1, 1) << '\n';
}
```
**Expected Output:**
```text
Sum:  11, 22, 33, 44
Diff: 9, 18, 27, 36
```

---

## Step 11: Element-Wise Multiplication (Hadamard Product)

`operator*` performs element-by-element multiplication:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<float> a({2, 2}, {2.0f, 3.0f, 4.0f, 5.0f});
    minitensor::Tensor<float> b({2, 2}, {10.0f, 10.0f, 0.5f, 2.0f});

    auto prod = a * b; // Hadamard product
    std::cout << "Product: " << prod(0, 0) << ", " << prod(0, 1) << ", "
              << prod(1, 0) << ", " << prod(1, 1) << '\n';
}
```
**Expected Output:**
```text
Product: 20, 30, 2, 10
```

---

## Step 12: Matrix Multiplication (`matmul`)

Matrix multiplication multiplies rows by columns:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    // 2x3 matrix
    minitensor::Tensor<int> a({2, 3}, {1, 2, 3, 4, 5, 6});
    // 3x2 matrix
    minitensor::Tensor<int> b({3, 2}, {7, 8, 9, 10, 11, 12});

    // Result is 2x2:
    auto c = a.matmul(b);

    std::cout << "Matrix Multiplication Result (2x2):\n";
    std::cout << c(0, 0) << " " << c(0, 1) << '\n';
    std::cout << c(1, 0) << " " << c(1, 1) << '\n';
}
```
**Expected Output:**
```text
Matrix Multiplication Result (2x2):
58 64
139 154
```

---

## Step 13: Reshaping Tensors

You can change the shape of a tensor as long as the total number of elements stays identical:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    // 1D vector of 6 elements
    minitensor::Tensor<int> vec({6}, {1, 2, 3, 4, 5, 6});

    // Reshape to 2x3 matrix:
    auto matrix = vec.reshape({2, 3});
    std::cout << "Reshaped to 2x3, element (1, 2): " << matrix(1, 2) << '\n';

    // Reshape to 3x2 matrix:
    auto matrix32 = vec.reshape({3, 2});
    std::cout << "Reshaped to 3x2, element (2, 1): " << matrix32(2, 1) << '\n';
}
```
**Expected Output:**
```text
Reshaped to 2x3, element (1, 2): 6
Reshaped to 3x2, element (2, 1): 6
```

---

## Step 14: Transposing Matrices

Transpose swaps the rows and columns of a 2D matrix:
```cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

int main() {
    minitensor::Tensor<int> original({2, 3}, {1, 2, 3, 4, 5, 6});
    auto transposed = original.transpose();

    std::cout << "Original shape:   {" << original.get_shape()[0] << ", " << original.get_shape()[1] << "}\n";
    std::cout << "Transposed shape: {" << transposed.get_shape()[0] << ", " << transposed.get_shape()[1] << "}\n";

    std::cout << "Transposed values:\n";
    for (std::size_t r = 0; r < transposed.get_shape()[0]; ++r) {
        for (std::size_t c = 0; c < transposed.get_shape()[1]; ++c) {
            std::cout << transposed(r, c) << ' ';
        }
        std::cout << '\n';
    }
}
```
**Expected Output:**
```text
Original shape:   {2, 3}
Transposed shape: {3, 2}
Transposed values:
1 4 
2 5 
3 6 
```

---

## Step 15: Putting It All Together: A Synthetic Image Processing Pipeline

Here is a complete, real-world application demonstrating how MiniTensor can be used to represent and process a color image (shape `{Height, Width, Channels}`):

```cpp
#include <minitensor/minitensor.hpp>
#include <cstddef>
#include <iomanip>
#include <iostream>

int main() {
    // 1. Create a 2x2 color image with 3 channels (RGB) -> Shape: {2, 2, 3}
    constexpr std::size_t height = 2;
    constexpr std::size_t width = 2;
    minitensor::Tensor<float> image({height, width, 3});

    // 2. Populate synthetic pixels (R, G, B)
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            image.set({y, x, 0}, static_cast<float>(y + 1) * 0.1f); // Red
            image.set({y, x, 1}, static_cast<float>(x + 1) * 0.2f); // Green
            image.set({y, x, 2}, 0.3f);                             // Blue
        }
    }

    // 3. Brighten the image: add 0.1f to every channel of every pixel
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            for (std::size_t c = 0; c < 3; ++c) {
                image.at({y, x, c}) += 0.1f;
            }
        }
    }

    // 4. Element-wise scaling: Dim the image by 50% using tensor multiplication
    minitensor::Tensor<float> factor({height, width, 3}, 0.5f);
    minitensor::Tensor<float> dimmed = image * factor;

    // 5. Channel swapping: Convert RGB to BGR
    minitensor::Tensor<float> bgr({height, width, 3});
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            bgr.set({y, x, 0}, dimmed.at({y, x, 2})); // Blue to Channel 0
            bgr.set({y, x, 1}, dimmed.at({y, x, 1})); // Green stays Channel 1
            bgr.set({y, x, 2}, dimmed.at({y, x, 0})); // Red to Channel 2
        }
    }

    // 6. Display results
    std::cout << "Processed Image (Height=2, Width=2, Channels=3):\n";
    std::cout << std::fixed << std::setprecision(2);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            std::cout << "Pixel (" << y << "," << x << ") BGR: ("
                      << bgr(y, x, 0) << ", "
                      << bgr(y, x, 1) << ", "
                      << bgr(y, x, 2) << ")  ";
        }
        std::cout << '\n';
    }
    return 0;
}
```
**Expected Output:**
```text
Processed Image (Height=2, Width=2, Channels=3):
Pixel (0,0) BGR: (0.20, 0.15, 0.10)  Pixel (0,1) BGR: (0.20, 0.25, 0.10)  
Pixel (1,0) BGR: (0.20, 0.15, 0.15)  Pixel (1,1) BGR: (0.20, 0.25, 0.15)  
```

Congratulations! You now have complete practical mastery over the MiniTensor library!

