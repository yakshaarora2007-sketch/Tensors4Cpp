# Inside MiniTensor: Architecture and Implementation Details

This document explains the internal mechanics of the `minitensor::Tensor<T>` class. It connects the conceptual mathematics of tensors directly to the exact C++ source code in `include/minitensor/tensor.hpp`.

If you want to know how shapes are stored, how strides are calculated, how arithmetic overflow is prevented, and how coordinates are mapped to physical memory addresses, this document is your complete guide.

---

## Table of Contents

1. [The Internal Representation of `Tensor<T>`](#1-the-internal-representation-of-tensort)
   - [Why Templated on `T`?](#why-templated-on-t)
   - [The Three Private Member Variables](#the-three-private-member-variables)
2. [Metadata Architecture: Shape, Strides, and Data](#2-metadata-architecture-shape-strides-and-data)
   - [Memory Structure Diagram](#memory-structure-diagram)
3. [Calculating Tensor Size: Safety and Overflow Protection](#3-calculating-tensor-size-safety-and-overflow-protection)
   - [Zero-Dimension Detection](#zero-dimension-detection)
   - [Multiplication Overflow Checking with `size_t`](#multiplication-overflow-checking-with-sizet)
4. [Stride Calculation: From Last Axis to First](#4-stride-calculation-from-last-axis-to-first)
   - [Row-Major Backward Accumulation](#row-major-backward-accumulation)
   - [Stride Behavior for Empty / Zero-Sized Tensors](#stride-behavior-for-empty--zero-sized-tensors)
5. [Coordinate Indexing: The `calculate_offset` Engine](#5-coordinate-indexing-the-calculate_offset-engine)
   - [Rank Validation](#rank-validation)
   - [Bounds Checking](#bounds-checking)
   - [Offset Calculation Formula](#offset-calculation-formula)
6. [Scalar Tensor Semantics (Rank 0)](#6-scalar-tensor-semantics-rank-0)
7. [Element-Wise and Matrix Multiplication Helpers](#7-element-wise-and-matrix-multiplication-helpers)
8. [Summary of Invariants](#8-summary-of-invariants)

---

## 1. The Internal Representation of `Tensor<T>`

In `include/minitensor/tensor.hpp`, the entire tensor abstraction is encapsulated in a single clean class template:

```cpp
namespace minitensor {

template <typename T>
class Tensor {
    // ...
private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<T> data_;
};

}
```

### Why Templated on `T`?
The template parameter `T` represents the type of each individual element stored in the tensor:
- `Tensor<float>`: Single-precision floating point (commonly used in deep learning and image processing for speed and reduced memory usage).
- `Tensor<double>`: Double-precision floating point (standard in scientific computing, physics simulations, and numerical analysis where high precision is required).
- `Tensor<int>`: Signed 32-bit integers (used for discrete labels, counts, coordinate systems, and index maps).
- `Tensor<std::uint8_t>`: 8-bit unsigned bytes (used for raw image pixels or binary masks).

By templating on `T`, MiniTensor delivers optimal performance without runtime type-erasure overhead (such as `void*` or virtual method tables).

### The Three Private Member Variables

Every `Tensor<T>` instance owns three things:
1. `std::vector<std::size_t> shape_`: The dimensional extents of each axis.
   - For example, `{2, 3}` indicates 2 rows and 3 columns.
   - The rank of the tensor is simply `shape_.size()`.
2. `std::vector<std::size_t> strides_`: The pre-computed step sizes needed to navigate the flat buffer.
   - For shape `{2, 3}`, `strides_` is `{3, 1}`.
   - The size of `strides_` always matches `shape_.size()`.
3. `std::vector<T> data_`: The contiguous, flat memory buffer containing all elements.
   - Its size is strictly equal to the product of all numbers in `shape_` (or 1 for scalars, 0 for empty tensors).

---

## 2. Metadata Architecture: Shape, Strides, and Data

A key design principle of MiniTensor is the **separation of metadata from payload data**.

The metadata (`shape_` and `strides_`) is very lightweight: two small `std::vector<std::size_t>` objects stored on the stack. The payload data (`data_`) is allocated in one single contiguous block on the heap.

### Memory Structure Diagram

Consider a $2 \times 3$ tensor initialized with values `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`:

```text
Stack Object: minitensor::Tensor<float>
+-------------------------------------------------------+
|  shape_   --> [ 2, 3 ]                                |
|  strides_ --> [ 3, 1 ]                                |
|  data_    --> pointer to heap block ----------------+ |
+-----------------------------------------------------|-+
                                                      |
Heap Memory:                                          v
+-----------+-----------+-----------+-----------+-----------+-----------+
|   1.0     |   2.0     |   3.0     |   4.0     |   5.0     |   6.0     |
+-----------+-----------+-----------+-----------+-----------+-----------+
  Offset 0    Offset 1    Offset 2    Offset 3    Offset 4    Offset 5
  Row 0,Col 0 Row 0,Col 1 Row 0,Col 2 Row 1,Col 0 Row 1,Col 1 Row 1,Col 2
```

When you query `tensor.get_shape()`, `tensor.get_strides()`, or `tensor.get_data()`, you receive a `const` reference to these underlying vectors, giving $O(1)$ introspection with zero memory allocation.

---

## 3. Calculating Tensor Size: Safety and Overflow Protection

Before any memory is allocated, MiniTensor computes the total number of elements required by the shape using the private static helper `calculate_size()`:

```cpp
static std::size_t calculate_size(const std::vector<std::size_t>& shape) {
    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            return 0;
        }
    }

    std::size_t total_size = 1;
    for (const std::size_t dimension : shape) {
        if (total_size > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::length_error("Tensor shape product overflows size_t");
        }
        total_size *= dimension;
    }
    return total_size;
}
```

### Zero-Dimension Detection
If any dimension in the shape is `0` (e.g., `{100, 0, 50}`), the tensor is empty. 
`calculate_size()` detects this immediately and returns `0`. No memory is allocated, and unnecessary multiplications are avoided.

### Multiplication Overflow Checking with `size_t`
On 64-bit systems, `std::size_t` can hold up to $18,446,744,073,709,551,615$ ($\approx 1.84 \times 10^{19}$). However, if an application requests a shape such as `[1000000000, 1000000000, 1000000000]`, the product would wrap around to a small positive number due to integer overflow, leading to severe buffer-overflow vulnerabilities.

MiniTensor prevents this using standard checked division:
```cpp
if (total_size > std::numeric_limits<std::size_t>::max() / dimension) {
    throw std::length_error("Tensor shape product overflows size_t");
}
```
If an overflow is imminent, MiniTensor refuses to allocate and throws `std::length_error`.

---

## 4. Stride Calculation: From Last Axis to First

Strides translate multidimensional coordinates into flat 1D offsets.
MiniTensor pre-computes strides during construction in `calculate_strides()`:

```cpp
static std::vector<std::size_t> calculate_strides(
    const std::vector<std::size_t>& shape) {
    static_cast<void>(calculate_size(shape));
    std::vector<std::size_t> strides(shape.size(), 0);
    if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        // No coordinate is valid, so zero strides avoid meaningless
        // products (and overflow) for empty tensors.
        return strides;
    }

    std::size_t stride = 1;
    for (std::size_t axis = shape.size(); axis > 0; --axis) {
        strides[axis - 1] = stride;
        if (stride >
            std::numeric_limits<std::size_t>::max() / shape[axis - 1]) {
            throw std::length_error("Tensor stride overflows size_t");
        }
        stride *= shape[axis - 1];
    }
    return strides;
}
```

### Row-Major Backward Accumulation
Notice how the loop traverses backwards: `for (std::size_t axis = shape.size(); axis > 0; --axis)`.
Let us trace this step-by-step for a 3D tensor with shape `{2, 3, 4}`:

1. **Initialize:** `stride = 1`.
2. **Axis 2 (Last dimension, length 4):**
   - `strides[2] = stride = 1`.
   - `stride = stride * shape[2] = 1 * 4 = 4`.
3. **Axis 1 (Middle dimension, length 3):**
   - `strides[1] = stride = 4`.
   - `stride = stride * shape[1] = 4 * 3 = 12`.
4. **Axis 0 (First dimension, length 2):**
   - `strides[0] = stride = 12`.
   - `stride = stride * shape[0] = 12 * 2 = 24`.
5. **Final Strides Vector:** `{12, 4, 1}`.

Notice what these numbers mean:
- Jumping 1 unit along Axis 2 skips **1** element.
- Jumping 1 unit along Axis 1 skips **4** elements (1 full row).
- Jumping 1 unit along Axis 0 skips **12** elements ($3 \times 4$, 1 full page).

### Stride Behavior for Empty / Zero-Sized Tensors
If the shape contains a zero (e.g., `{2, 0, 3}`), no element exists. What should the strides be?
Rather than computing meaningless or overflowing strides, MiniTensor sets all strides to `0`:
`strides = {0, 0, 0}`.
Since no coordinate is valid in an empty tensor, any indexing attempt will fail the bounds check before strides are ever used.

---

## 5. Coordinate Indexing: The `calculate_offset` Engine

Whenever an element is accessed via `tensor.at({i, j})`, `tensor.set({i, j}, val)`, or `tensor(i, j)`, the private method `calculate_offset()` is called:

```cpp
[[nodiscard]] std::size_t calculate_offset(
    const std::vector<std::size_t>& indices) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument(
            "Number of indices must match tensor rank");
    }

    std::size_t offset = 0;
    for (std::size_t axis = 0; axis < rank(); ++axis) {
        if (indices[axis] >= shape_[axis]) {
            throw std::out_of_range("Tensor index is out of bounds");
        }
        offset += indices[axis] * strides_[axis];
    }
    return offset;
}
```

Every single access performs three layers of validation:

### 1. Rank Validation
- If the tensor has rank 3 (`{2, 3, 4}`) and you provide only 2 indices (`tensor.at({1, 2})`), it throws:
  `std::invalid_argument("Number of indices must match tensor rank")`.
- You cannot accidentally access a multidimensional tensor with too few or too many coordinates.

### 2. Bounds Checking
- Along each axis `axis`, the supplied index `indices[axis]` is checked against the maximum allowed bound `shape_[axis]`.
- Because indices are unsigned (`std::size_t`), negative indices passed as signed values wrap to huge positive numbers and are immediately caught.
- If `indices[axis] >= shape_[axis]`, it throws:
  `std::out_of_range("Tensor index is out of bounds")`.

### 3. Flat Vector Offset Calculation
- Once validated, the linear offset is computed:
  $$\text{offset} = \sum_{\text{axis}=0}^{\text{rank}-1} \text{indices}[\text{axis}] \times \text{strides}[\text{axis}]$$
- The element is returned using `data_.at(offset)`.

---

## 6. Scalar Tensor Semantics (Rank 0)

A scalar tensor represents a single dimensionless number.
In MiniTensor:
```cpp
Tensor() : shape_{}, strides_{}, data_(1, T{}) {}
```
- `shape_` is empty (`{}`), so `tensor.rank() == 0`.
- `strides_` is empty (`{}`).
- `data_` contains exactly 1 element initialized to `T{}` (e.g. `0` for `int`, `0.0f` for `float`).
- `tensor.size() == 1`.

To access or modify the scalar value:
```cpp
minitensor::Tensor<double> s;
s() = 42.0;          // Via operator()()
double val = s();     // Read value
```
When `operator()()` is invoked on a scalar:
1. It passes an empty index vector `{}` to `at()`.
2. `calculate_offset({})` checks `indices.size() == rank()` ($0 == 0$, which passes).
3. The loop runs 0 times; `offset = 0`.
4. It returns `data_.at(0)`.
Clean, mathematically consistent, and zero-overhead.

---

## 7. Element-Wise and Matrix Multiplication Helpers

### The Private `elementwise` Method Template
Binary operations (`+`, `-`, `*`) share the same internal engine:

```cpp
template <typename Operation>
[[nodiscard]] Tensor elementwise(
    const Tensor& other, Operation operation) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument(
            "Element-wise operations require identical tensor shapes");
    }

    std::vector<T> result_data;
    result_data.reserve(size());
    for (std::size_t index = 0; index < size(); ++index) {
        result_data.push_back(operation(data_[index], other.data_[index]));
    }
    return Tensor(shape_, std::move(result_data));
}
```
Key observations:
1. **Strict Shape Equality:** Both tensors must have the exact same shape vector (`shape_ == other.shape_`). There is no implicit broadcasting.
2. **Linear Memory Traversal:** Because both tensors share identical shapes and contiguous row-major layouts, the operation can be performed as a single flat 1D loop from `index = 0` to `size() - 1`. This allows the C++ compiler to automatically vectorize the loop using SIMD instructions (AVX2 / SSE / NEON).
3. **Pre-Allocation:** `result_data.reserve(size())` prevents heap reallocations.

### The Private `validate_matmul` Helper
```cpp
struct MatrixDimensions {
    std::size_t rows;
    std::size_t inner;
    std::size_t columns;
};

[[nodiscard]] MatrixDimensions validate_matmul(const Tensor& other) const {
    if (rank() != 2 || other.rank() != 2) {
        throw std::invalid_argument(
            "Matrix multiplication requires two-dimensional tensors");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument(
            "Matrix multiplication inner dimensions must match");
    }
    return {shape_[0], shape_[1], other.shape_[1]};
}
```
Matrix multiplication strictly requires:
1. Both operands must have `rank() == 2`.
2. The columns of the left matrix (`shape_[1]`) must equal the rows of the right matrix (`other.shape_[0]`).

---

## 8. Summary of Invariants

At all times during the lifetime of a `Tensor<T>`:
1. `shape_.size() == strides_.size() == rank()`.
2. If `rank() == 0`, `size() == 1` and `data_.size() == 1`.
3. If any `shape_[i] == 0`, `size() == 0` and all `strides_[i] == 0`.
4. If `size() > 0`, `strides_[rank - 1] == 1` (the last dimension is always contiguous with unit stride).
5. For any axis $k < \text{rank} - 1$, $\text{strides}[k] = \text{strides}[k+1] \times \text{shape}[k+1]$.
6. `data_.size() == calculate_size(shape_)`.

Proceed to:
- [Memory Management](memory_management.md) to understand RAII, copy independence, and cache locality.
- [Mathematical Operations](mathematical_operations.md) to explore the algorithms behind addition, multiplication, reshape, and transpose.
- [API Reference](api_reference.md) for full method documentation.

