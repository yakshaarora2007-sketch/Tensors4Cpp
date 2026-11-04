# MiniTensor Complete API Reference

This document provides an exhaustive, function-by-function, and operator-by-operator specification of the public API of MiniTensor. Every entry reflects the exact signatures implemented in `include/minitensor/tensor.hpp`.

---

## Namespace and Header

```cpp
#include <minitensor/minitensor.hpp>
// or individually:
#include <minitensor/tensor.hpp>
#include <minitensor/indexing.hpp>
#include <minitensor/linalg.hpp>
#include <minitensor/operations.hpp>

namespace minitensor {
    template <typename T>
    class Tensor;
}
```

---

## Quick Navigation

- [Constructors and Lifetime](#constructors-and-lifetime)
  - [`Tensor()`](#tensor)
  - [`Tensor(shape)`](#tensorshape)
  - [`Tensor(shape, initial_value)`](#tensorshape-initial_value)
  - [`Tensor(shape, initial_data)`](#tensorshape-initial_data)
  - [Copy & Move Constructors / Operators](#copy-and-move-semantics)
- [Introspection and Metadata](#introspection-and-metadata)
  - [`size()`](#size)
  - [`rank()`](#rank)
  - [`get_shape()`](#get_shape)
  - [`get_strides()`](#get_strides)
  - [`get_data()`](#get_data)
- [Data Modification](#data-modification)
  - [`fill(value)`](#fillvalue)
  - [`set_data(values)`](#set_datavalues)
- [Element Access and Indexing](#element-access-and-indexing)
  - [`at(indices)`](#atindices)
  - [`set(indices, value)`](#setindices-value)
  - [`operator()()`](#operator)
  - [`operator()(indices...)`](#operatorindices)
- [Mathematical Operators](#mathematical-operators)
  - [`operator+`](#operator-addition)
  - [`operator-`](#operator--subtraction)
  - [`operator*`](#operator-element-wise-multiplication)
- [Linear Algebra and Matrix Multiplication](#linear-algebra-and-matrix-multiplication)
  - [`matmul(other)`](#matmulother)
  - [`matmul_blocked(other, block_size)`](#matmul_blockedother-block_size)
  - [`matmul_parallel(other, block_size)`](#matmul_parallelother-block_size)
- [Shape Transformations](#shape-transformations)
  - [`reshape(new_shape)`](#reshapenew_shape)
  - [`transpose()`](#transpose)

---

## Constructors and Lifetime

### `Tensor()`
```cpp
Tensor();
```
- **Description:** Default constructor. Constructs a rank-0 **scalar** tensor containing a single value-initialized element (`T{}`).
- **Why use it:** When you need a single numerical scalar wrapped in a tensor container, or as a default-constructed placeholder.
- **Parameters:** None.
- **Return Value:** None (constructor).
- **Exceptions:** Does not throw (`noexcept`-compatible allocation for 1 element).
- **Example:**
  ```cpp
  minitensor::Tensor<double> scalar;
  std::cout << "Rank: " << scalar.rank() << ", Size: " << scalar.size() << '\n';
  // Output: Rank: 0, Size: 1
  ```

---

### `Tensor(shape)`
```cpp
explicit Tensor(std::vector<std::size_t> shape);
```
- **Description:** Constructs a tensor with the given shape, value-initializing all elements to `T{}` (e.g. `0` for numeric types).
- **Why use it:** When creating a zero-initialized tensor of any rank and dimension.
- **Parameters:**
  - `shape`: Vector of dimension lengths, e.g. `{2, 3}` for a $2 \times 3$ matrix.
- **Exceptions:**
  - `std::length_error`: If the product of the shape dimensions exceeds `std::numeric_limits<std::size_t>::max()`.
- **Example:**
  ```cpp
  minitensor::Tensor<float> matrix({2, 3});
  // matrix is 2x3 filled with 0.0f
  ```

---

### `Tensor(shape, initial_value)`
```cpp
Tensor(std::vector<std::size_t> shape, const T& initial_value);
```
- **Description:** Constructs a tensor with the specified shape and fills every element with `initial_value`.
- **Why use it:** To create a tensor filled with a constant number (e.g., all ones, all `-1`, etc.).
- **Parameters:**
  - `shape`: Vector of dimension lengths.
  - `initial_value`: Value copied to every element.
- **Exceptions:**
  - `std::length_error`: If dimension product overflows `size_t`.
- **Example:**
  ```cpp
  minitensor::Tensor<int> ones({3, 3}, 1);
  // ones is 3x3 filled with 1
  ```

---

### `Tensor(shape, initial_data)`
```cpp
Tensor(std::vector<std::size_t> shape, std::vector<T> initial_data);
```
- **Description:** Constructs a tensor with the given shape and transfers `initial_data` into the internal buffer.
- **Why use it:** When you already have a flat vector of values (from disk, network, or computation) that you want to structure as a tensor.
- **Parameters:**
  - `shape`: Desired shape vector.
  - `initial_data`: Vector of values in row-major order.
- **Exceptions:**
  - `std::invalid_argument`: If `initial_data.size()` does not exactly equal the calculated size of `shape`.
  - `std::length_error`: If dimension product overflows `size_t`.
- **Example:**
  ```cpp
  minitensor::Tensor<int> matrix({2, 2}, std::vector<int>{1, 2, 3, 4});
  ```

---

### Copy and Move Semantics
```cpp
Tensor(const Tensor& other);            // Copy constructor (Deep copy)
Tensor(Tensor&& other) noexcept;        // Move constructor (O(1) pointer transfer)
Tensor& operator=(const Tensor& other); // Copy assignment (Deep copy)
Tensor& operator=(Tensor&& other) noexcept; // Move assignment (O(1))
```
- **Description:** Standard value semantics. Copies allocate independent buffers; moves transfer internal vector buffers in $O(1)$ time with zero allocations.
- **Example:**
  ```cpp
  minitensor::Tensor<float> a({100, 100}, 1.0f);
  minitensor::Tensor<float> b = a;            // Deep copy: b owns new buffer
  minitensor::Tensor<float> c = std::move(a); // Fast move: c takes a's buffer
  ```

---

## Introspection and Metadata

### `size()`
```cpp
[[nodiscard]] std::size_t size() const noexcept;
```
- **Description:** Returns the total number of elements in the tensor.
- **Return Value:** `std::size_t` representing the element count.
- **Example:**
  ```cpp
  minitensor::Tensor<int> t({2, 3, 4});
  std::cout << t.size() << '\n'; // 24
  ```

---

### `rank()`
```cpp
[[nodiscard]] std::size_t rank() const noexcept;
```
- **Description:** Returns the number of dimensions/axes.
- **Return Value:** `std::size_t` representing `shape_.size()`.
- **Example:**
  ```cpp
  minitensor::Tensor<int> t({2, 3, 4});
  std::cout << t.rank() << '\n'; // 3
  ```

---

### `get_shape()`
```cpp
[[nodiscard]] const std::vector<std::size_t>& get_shape() const noexcept;
```
- **Description:** Returns an immutable reference to the shape vector.
- **Return Value:** `const std::vector<std::size_t>&`.
- **Example:**
  ```cpp
  minitensor::Tensor<int> t({2, 3});
  for (std::size_t dim : t.get_shape()) {
      std::cout << dim << " "; // 2 3 
  }
  ```

---

### `get_strides()`
```cpp
[[nodiscard]] const std::vector<std::size_t>& get_strides() const noexcept;
```
- **Description:** Returns an immutable reference to the row-major strides.
- **Return Value:** `const std::vector<std::size_t>&`.
- **Example:**
  ```cpp
  minitensor::Tensor<int> t({2, 3, 4});
  // t.get_strides() contains {12, 4, 1}
  ```

---

### `get_data()`
```cpp
[[nodiscard]] const std::vector<T>& get_data() const noexcept;
```
- **Description:** Returns an immutable reference to the underlying contiguous flat storage buffer.
- **Return Value:** `const std::vector<T>&`.
- **Example:**
  ```cpp
  minitensor::Tensor<int> t({2, 2}, {1, 2, 3, 4});
  const auto& data = t.get_data(); // Direct read-only access to vector
  ```

---

## Data Modification

### `fill(value)`
```cpp
void fill(const T& value);
```
- **Description:** Overwrites every element in the tensor with `value`.
- **Parameters:**
  - `value`: The value to assign to each element.
- **Example:**
  ```cpp
  minitensor::Tensor<int> t({2, 2});
  t.fill(7); // Every element is now 7
  ```

---

### `set_data(values)`
```cpp
void set_data(const std::vector<T>& values);
```
- **Description:** Replaces the entire payload buffer with `values`. Provides the strong exception guarantee.
- **Parameters:**
  - `values`: Vector of new values. Must match `this->size()`.
- **Exceptions:**
  - `std::invalid_argument`: If `values.size() != size()`.
- **Example:**
  ```cpp
  minitensor::Tensor<int> t({2, 2}, 0);
  t.set_data({10, 20, 30, 40});
  ```

---

## Element Access and Indexing

### `at(indices)`
```cpp
[[nodiscard]] T& at(const std::vector<std::size_t>& indices);
[[nodiscard]] const T& at(const std::vector<std::size_t>& indices) const;
[[nodiscard]] T& at(std::initializer_list<std::size_t> indices);
[[nodiscard]] const T& at(std::initializer_list<std::size_t> indices) const;
```
- **Description:** Bounds-checked element access using a coordinate vector or initializer list.
- **Parameters:**
  - `indices`: Coordinate list, e.g. `{1, 2}`.
- **Return Value:** Reference to the element at the specified coordinate.
- **Exceptions:**
  - `std::invalid_argument`: If `indices.size() != rank()`.
  - `std::out_of_range`: If any coordinate is $\ge$ the dimension length.
- **Example:**
  ```cpp
  minitensor::Tensor<float> t({2, 3}, 0.0f);
  t.at({0, 1}) = 4.2f;
  float val = t.at({0, 1}); // 4.2f
  ```

---

### `set(indices, value)`
```cpp
void set(const std::vector<std::size_t>& indices, const T& value);
void set(std::initializer_list<std::size_t> indices, const T& value);
```
- **Description:** Assigns `value` to the element at `indices`. Convenience wrapper around `at(indices) = value`.
- **Parameters:**
  - `indices`: Coordinate list.
  - `value`: Value to store.
- **Exceptions:** Same as `at()`.
- **Example:**
  ```cpp
  minitensor::Tensor<int> t({2, 2});
  t.set({1, 0}, 99);
  ```

---

### `operator()()`
```cpp
[[nodiscard]] T& operator()();
[[nodiscard]] const T& operator()() const;
```
- **Description:** Accesses the single value of a **scalar tensor** (Rank 0, shape `{}`).
- **Return Value:** Reference to the scalar value.
- **Exceptions:**
  - `std::invalid_argument`: If called on a non-scalar tensor (rank $> 0$).
- **Example:**
  ```cpp
  minitensor::Tensor<double> scalar;
  scalar() = 3.14;
  std::cout << scalar() << '\n'; // 3.14
  ```

---

### `operator()(indices...)`
```cpp
template <typename... Indices>
    requires(sizeof...(Indices) > 0 &&
             (std::is_convertible_v<Indices, std::size_t> && ...))
[[nodiscard]] T& operator()(Indices... indices);

template <typename... Indices>
    requires(sizeof...(Indices) > 0 &&
             (std::is_convertible_v<Indices, std::size_t> && ...))
[[nodiscard]] const T& operator()(Indices... indices) const;
```
- **Description:** Variadic multidimensional call operator. Enables intuitive coordinate access: `t(row, col)`.
- **Parameters:** Variadic arguments convertible to `std::size_t`.
- **Return Value:** Reference to the element.
- **Exceptions:**
  - `std::invalid_argument`: If argument count does not equal `rank()`.
  - `std::out_of_range`: If any coordinate is out of bounds.
- **Example:**
  ```cpp
  minitensor::Tensor<int> matrix({3, 3}, 0);
  matrix(1, 2) = 42;
  int x = matrix(1, 2); // 42
  ```

---

## Mathematical Operators

### `operator+` (Addition)
```cpp
[[nodiscard]] Tensor operator+(const Tensor& other) const;
```
- **Description:** Computes the element-wise sum of two tensors.
- **Exceptions:**
  - `std::invalid_argument`: If `shape_ != other.shape_`.
- **Example:**
  ```cpp
  auto c = a + b;
  ```

---

### `operator-` (Subtraction)
```cpp
[[nodiscard]] Tensor operator-(const Tensor& other) const;
```
- **Description:** Computes the element-wise difference of two tensors.
- **Exceptions:**
  - `std::invalid_argument`: If `shape_ != other.shape_`.
- **Example:**
  ```cpp
  auto c = a - b;
  ```

---

### `operator*` (Element-Wise Multiplication)
```cpp
[[nodiscard]] Tensor operator*(const Tensor& other) const;
```
- **Description:** Computes the element-wise Hadamard product of two tensors. **(Not matrix multiplication!)**
- **Exceptions:**
  - `std::invalid_argument`: If `shape_ != other.shape_`.
- **Example:**
  ```cpp
  auto c = a * b; // Element-by-element
  ```

---

## Linear Algebra and Matrix Multiplication

### `matmul(other)`
```cpp
[[nodiscard]] Tensor matmul(const Tensor& other) const;
```
- **Description:** Performs serial matrix multiplication between two 2D matrices using a cache-friendly row-major `i-k-j` algorithm.
- **Parameters:**
  - `other`: Right-hand matrix $(K \times N)$.
- **Return Value:** New `Tensor<T>` of shape $(M \times N)$.
- **Exceptions:**
  - `std::invalid_argument`: If either tensor is not Rank 2 (`rank() != 2`).
  - `std::invalid_argument`: If inner dimensions do not match (`shape_[1] != other.shape_[0]`).
- **Example:**
  ```cpp
  minitensor::Tensor<float> a({2, 3}, 1.0f);
  minitensor::Tensor<float> b({3, 4}, 2.0f);
  auto c = a.matmul(b); // Shape {2, 4}
  ```

---

### `matmul_blocked(other, block_size)`
```cpp
[[nodiscard]] Tensor matmul_blocked(const Tensor& other, std::size_t block_size = 64) const;
```
- **Description:** Performs cache-blocked (tiled) matrix multiplication, keeping active sub-matrices inside the CPU L1/L2 cache.
- **Parameters:**
  - `other`: Right-hand matrix.
  - `block_size`: Dimension tile size (default `64`). Must be $> 0$.
- **Exceptions:**
  - `std::invalid_argument`: If `block_size == 0`.
  - Same dimension exceptions as `matmul()`.
- **Example:**
  ```cpp
  auto c = a.matmul_blocked(b, 64);
  ```

---

### `matmul_parallel(other, block_size)`
```cpp
[[nodiscard]] Tensor matmul_parallel(const Tensor& other, std::size_t block_size = 64) const;
```
- **Description:** Performs multithreaded matrix multiplication using OpenMP (when compiled with `MINITENSOR_ENABLE_OPENMP=ON`). Falls back to `matmul(other)` if OpenMP is not enabled or for small workloads ($< 64 \times 64 \times 64$).
- **Parameters:**
  - `other`: Right-hand matrix.
  - `block_size`: Inner block size (default `64`).
- **Exceptions:** Same as `matmul_blocked()`.
- **Example:**
  ```cpp
  auto c = a.matmul_parallel(b);
  ```

---

## Shape Transformations

### `reshape(new_shape)`
```cpp
[[nodiscard]] Tensor reshape(const std::vector<std::size_t>& new_shape) const;
```
- **Description:** Creates a new owning tensor with `new_shape` containing the exact same data in the same row-major order.
- **Parameters:**
  - `new_shape`: Desired shape vector.
- **Return Value:** New owning `Tensor<T>` with `new_shape`.
- **Exceptions:**
  - `std::invalid_argument`: If `calculate_size(new_shape) != this->size()`.
- **Example:**
  ```cpp
  minitensor::Tensor<int> a({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = a.reshape({3, 2});
  ```

---

### `transpose()`
```cpp
[[nodiscard]] Tensor transpose() const;
```
- **Description:** Returns the transposed matrix with swapped rows and columns.
- **Restrictions:** **Only supported for Rank-2 tensors.**
- **Return Value:** New `Tensor<T>` of shape `{shape_[1], shape_[0]}`.
- **Exceptions:**
  - `std::invalid_argument`: If `rank() != 2`.
- **Example:**
  ```cpp
  minitensor::Tensor<int> a({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = a.transpose(); // b has shape {3, 2}
  ```

---

Proceed to:
- [Usage Tutorial](usage_tutorial.md) for 15 progressive hands-on coding examples.
- [Mathematical Operations](mathematical_operations.md) for theory and complexity analysis.
- [Testing Guide](testing.md) for error handling and test verification.

