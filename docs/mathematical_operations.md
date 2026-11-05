# Mathematical Operations in MiniTensor: Theory, Algorithms, and Complexity

This document provides a comprehensive mathematical, visual, and computational guide to every operation implemented in MiniTensor.

Whether you are performing basic element-by-element arithmetic, high-performance matrix multiplication, or geometric transformations like reshaping and transposing, this guide explains what happens mathematically, what the C++ code looks like, how the algorithm works under the hood, and how it scales with respect to time and memory.

---

## Table of Contents

1. [Element-Wise Operations vs. Linear Algebra: The Crucial Distinction](#1-element-wise-operations-vs-linear-algebra-the-crucial-distinction)
2. [Tensor Addition (`operator+`)](#2-tensor-addition-operator)
3. [Tensor Subtraction (`operator-`)](#3-tensor-subtraction-operator-)
4. [Element-Wise Multiplication (`operator*`)](#4-element-wise-multiplication-operator)
5. [The "No-Broadcasting" Rule](#5-the-no-broadcasting-rule)
6. [Matrix Multiplication (`matmul`)](#6-matrix-multiplication-matmul)
   - [Mathematical Definition and the Inner-Dimension Rule](#mathematical-definition-and-the-inner-dimension-rule)
   - [Worked Numerical Walkthrough](#worked-numerical-walkthrough)
   - [The Cache-Friendly `i-k-j` Loop Traversal](#the-cache-friendly-i-k-j-loop-traversal)
   - [Zero-Sized Inner Dimensions](#zero-sized-inner-dimensions)
7. [Cache-Blocked Matrix Multiplication (`matmul_blocked`)](#7-cache-blocked-matrix-multiplication-matmul_blocked)
   - [Why Tiling Matters for Large Matrices](#why-tiling-matters-for-large-matrices)
   - [The Algorithm and Block Size](#the-algorithm-and-block-size)
8. [Parallel Matrix Multiplication (`matmul_parallel`)](#8-parallel-matrix-multiplication-matmul_parallel)
   - [OpenMP Threading Model](#openmp-threading-model)
   - [Small-Workload Heuristic and Serial Fallback](#small-workload-heuristic-and-serial-fallback)
9. [Reshaping Tensors (`reshape`)](#9-reshaping-tensors-reshape)
10. [Transposing Matrices (`transpose`)](#10-transposing-matrices-transpose)
11. [Summary Table of Time and Space Complexity](#11-summary-table-of-time-and-space-complexity)

---

## 1. Element-Wise Operations vs. Linear Algebra: The Crucial Distinction

Before writing any code, every programmer must understand the difference between **element-wise operations** and **matrix multiplication**:

- **Element-Wise Operations (`+`, `-`, `*`):** The two input tensors must have the **exact same shape**. The operation is applied independently to each corresponding pair of numbers at the same coordinates.
- **Matrix Multiplication (`A.matmul(B)`):** This is the linear algebraic dot product of rows and columns. It applies **only to 2D matrices**, and the shapes do *not* have to be identical; rather, the number of columns of the first matrix must match the number of rows of the second matrix.

In MiniTensor:
- `A * B` is **element-wise multiplication** (the Hadamard product).
- `A.matmul(B)` is **matrix multiplication**.

---

## 2. Tensor Addition (`operator+`)

### A. Mathematical Explanation
For two tensors $A$ and $B$ of the same shape, each element in the resulting tensor $C$ is the sum of the corresponding elements:
$$C[i_0, i_1, \dots] = A[i_0, i_1, \dots] + B[i_0, i_1, \dots]$$

### B. Visual Explanation
$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} + \begin{pmatrix} 10 & 20 \\ 30 & 40 \end{pmatrix} = \begin{pmatrix} 1+10 & 2+20 \\ 3+30 & 4+40 \end{pmatrix} = \begin{pmatrix} 11 & 22 \\ 33 & 44 \end{pmatrix}$$

### C. C++ Code Example
```cpp
minitensor::Tensor<int> a({2, 2}, {1, 2, 3, 4});
minitensor::Tensor<int> b({2, 2}, {10, 20, 30, 40});

minitensor::Tensor<int> c = a + b;
// c contains {11, 22, 33, 44}
```

### D. Internal Implementation
Because both tensors have identical shapes and contiguous row-major storage, MiniTensor does not need to compute multidimensional strides. It simply executes a single flat loop over the data buffers:
```cpp
for (std::size_t index = 0; index < size(); ++index) {
    result_data.push_back(data_[index] + other.data_[index]);
}
```
This flat memory access pattern allows modern compilers to auto-vectorize the loop into SIMD vector instructions (e.g., AVX2 processing 8 floats per CPU cycle).

### E. Complexity
- **Time Complexity:** $O(N)$, where $N = \text{total number of elements}$. Every number is visited exactly once.
- **Space Complexity:** $O(N)$ for the returned owning tensor.

---

## 3. Tensor Subtraction (`operator-`)

### A. Mathematical Explanation
Subtracts each element of tensor $B$ from the corresponding element of tensor $A$:
$$C[i_0, i_1, \dots] = A[i_0, i_1, \dots] - B[i_0, i_1, \dots]$$

### B. Visual Explanation
$$\begin{pmatrix} 10 & 20 \\ 30 & 40 \end{pmatrix} - \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 9 & 18 \\ 27 & 36 \end{pmatrix}$$

### C. C++ Code Example
```cpp
minitensor::Tensor<int> a({2, 2}, {10, 20, 30, 40});
minitensor::Tensor<int> b({2, 2}, {1, 2, 3, 4});

minitensor::Tensor<int> c = a - b;
// c contains {9, 18, 27, 36}
```

### D. Internal Implementation and Complexity
- Implemented via the private `elementwise` helper with lambda `[](const T& left, const T& right) { return left - right; }`.
- **Time Complexity:** $O(N)$.
- **Space Complexity:** $O(N)$.

---

## 4. Element-Wise Multiplication (`operator*`)

### A. Mathematical Explanation
Multiplies corresponding elements of two identically shaped tensors. In mathematics, this is known as the **Hadamard product** (denoted $A \odot B$):
$$C[i_0, i_1, \dots] = A[i_0, i_1, \dots] \times B[i_0, i_1, \dots]$$

### B. Visual Explanation
$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \odot \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 1 \times 5 & 2 \times 6 \\ 3 \times 7 & 4 \times 8 \end{pmatrix} = \begin{pmatrix} 5 & 12 \\ 21 & 32 \end{pmatrix}$$

### C. C++ Code Example
```cpp
minitensor::Tensor<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
minitensor::Tensor<float> b({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

// Note: This is ELEMENT-WISE, not matrix multiplication!
minitensor::Tensor<float> c = a * b;
// c contains {5.0f, 12.0f, 21.0f, 32.0f}
```

### D. Internal Implementation and Complexity
- **Time Complexity:** $O(N)$.
- **Space Complexity:** $O(N)$.

---

## 5. The "No-Broadcasting" Rule

Many high-level libraries (such as NumPy or PyTorch) implement *broadcasting*, an automatic mechanism where a 1D tensor or scalar is implicitly stretched or replicated to match the shape of a higher-dimensional tensor.

MiniTensor **strictly does NOT perform implicit broadcasting**.

- If you attempt to add, subtract, or multiply tensors with different shapes (e.g., adding `{2, 2}` and `{2, 1}`, or adding a scalar `{}` to a matrix `{2, 3}`):
```cpp
minitensor::Tensor<int> matrix({2, 2}, 1);
minitensor::Tensor<int> vector({2}, 1);

// Throws std::invalid_argument:
auto result = matrix + vector; 
```
- Exception thrown: `std::invalid_argument("Element-wise operations require identical tensor shapes")`.
- **Why this design choice?** Explicit shapes eliminate an entire category of subtle, hard-to-detect bugs where dimensions are mismatched by mistake.

---

## 6. Matrix and Tensor Multiplication (`matmul`) Across Arbitrary Ranks

MiniTensor provides a generalized tensor multiplication engine that extends standard linear algebra across arbitrary dimensional ranks:

### Supported Rank Configurations

1. **Rank 1 with Rank 1 (Vector Dot Product):**
   Given $A \in \mathbb{R}^K$ and $B \in \mathbb{R}^K$, computes the inner scalar dot product:
   $$C = \sum_{k=0}^{K-1} A_k \cdot B_k$$
   Returns a **Rank 0 scalar tensor** of shape `{}`.

2. **Rank 2 with Rank 1 (Matrix-Vector Product):**
   Given $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^K$, computes:
   $$C_m = \sum_{k=0}^{K-1} A_{m, k} \cdot B_k$$
   Returns a **Rank 1 tensor** of shape `{M}`.

3. **Rank 1 with Rank 2 (Vector-Matrix Product):**
   Given $A \in \mathbb{R}^K$ and $B \in \mathbb{R}^{K \times N}$, computes:
   $$C_n = \sum_{k=0}^{K-1} A_k \cdot B_{k, n}$$
   Returns a **Rank 1 tensor** of shape `{N}`.

4. **Rank 2 with Rank 2 (Standard Matrix Multiplication):**
   Given $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, computes:
   $$C_{m, n} = \sum_{k=0}^{K-1} A_{m, k} \cdot B_{k, n}$$
   Returns a **Rank 2 tensor** of shape `{M, N}`.

5. **Rank 3+ (Arbitrary-Rank Batched Matrix Multiplication):**
   Given tensors of rank $\ge 2$ with leading batch dimensions $(B_0, \dots, B_{p-1})$:
   $$(B_0, \dots, B_{p-1}, M, K) \times (B_0, \dots, B_{p-1}, K, N) \to (B_0, \dots, B_{p-1}, M, N)$$
   For each batch coordinate, the 2D matrix slices are multiplied independently.

6. **Batch Dimension Broadcasting:**
   If one operand has batch dimensions (e.g. $(B \times M \times K)$) and the other is a 2D matrix ($(K \times N)$), MiniTensor automatically broadcasts the 2D matrix across all batches of the higher-rank tensor, producing shape $(B \times M \times N)$ without data duplication.

### Mathematical Definition and the Inner-Dimension Rule

In all cases, the contraction dimension (the inner dimension) must match:
$$\text{Inner Dimension of } A == \text{Inner Dimension of } B$$

Each element $C[i, j]$ is computed by taking Row $i$ of matrix $A$, Column $j$ of matrix $B$, multiplying their matching entries, and adding up the results (the dot product):

$$C[i, j] = \sum_{k=0}^{K-1} A[i, k] \times B[k, j]$$

### Worked Numerical Walkthrough

Let us multiply a $(2 \times 3)$ matrix by a $(3 \times 2)$ matrix:

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}_{(2 \times 3)}, \quad B = \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}_{(3 \times 2)}$$

The inner dimension is $3$. The result shape is $(2 \times 2)$:

1. **Calculate $C[0, 0]$ (Row 0 of A, Col 0 of B):**
   $$(1 \times 7) + (2 \times 9) + (3 \times 11) = 7 + 18 + 33 = 58$$
2. **Calculate $C[0, 1]$ (Row 0 of A, Col 1 of B):**
   $$(1 \times 8) + (2 \times 10) + (3 \times 12) = 8 + 20 + 36 = 64$$
3. **Calculate $C[1, 0]$ (Row 1 of A, Col 0 of B):**
   $$(4 \times 7) + (5 \times 9) + (6 \times 11) = 28 + 45 + 66 = 139$$
4. **Calculate $C[1, 1]$ (Row 1 of A, Col 1 of B):**
   $$(4 \times 8) + (5 \times 10) + (6 \times 12) = 32 + 50 + 72 = 154$$

$$C = \begin{pmatrix} 58 & 64 \\ 139 & 154 \end{pmatrix}$$

### The Cache-Friendly `i-k-j` Loop Traversal

In classic textbook pseudocode, matrix multiplication is taught with three nested loops ordered $i \rightarrow j \rightarrow k$:
```text
// Textbook algorithm (Bad for CPU cache!):
for i:
    for j:
        for k:
            C[i, j] += A[i, k] * B[k, j] // B[k, j] strides vertically across memory!
```
In row-major order, accessing `B[k, j]` while incrementing $k$ jumps across memory by an entire row on every iteration. This causes continuous **cache misses** for matrix $B$.

MiniTensor implements the highly optimized **`i-k-j` loop reordering**:

```cpp
for (std::size_t row = 0; row < dimensions.rows; ++row) {
    for (std::size_t inner = 0; inner < dimensions.inner; ++inner) {
        const T left = data_[row * dimensions.inner + inner];
        for (std::size_t column = 0; column < dimensions.columns; ++column) {
            result.data_[row * dimensions.columns + column] +=
                left * other.data_[inner * dimensions.columns + column];
        }
    }
}
```

Notice what happens in the innermost loop over `column`:
- `left` is cached in a CPU register.
- `other.data_` is accessed sequentially: `[inner * cols + column]` advances by $1$ on each step!
- `result.data_` is also accessed sequentially: `[row * cols + column]` advances by $1$ on each step!
- Both arrays stream through memory in perfect contiguous order, allowing full utilization of CPU cache lines and SIMD vector registers.

### Zero-Sized Inner Dimensions
If the inner dimension is $0$ (for example, multiplying $(2 \times 0)$ by $(0 \times 3)$), there are 0 terms in the summation. The result is a $(2 \times 3)$ matrix filled with zeros (`T{}`), which MiniTensor handles correctly without performing any invalid loops.

### C++ Code Example
```cpp
minitensor::Tensor<double> a({2, 3}, {1, 2, 3, 4, 5, 6});
minitensor::Tensor<double> b({3, 2}, {7, 8, 9, 10, 11, 12});

minitensor::Tensor<double> c = a.matmul(b);
// c has shape {2, 2} and elements {58, 64, 139, 154}
```

### Complexity
- **Time Complexity:** $O(M \times K \times N)$. For two $N \times N$ square matrices, this is $O(N^3)$.
- **Space Complexity:** $O(M \times N)$ for the returned result matrix.

---

## 7. Cache-Blocked Matrix Multiplication (`matmul_blocked`)

### Why Tiling Matters for Large Matrices
When matrices are small (e.g., $64 \times 64$), all matrices fit entirely into the CPU's fast L1/L2 data cache.
However, when matrices grow to $512 \times 512$ or $1024 \times 1024$, they require several megabytes of RAM. The CPU cache fills up, and data from earlier rows gets evicted before it can be reused.

**Cache Tiling (Blocking)** divides large matrices into small sub-blocks of size $B \times B$ (by default, `block_size = 64`). The algorithm performs all multiplications for one local tile while its data is still hot in the CPU cache, drastically reducing memory bus traffic to main RAM.

```text
Full Matrix (e.g., 512 x 512)            Divided into 64x64 Blocks:
+---------------------------------+     +-----+-----+-----+-----+
|                                 |     | B00 | B01 | B02 | ... |
|                                 |     +-----+-----+-----+-----+
|                                 | --> | B10 | B11 | B12 | ... |
|                                 |     +-----+-----+-----+-----+
|                                 |     | ... |     |     |     |
+---------------------------------+     +-----+-----+-----+-----+
```

### The Algorithm and Block Size
In `include/minitensor/tensor.hpp`:
```cpp
Tensor matmul_blocked(const Tensor& other, std::size_t block_size = 64) const;
```
- Validates that `block_size > 0` (throws `std::invalid_argument` if 0).
- Tiles the `row` and `inner` dimensions in steps of `block_size`.
- Keeps the inner `column` loop contiguous for maximum vectorization.

---

## 8. Parallel Matrix Multiplication (`matmul_parallel`)

### OpenMP Threading Model
Modern desktop and server CPUs have 4, 8, 16, or more physical cores. Running on a single thread wastes the computational power of the remaining cores.

MiniTensor provides `matmul_parallel(const Tensor& other, std::size_t block_size = 64)`.

When compiled with OpenMP (`-DMINITENSOR_ENABLE_OPENMP=ON`):
```cpp
#pragma omp parallel for if(large_workload) schedule(static)
for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(dimensions.rows); ++row) {
    // Process row using blocked inner loops...
}
```
- OpenMP distributes the rows of the output matrix across available hardware worker threads.
- Because each thread writes to its own distinct rows, there are **no race conditions, no locks, and no mutexes** required.

### Small-Workload Heuristic and Serial Fallback
1. **Small Workloads:** Launching and coordinating threads has a small operating system overhead. If the matrix is smaller than $64 \times 64 \times 64$, MiniTensor sets `large_workload = false`, causing OpenMP to execute the loop serially without thread spawn overhead.
2. **Serial Fallback:** If the library was compiled without OpenMP support (`MINITENSOR_ENABLE_OPENMP=OFF`), `matmul_parallel()` automatically and gracefully falls back to the serial `matmul()` implementation. Your code compiles and runs seamlessly on any platform!

---

## 9. Reshaping Tensors (`reshape`)

### A. Mathematical Explanation
Reshaping reorganizes the dimensions of a tensor without changing its values or their relative ordering. 
The only requirement is that the total number of elements must remain identical:
$$\prod \text{old\_dimensions} = \prod \text{new\_dimensions}$$

### B. Visual Explanation
Transforming a $2 \times 3$ matrix into a $3 \times 2$ matrix:
$$\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}_{(2 \times 3)} \xrightarrow{\text{reshape}} \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}_{(3 \times 2)}$$
Notice that the row-major sequence `[1, 2, 3, 4, 5, 6]` is completely unchanged!

### C. C++ Code Example
```cpp
minitensor::Tensor<int> matrix({2, 3}, {1, 2, 3, 4, 5, 6});

// Reshape to 3 rows, 2 columns:
minitensor::Tensor<int> reshaped = matrix.reshape({3, 2});

// Reshape to a 1D vector of 6 elements:
minitensor::Tensor<int> flat_vector = matrix.reshape({6});
```

### D. Implementation Details and Validation
- Checks `calculate_size(new_shape) != size()`. If the element counts do not match, it throws `std::invalid_argument("Reshaped tensor must have the same number of elements")`.
- **Note on Memory:** In MiniTensor, `reshape()` returns a **new owning `Tensor<T>`** that copies the data buffer. It does not return a non-owning view.
- **Complexity:** $O(N)$ time (copying $N$ elements) and $O(N)$ space.

---

## 10. Transposing Matrices (`transpose`)

### A. Mathematical Explanation
Transposing flips a 2D matrix over its diagonal, switching its rows and columns.
If matrix $A$ has shape $(M \times N)$, then $A^T$ has shape $(N \times M)$:
$$A^T[j, i] = A[i, j]$$

### B. Visual Explanation
$$\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}_{(2 \times 3)} \xrightarrow{\text{transpose}} \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}_{(3 \times 2)}$$
Notice how Row 0 (`1, 2, 3`) becomes Column 0, and Row 1 (`4, 5, 6`) becomes Column 1.

### C. C++ Code Example
```cpp
minitensor::Tensor<int> matrix({2, 3}, {1, 2, 3, 4, 5, 6});

minitensor::Tensor<int> transposed = matrix.transpose();
// transposed has shape {3, 2} and elements: {1, 4, 2, 5, 3, 6}
```

### D. Implementation Details and Restrictions
- **Rank Restriction:** MiniTensor currently supports `transpose()` **only for Rank 2 matrices**. Calling `transpose()` on a 1D vector or 3D tensor throws:
  `std::invalid_argument("Transpose is supported only for two-dimensional tensors")`.
- It allocates a new tensor of shape `{shape_[1], shape_[0]}` and writes the re-indexed values:
  `result.data_[col * shape_[0] + row] = data_[row * shape_[1] + col];`.
- **Complexity:** $O(M \times N)$ time and $O(M \times N)$ space.

---

## 11. General Tensor Contraction (`tensordot`)

### A. Mathematical Explanation
`tensordot` computes tensor contractions along arbitrary specified axes:
- `A.tensordot(B, axes)`: contracts the last `axes` dimensions of $A$ with the first `axes` dimensions of $B$.
- `A.tensordot(B, axes_a, axes_b)`: contracts specific axis pairs $(a_i, b_i)$.

### B. C++ Code Example
```cpp
minitensor::Tensor<int> a({2, 3, 4}, 1);
minitensor::Tensor<int> b({4, 3, 5}, 2);

// Contract axis 2 of a with axis 0 of b, and axis 1 of a with axis 1 of b:
auto result = a.tensordot(b, {2, 1}, {0, 1}); // Result shape: {2, 5}
```

---

## 12. Summary Table of Time and Space Complexity

| Operation | C++ Method / Operator | Required Ranks / Constraints | Time Complexity | Extra Space Complexity |
| :--- | :--- | :--- | :---: | :---: |
| **Addition** | `A + B` | Identical shapes required | $O(N)$ | $O(N)$ |
| **Subtraction** | `A - B` | Identical shapes required | $O(N)$ | $O(N)$ |
| **Element Multiplication** | `A * B` | Identical shapes required | $O(N)$ | $O(N)$ |
| **Matrix/Tensor Multiplication** | `A.matmul(B)` | Rank $\ge 1$; inner dimensions must match | $O(B_{\text{total}} \cdot M \cdot K \cdot N)$ | $O(B_{\text{total}} \cdot M \cdot N)$ |
| **Blocked Tensor Mul** | `A.matmul_blocked(B, block)` | Rank $\ge 1$; `block_size > 0` | $O(B_{\text{total}} \cdot M \cdot K \cdot N)$ | $O(B_{\text{total}} \cdot M \cdot N)$ |
| **Parallel Tensor Mul** | `A.matmul_parallel(B, block)`| Rank $\ge 1$; OpenMP optional | $O(\frac{B_{\text{total}} \cdot M \cdot K \cdot N}{P})^*$ | $O(B_{\text{total}} \cdot M \cdot N)$ |
| **Tensor Contraction** | `A.tensordot(B, axes_a, axes_b)` | Matching contracted dimension lengths | $O(\text{size}_A \cdot \text{size}_B / K)$ | $O(\text{result\_size})$ |
| **Reshape** | `A.reshape(new_shape)` | New shape product == old size | $O(N)$ | $O(N)$ |
| **Transpose** | `A.transpose()` | Rank 2 only | $O(M \times N)$ | $O(M \times N)$ |

*\* Where $P$ is the number of active OpenMP worker threads (up to available CPU cores).*

---

Proceed to:
- [Complete API Reference](api_reference.md) for detailed signature specifications.
- [Usage Tutorial](usage_tutorial.md) for complete, runnable code examples.
- [Performance and Optimization](performance.md) for cache analysis and benchmark data.

