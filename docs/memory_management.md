# Memory Management and Safety in MiniTensor

Memory management is one of the most critical topics in systems programming. In languages like C and C++, poor memory management leads to memory leaks, dangling pointers, buffer overflows, and segmentation faults.

MiniTensor was designed with modern C++ memory management principles from day one. It guarantees **100% leak-free, exception-safe, deterministic memory management** without requiring the programmer to write a single `new`, `delete`, `malloc`, or `free`.

This document explains the memory architecture of MiniTensor, how RAII works, the difference between copying and moving tensors, and why contiguous storage provides exceptional hardware performance.

---

## Table of Contents

1. [Modern C++ Memory Management: The RAII Principle](#1-modern-c-memory-management-the-raii-principle)
   - [What is RAII?](#what-is-raii)
   - [How `std::vector` Eliminates Leaks](#how-stdvector-eliminates-leaks)
2. [Deep Copies vs. Move Semantics](#2-deep-copies-vs-move-semantics)
   - [Copy Construction and Copy Assignment: Total Independence](#copy-construction-and-copy-assignment-total-independence)
   - [Move Construction and Move Assignment: Blazing Fast Transfers](#move-construction-and-move-assignment-blazing-fast-transfers)
3. [The Strong Exception Guarantee: Safe Buffer Swapping](#3-the-strong-exception-guarantee-safe-buffer-swapping)
4. [Hardware Performance: Why Contiguous Memory Matters](#4-hardware-performance-why-contiguous-memory-matters)
   - [The Memory Hierarchy: Registers, Caches, and RAM](#the-memory-hierarchy-registers-caches-and-ram)
   - [Spatial Locality and Cache Line Prefetching](#spatial-locality-and-cache-line-prefetching)
   - [The "Pointer-Chasing" Tax of Nested Vectors](#the-pointer-chasing-tax-of-nested-vectors)
5. [Summary Comparison: Memory Models](#5-summary-comparison-memory-models)

---

## 1. Modern C++ Memory Management: The RAII Principle

### What is RAII?
**RAII** stands for *Resource Acquisition Is Initialization*. It is the foundational idiom of modern C++:
- When an object is created on the stack (its lifetime begins), it acquires the resources it needs (such as allocating heap memory).
- When the object goes out of scope (e.g., exiting a function, leaving an `if` block, or unwinding the stack due to an exception), its destructor is automatically called, instantly releasing all acquired resources.

You never have to remember to "clean up" or "free" a tensor. It cleans up after itself automatically.

### How `std::vector` Eliminates Leaks
MiniTensor uses `std::vector` for all its storage:
- `std::vector<std::size_t> shape_`
- `std::vector<std::size_t> strides_`
- `std::vector<T> data_`

```cpp
void process_data() {
    minitensor::Tensor<float> my_tensor({1000, 1000}); // Allocates 1,000,000 floats (~4 MB)
    my_tensor.fill(1.0f);
    
    // ... do calculations ...
    
    if (some_error_condition) {
        throw std::runtime_error("Something went wrong!");
        // Even if an exception is thrown, my_tensor is completely destroyed,
        // and its 4 MB of heap memory is returned to the OS immediately!
    }
} // When exiting normally, memory is also freed automatically here.
```

There are no raw pointers (`T*`), no manual calls to `delete[]`, and zero risk of memory leaks.

---

## 2. Deep Copies vs. Move Semantics

Understanding the difference between **copying** and **moving** a tensor is essential for writing efficient C++ code.

### Copy Construction and Copy Assignment: Total Independence

When you copy a `Tensor<T>`, MiniTensor performs a **deep copy**. The new tensor allocates its own brand new heap buffer and duplicates every element from the original:

```cpp
minitensor::Tensor<int> tensor_a({3}, {10, 20, 30});

// Copy constructor:
minitensor::Tensor<int> tensor_b = tensor_a;

// Modify tensor_b:
tensor_b.set({0}, 999);

// tensor_a is completely untouched!
// tensor_a contains {10, 20, 30}
// tensor_b contains {999, 20, 30}
```

```text
tensor_a.data_ ---> [ Heap Address 0x1000 ]: [ 10,  20,  30 ]
                         (Independent)
tensor_b.data_ ---> [ Heap Address 0x2000 ]: [ 999, 20,  30 ]
```

Because MiniTensor does not use a shared-buffer view model, there is no risk of one function silently mutating the data of another tensor elsewhere in your program.

### Move Construction and Move Assignment: Blazing Fast Transfers

What if you have a huge tensor (say, a 1 gigabyte matrix) and you want to return it from a function? Making a deep copy would duplicate 1 GB of RAM and take hundreds of milliseconds.

This is where **C++ Move Semantics** shine!

When an object is moved, its internal pointers are simply transferred to the new object, and the old object is left in an empty state. This takes less than **1 nanosecond**, regardless of how many gigabytes the tensor contains!

```cpp
minitensor::Tensor<float> create_large_matrix() {
    minitensor::Tensor<float> large({5000, 5000}); // 25,000,000 floats (100 MB)
    large.fill(1.0f);
    return large; // Return by value: automatically MOVES! (Zero copy!)
}

int main() {
    // Moved directly into matrix; NO copying occurs:
    minitensor::Tensor<float> matrix = create_large_matrix();
    
    // You can also explicitly move an existing tensor:
    minitensor::Tensor<float> destination = std::move(matrix);
    // Now 'destination' owns the 100 MB buffer.
    // 'matrix' is empty.
}
```

```text
Before Move:
  matrix.data_      ---> [ Heap Address 0x5000 ]: [ 100 MB of data ]
  destination.data_ ---> [ Unallocated ]

After Move:
  matrix.data_      ---> [ Empty / nullptr ]
  destination.data_ ---> [ Heap Address 0x5000 ]: [ 100 MB of data ]
```

---

## 3. The Strong Exception Guarantee: Safe Buffer Swapping

Consider replacing a tensor's data with `tensor.set_data(new_values)`. What happens if `new_values` has the wrong size, or if memory allocation fails halfway through?

MiniTensor implements the **Copy-and-Swap** idiom to provide the **Strong Exception Guarantee**:
> *If an operation fails, the program state is left exactly as it was before the operation was attempted. No data is lost or corrupted.*

Let us inspect the implementation in `include/minitensor/tensor.hpp`:

```cpp
void set_data(const std::vector<T>& values) {
    if (values.size() != size()) {
        throw std::invalid_argument(
            "Replacement data size must match the tensor size");
    }
    std::vector<T> replacement(values); // 1. Attempt to copy new values
    data_.swap(replacement);            // 2. Safely swap pointers (noexcept)
}
```

- If `values.size()` does not match `size()`, an exception is thrown **before** any changes are made.
- If creating `replacement` runs out of memory (`std::bad_alloc`), the exception occurs before `data_` is modified.
- `data_.swap(replacement)` only executes once everything has succeeded. It swaps raw pointers inside `std::vector`, which is guaranteed never to throw an exception (`noexcept`).
- The old buffer is cleaned up automatically when `replacement` goes out of scope.

Your original data is 100% safe against corruption.

---

## 4. Hardware Performance: Why Contiguous Memory Matters

To understand why MiniTensor stores all numbers in a single contiguous `std::vector<T>` rather than nested arrays, we must look at how modern CPU hardware reads memory.

### The Memory Hierarchy: Registers, Caches, and RAM

```text
+------------------------------------+  Fastest (~0.5 ns)
|          CPU Registers             |  Smallest (a few hundred bytes)
+------------------------------------+
|          L1 Data Cache             |  Fast (~1 ns)
+------------------------------------+
|          L2 Cache                  |  Medium (~3-5 ns)
+------------------------------------+
|          L3 Cache (Shared)         |  Moderate (~10-20 ns)
+------------------------------------+
|          Main Memory (RAM)         |  Slowest (~60-100 ns)
+------------------------------------+  Largest (Gigabytes)
```

CPUs are blindingly fast: they can perform billions of arithmetic calculations per second. However, retrieving a number from Main Memory (RAM) takes around 60 to 100 nanoseconds. 

If the CPU had to wait for RAM on every calculation, it would spend **99% of its time sitting idle**!

### Spatial Locality and Cache Line Prefetching

To solve this, CPUs do not fetch individual numbers from RAM. Whenever a memory address is accessed, the CPU's memory controller fetches a **Cache Line**—a contiguous chunk of **64 bytes** containing the requested number plus all its immediate neighbors!

- **Spatial Locality:** If a program accesses memory address $X$, it is overwhelmingly likely to access address $X+1, X+2, \dots$ immediately afterwards.
- **Hardware Prefetcher:** Modern CPUs detect sequential memory access patterns and proactively pull the next cache lines from RAM into L1 cache *before* your code even asks for them!

Because MiniTensor stores all elements in a **single contiguous buffer in row-major order**, looping over a tensor means the CPU prefetcher streams data into cache at maximum bus bandwidth.

### The "Pointer-Chasing" Tax of Nested Vectors

Now contrast this with nested vectors: `std::vector<std::vector<float>>`.

```text
Nested Vectors Memory Layout:
  outer_vector:
  [ Ptr to Row 0 ] -------------------> Heap Address 0x1080: [ float, float, float ]
  [ Ptr to Row 1 ] --------+
  [ Ptr to Row 2 ] ----+   +----------> Heap Address 0x9420: [ float, float, float ]
                       |
                       +--------------> Heap Address 0x3100: [ float, float, float ]
```

When looping over rows:
1. The CPU must read the pointer for Row 0.
2. It suffers a **Cache Miss** loading Row 0 from address `0x1080`.
3. It must read the pointer for Row 1.
4. Address `0x9420` is far away in RAM: another **Cache Miss**! The hardware prefetcher cannot predict where the next row is located.
5. In addition, each row vector incurs heap allocation overhead (24 bytes for vector headers + heap allocator bookkeeping).

In benchmarks, contiguous flat buffers frequently outperform nested vector structures by **$5\times$ to $20\times$** on matrix operations simply due to cache efficiency!

---

## 5. Summary Comparison: Memory Models

| Feature | `std::vector<std::vector<T>>` | Raw Pointer (`T*` / `malloc`) | `minitensor::Tensor<T>` |
| :--- | :---: | :---: | :---: |
| **Memory Contiguity** | ❌ Fragmented across heap | ✔ Contiguous | ✔ Contiguous |
| **Cache Locality** | ❌ Terrible (Pointer chasing) | ✔ Excellent | ✔ Excellent |
| **Leak-Free Safety** | ✔ Yes (RAII) | ❌ High risk (Manual free) | ✔ Yes (RAII) |
| **Out-of-Bounds Protection** | ❌ Manual checking | ❌ None (Undefined Behavior) | ✔ Yes (`std::out_of_range`) |
| **Copy Independence** | ✔ Deep copies | ❌ Shallow pointer copies | ✔ Deep copies |
| **Move Semantics ($O(1)$ transfer)** | ✔ Yes | ❌ Manual pointer swaps | ✔ Yes (`noexcept` moves) |
| **Multidimensional Abstraction** | ❌ Hardcoded nesting | ❌ Manual arithmetic | ✔ Dynamic arbitrary rank |

MiniTensor gives you the **safety of modern C++ containers** combined with the **raw performance of contiguous C-style arrays**.

---

Proceed to:
- [Mathematical Operations](mathematical_operations.md) to learn how operations leverage contiguous storage.
- [API Reference](api_reference.md) for complete method signatures.
- [Usage Tutorial](usage_tutorial.md) for step-by-step code examples.

