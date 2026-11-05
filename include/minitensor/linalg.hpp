#ifndef MINITENSOR_LINALG_HPP
#define MINITENSOR_LINALG_HPP

#include "minitensor/tensor.hpp"

namespace minitensor {

template <typename T>
[[nodiscard]] inline Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    return a.matmul(b);
}

template <typename T>
[[nodiscard]] inline Tensor<T> matmul_blocked(const Tensor<T>& a,
                                             const Tensor<T>& b,
                                             std::size_t block_size = 64) {
    return a.matmul_blocked(b, block_size);
}

template <typename T>
[[nodiscard]] inline Tensor<T> matmul_parallel(const Tensor<T>& a,
                                              const Tensor<T>& b,
                                              std::size_t block_size = 64) {
    return a.matmul_parallel(b, block_size);
}

template <typename T>
[[nodiscard]] inline Tensor<T> tensordot(const Tensor<T>& a,
                                        const Tensor<T>& b,
                                        std::size_t axes = 2) {
    return a.tensordot(b, axes);
}

template <typename T>
[[nodiscard]] inline Tensor<T> tensordot(
    const Tensor<T>& a, const Tensor<T>& b,
    const std::vector<std::size_t>& axes_a,
    const std::vector<std::size_t>& axes_b) {
    return a.tensordot(b, axes_a, axes_b);
}

}  // namespace minitensor

#endif  // MINITENSOR_LINALG_HPP
