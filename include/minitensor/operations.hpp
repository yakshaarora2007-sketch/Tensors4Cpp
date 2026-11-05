#ifndef MINITENSOR_OPERATIONS_HPP
#define MINITENSOR_OPERATIONS_HPP

#include "minitensor/tensor.hpp"

namespace minitensor {

template <typename T>
[[nodiscard]] inline Tensor<T> transpose(const Tensor<T>& a) {
    return a.transpose();
}

template <typename T>
[[nodiscard]] inline Tensor<T> reshape(const Tensor<T>& a,
                                       const std::vector<std::size_t>& new_shape) {
    return a.reshape(new_shape);
}

}  // namespace minitensor

#endif  // MINITENSOR_OPERATIONS_HPP
