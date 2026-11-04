#include "minitensor/minitensor.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

template <typename T>
void print_tensor_info(const minitensor::Tensor<T>& tensor) {
    std::cout << "Shape: {";
    for (std::size_t index = 0; index < tensor.get_shape().size(); ++index) {
        if (index != 0) {
            std::cout << ", ";
        }
        std::cout << tensor.get_shape()[index];
    }
    std::cout << "}, rank: " << tensor.rank() << ", size: " << tensor.size()
              << '\n';
}

int main() {
    minitensor::Tensor<float> matrix(
        {2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    std::cout << "2D float tensor:\n";
    print_tensor_info(matrix);
    std::cout << "Initial data: ";
    for (const float value : matrix.get_data()) {
        std::cout << value << ' ';
    }
    std::cout << '\n';

    matrix.fill(5.0F);
    std::cout << "After fill(5): ";
    for (const float value : matrix.get_data()) {
        std::cout << value << ' ';
    }
    std::cout << "\n\n";

    const minitensor::Tensor<double> volume({2, 3, 4});
    std::cout << "3D double tensor:\n";
    print_tensor_info(volume);

    const minitensor::Tensor<int> scalar(std::vector<std::size_t>{});
    std::cout << "Scalar tensor:\n";
    print_tensor_info(scalar);

    const minitensor::Tensor<float> empty({2, 0, 3});
    std::cout << "Zero-sized tensor:\n";
    print_tensor_info(empty);
}
