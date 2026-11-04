#include "minitensor/tensor.hpp"

#include <cstddef>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        std::forward<Function>(function)();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_creation_types_and_shapes() {
    const minitensor::Tensor<int> integers({4});
    const minitensor::Tensor<float> floats({2, 3});
    const minitensor::Tensor<double> doubles({2, 3, 4});

    require(integers.size() == 4 && integers.rank() == 1,
            "1D integer tensor has incorrect shape or size");
    require(floats.size() == 6 && floats.rank() == 2,
            "2D float tensor has incorrect shape or size");
    require(doubles.size() == 24 && doubles.rank() == 3,
            "3D double tensor has incorrect shape or size");
    require(floats.get_shape() == std::vector<std::size_t>({2, 3}),
            "Tensor shape was not retained");
}

void test_scalar_and_zero_sized_tensors() {
    const minitensor::Tensor<int> default_tensor;
    const minitensor::Tensor<double> scalar(std::vector<std::size_t>{});
    minitensor::Tensor<float> zero_sized({2, 0, 3});

    require(default_tensor.rank() == 0 && default_tensor.size() == 1,
            "Default tensor should be a scalar");
    require(scalar.rank() == 0 && scalar.size() == 1,
            "Empty shape should represent a scalar");
    require(scalar.get_data().size() == 1 && scalar.get_data()[0] == 0.0,
            "Scalar should be value-initialized");
    require(scalar.get_strides().empty() && scalar() == 0.0,
            "Scalar should have no strides and support zero-index access");
    require(zero_sized.rank() == 3 && zero_sized.size() == 0,
            "A zero dimension should produce an empty tensor");
    zero_sized.fill(4.0F);
    zero_sized.set_data({});
    require(zero_sized.size() == 0 && zero_sized.get_data().empty(),
            "Empty tensor fill and data replacement should remain valid");
}

void test_initialization_and_fill() {
    const minitensor::Tensor<int> initialized({3});
    require(initialized.get_data() == std::vector<int>({0, 0, 0}),
            "Default elements should be value-initialized");

    minitensor::Tensor<float> filled({2, 2}, 1.5F);
    require(filled.get_data() == std::vector<float>({1.5F, 1.5F, 1.5F, 1.5F}),
            "Fill-value constructor did not initialize every element");

    filled.fill(3.0F);
    require(filled.get_data() == std::vector<float>({3.0F, 3.0F, 3.0F, 3.0F}),
            "fill did not update every element");

    minitensor::Tensor<int> supplied({2, 2}, std::vector<int>{1, 2, 3, 4});
    require(supplied.get_data() == std::vector<int>({1, 2, 3, 4}),
            "Data constructor did not retain supplied values");
    supplied.set_data({4, 3, 2, 1});
    require(supplied.get_data() == std::vector<int>({4, 3, 2, 1}),
            "set_data did not replace tensor contents");
}

void test_invalid_data_sizes() {
    require_throws<std::invalid_argument>(
        [] { minitensor::Tensor<int> invalid({3}, std::vector<int>{1, 2}); },
        "Constructor should reject mismatched data size");

    minitensor::Tensor<int> tensor({2}, 7);
    require_throws<std::invalid_argument>(
        [&tensor] { tensor.set_data({1, 2, 3}); },
        "set_data should reject mismatched data size");
    require(tensor.get_data() == std::vector<int>({7, 7}),
            "Failed set_data must leave the tensor unchanged");
}

void test_size_overflow() {
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    require_throws<std::length_error>(
        [maximum] {
            minitensor::Tensor<int> invalid(
                std::vector<std::size_t>{maximum, 2});
        },
        "Overflowing shape product should throw length_error");

    const minitensor::Tensor<int> zero_with_large_dimensions(
        std::vector<std::size_t>{maximum, maximum, 0});
    require(zero_with_large_dimensions.size() == 0,
            "A zero dimension should result in zero elements");
}

void test_copy_and_independent_storage() {
    minitensor::Tensor<int> original({3}, std::vector<int>{1, 2, 3});
    minitensor::Tensor<int> copied(original);
    copied.fill(9);
    require(original.get_data() == std::vector<int>({1, 2, 3}),
            "Copy construction should create independent storage");

    minitensor::Tensor<int> assigned({1}, 0);
    assigned = original;
    assigned.fill(8);
    require(original.get_data() == std::vector<int>({1, 2, 3}),
            "Copy assignment should create independent storage");
}

void test_move_construction_and_assignment() {
    minitensor::Tensor<double> source({2}, std::vector<double>{2.5, 4.5});
    minitensor::Tensor<double> moved(std::move(source));
    require(moved.get_shape() == std::vector<std::size_t>({2}) &&
                moved.get_data() == std::vector<double>({2.5, 4.5}),
            "Move construction did not transfer tensor contents");

    minitensor::Tensor<double> move_assigned({1}, -1.0);
    move_assigned = std::move(moved);
    require(move_assigned.get_shape() == std::vector<std::size_t>({2}) &&
                move_assigned.get_data() == std::vector<double>({2.5, 4.5}),
            "Move assignment did not transfer tensor contents");
}

void test_strides_and_indexing() {
    const minitensor::Tensor<int> vector({4});
    const minitensor::Tensor<int> matrix({2, 3});
    require(vector.get_strides() == std::vector<std::size_t>({1}) &&
                matrix.get_strides() == std::vector<std::size_t>({3, 1}),
            "1D or 2D row-major strides are incorrect");

    minitensor::Tensor<int> tensor(
        {2, 3, 4}, std::vector<int>(24, 0));
    require(tensor.get_strides() == std::vector<std::size_t>({12, 4, 1}),
            "3D row-major strides are incorrect");
    tensor.set({1, 2, 3}, 42);
    require(tensor.at({1, 2, 3}) == 42 && tensor(1, 2, 3) == 42,
            "Multidimensional indexing or set failed");

    const auto& const_tensor = tensor;
    require(const_tensor.at({1, 2, 3}) == 42,
            "Const multidimensional access failed");
    require_throws<std::invalid_argument>(
        [&tensor] { static_cast<void>(tensor.at({1, 2})); },
        "Index rank mismatch should throw invalid_argument");
    require_throws<std::out_of_range>(
        [&tensor] { static_cast<void>(tensor.at({2, 0, 0})); },
        "Out-of-bounds index should throw out_of_range");
    require_throws<std::invalid_argument>(
        [] {
            const minitensor::Tensor<int> scalar;
            static_cast<void>(scalar.at({0}));
        },
        "Scalar index rank mismatch should throw invalid_argument");

    const minitensor::Tensor<int> empty({2, 0, 3});
    require(empty.get_strides() == std::vector<std::size_t>({0, 0, 0}),
            "Zero-sized tensor strides should be zero");
    require_throws<std::out_of_range>(
        [&empty] { static_cast<void>(empty.at({0, 0, 0})); },
        "Indexing a zero dimension should throw out_of_range");
}

void test_elementwise_operations() {
    const minitensor::Tensor<int> left({2, 2},
                                       std::vector<int>{1, 2, 3, 4});
    const minitensor::Tensor<int> right({2, 2},
                                        std::vector<int>{5, 6, 7, 8});
    require((left + right).get_data() == std::vector<int>({6, 8, 10, 12}),
            "Addition result is incorrect");
    require((right - left).get_data() == std::vector<int>({4, 4, 4, 4}),
            "Subtraction result is incorrect");
    require((left * right).get_data() == std::vector<int>({5, 12, 21, 32}),
            "Element-wise multiplication result is incorrect");

    require_throws<std::invalid_argument>(
        [&left] {
            const minitensor::Tensor<int> different_shape({4});
            static_cast<void>(left + different_shape);
        },
        "Element-wise operation should reject incompatible shapes");
}

void test_matrix_multiplication() {
    const minitensor::Tensor<int> left({2, 2},
                                       std::vector<int>{1, 2, 3, 4});
    const minitensor::Tensor<int> right({2, 2},
                                        std::vector<int>{5, 6, 7, 8});
    const std::vector<int> expected{19, 22, 43, 50};
    require(left.matmul(right).get_data() == expected,
            "Serial matrix multiplication result is incorrect");
    require(left.matmul_blocked(right, 1).get_data() == expected,
            "Blocked matrix multiplication result is incorrect");
    require(left.matmul_parallel(right, 2).get_data() == expected,
            "Parallel/fallback matrix multiplication result is incorrect");

    const minitensor::Tensor<double> non_square_left(
        {2, 3}, std::vector<double>{1, 2, 3, 4, 5, 6});
    const minitensor::Tensor<double> non_square_right(
        {3, 2}, std::vector<double>{7, 8, 9, 10, 11, 12});
    const auto product = non_square_left.matmul(non_square_right);
    require(product.get_shape() == std::vector<std::size_t>({2, 2}) &&
                product.get_data() ==
                    std::vector<double>({58, 64, 139, 154}),
            "Non-square matrix product is incorrect");
    require(non_square_left.matmul_blocked(non_square_right, 2)
                    .get_data() == product.get_data(),
            "Blocked matrix product differs from serial result");

    const minitensor::Tensor<double> empty_inner_left({2, 0});
    const minitensor::Tensor<double> empty_inner_right({0, 3});
    require(empty_inner_left.matmul(empty_inner_right).get_data() ==
                std::vector<double>(6, 0.0),
            "Zero-inner-dimension product should be zero-filled");

    require_throws<std::invalid_argument>(
        [&left] {
            const minitensor::Tensor<int> incompatible({3, 2});
            static_cast<void>(left.matmul(incompatible));
        },
        "Matrix multiplication should reject mismatched inner dimensions");
    require_throws<std::invalid_argument>(
        [] {
            const minitensor::Tensor<int> vector({2});
            static_cast<void>(vector.matmul(vector));
        },
        "Matrix multiplication should reject non-matrix tensors");
    require_throws<std::invalid_argument>(
        [&left] { static_cast<void>(left.matmul_blocked(left, 0)); },
        "Blocked multiplication should reject a zero block size");
}

void test_reshape_and_transpose() {
    const minitensor::Tensor<int> original(
        {2, 3}, std::vector<int>{1, 2, 3, 4, 5, 6});
    const auto reshaped = original.reshape({3, 2});
    require(reshaped.get_shape() == std::vector<std::size_t>({3, 2}) &&
                reshaped.get_data() == original.get_data(),
            "Reshape should preserve row-major element order");
    require(reshaped.get_strides() == std::vector<std::size_t>({2, 1}),
            "Reshape should recalculate strides");
    require_throws<std::invalid_argument>(
        [&original] { static_cast<void>(original.reshape({4, 2})); },
        "Reshape should reject a different element count");

    const auto transposed = original.transpose();
    require(transposed.get_shape() == std::vector<std::size_t>({3, 2}) &&
                transposed.get_data() ==
                    std::vector<int>({1, 4, 2, 5, 3, 6}),
            "Transpose result is incorrect");
    require_throws<std::invalid_argument>(
        [] {
            const minitensor::Tensor<int> vector({3});
            static_cast<void>(vector.transpose());
        },
        "Transpose should require a matrix");
    const minitensor::Tensor<int> scalar;
    require(scalar.reshape({1}).get_data() == std::vector<int>({0}),
            "Scalar reshape to one element should work");
    const minitensor::Tensor<int> empty({2, 0});
    require(empty.reshape({0, 4}).size() == 0,
            "Zero-sized reshape with equal element count should work");
}

void test_floating_point_operations() {
    const minitensor::Tensor<double> left(
        {2, 2}, std::vector<double>{0.1, 0.2, 0.3, 0.4});
    const auto product = left.matmul(left);
    const std::vector<double> expected{0.07, 0.10, 0.15, 0.22};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(std::abs(product.get_data()[index] - expected[index]) < 1e-12,
                "Floating-point matrix multiplication exceeds tolerance");
    }
}

}  // namespace

#ifdef MINITENSOR_USE_GTEST
#include <gtest/gtest.h>

TEST(MiniTensor, CreationTypesAndShapes) { test_creation_types_and_shapes(); }
TEST(MiniTensor, ScalarAndZeroSizedTensors) {
    test_scalar_and_zero_sized_tensors();
}
TEST(MiniTensor, InitializationAndFill) { test_initialization_and_fill(); }
TEST(MiniTensor, InvalidDataSizes) { test_invalid_data_sizes(); }
TEST(MiniTensor, SizeOverflow) { test_size_overflow(); }
TEST(MiniTensor, CopyAndIndependentStorage) {
    test_copy_and_independent_storage();
}
TEST(MiniTensor, MoveConstructionAndAssignment) {
    test_move_construction_and_assignment();
}
TEST(MiniTensor, StridesAndIndexing) { test_strides_and_indexing(); }
TEST(MiniTensor, ElementwiseOperations) { test_elementwise_operations(); }
TEST(MiniTensor, MatrixMultiplication) { test_matrix_multiplication(); }
TEST(MiniTensor, ReshapeAndTranspose) { test_reshape_and_transpose(); }
TEST(MiniTensor, FloatingPointOperations) {
    test_floating_point_operations();
}
#else
int main() {
    try {
        test_creation_types_and_shapes();
        test_scalar_and_zero_sized_tensors();
        test_initialization_and_fill();
        test_invalid_data_sizes();
        test_size_overflow();
        test_copy_and_independent_storage();
        test_move_construction_and_assignment();
        test_strides_and_indexing();
        test_elementwise_operations();
        test_matrix_multiplication();
        test_reshape_and_transpose();
        test_floating_point_operations();
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
    std::cout << "All MiniTensor tests passed.\n";
    return 0;
}
#endif
