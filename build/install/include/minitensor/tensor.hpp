#ifndef MINITENSOR_TENSOR_HPP
#define MINITENSOR_TENSOR_HPP

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace minitensor {

template <typename T>
class Tensor {
public:
    // An empty shape represents a scalar, which contains one value.
    Tensor() : shape_{}, strides_{}, data_(1, T{}) {}

    explicit Tensor(std::vector<std::size_t> shape)
        : shape_(std::move(shape)),
          strides_(calculate_strides(shape_)),
          data_(calculate_size(shape_), T{}) {}

    Tensor(std::vector<std::size_t> shape, const T& initial_value)
        : shape_(std::move(shape)),
          strides_(calculate_strides(shape_)),
          data_(calculate_size(shape_), initial_value) {}

    Tensor(std::vector<std::size_t> shape, std::vector<T> initial_data)
        : shape_(std::move(shape)), strides_(calculate_strides(shape_)) {
        const std::size_t expected_size = calculate_size(shape_);
        if (initial_data.size() != expected_size) {
            throw std::invalid_argument(
                "Initial data size must match the tensor shape");
        }
        data_ = std::move(initial_data);
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return data_.size();
    }

    [[nodiscard]] std::size_t rank() const noexcept {
        return shape_.size();
    }

    [[nodiscard]] const std::vector<std::size_t>& get_shape() const noexcept {
        return shape_;
    }

    [[nodiscard]] const std::vector<std::size_t>& get_strides() const noexcept {
        return strides_;
    }

    [[nodiscard]] const std::vector<T>& get_data() const noexcept {
        return data_;
    }

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    void set_data(const std::vector<T>& values) {
        if (values.size() != size()) {
            throw std::invalid_argument(
                "Replacement data size must match the tensor size");
        }
        std::vector<T> replacement(values);
        data_.swap(replacement);
    }

    [[nodiscard]] T& at(const std::vector<std::size_t>& indices) {
        return data_.at(calculate_offset(indices));
    }

    [[nodiscard]] const T& at(
        const std::vector<std::size_t>& indices) const {
        return data_.at(calculate_offset(indices));
    }

    [[nodiscard]] T& at(std::initializer_list<std::size_t> indices) {
        return at(std::vector<std::size_t>(indices));
    }

    [[nodiscard]] const T& at(
        std::initializer_list<std::size_t> indices) const {
        return at(std::vector<std::size_t>(indices));
    }

    void set(const std::vector<std::size_t>& indices, const T& value) {
        at(indices) = value;
    }

    void set(std::initializer_list<std::size_t> indices, const T& value) {
        set(std::vector<std::size_t>(indices), value);
    }

    [[nodiscard]] T& operator()() {
        return at(std::vector<std::size_t>{});
    }

    [[nodiscard]] const T& operator()() const {
        return at(std::vector<std::size_t>{});
    }

    template <typename... Indices>
        requires(sizeof...(Indices) > 0 &&
                 (std::is_convertible_v<Indices, std::size_t> && ...))
    [[nodiscard]] T& operator()(Indices... indices) {
        return at(std::vector<std::size_t>{
            static_cast<std::size_t>(indices)...});
    }

    template <typename... Indices>
        requires(sizeof...(Indices) > 0 &&
                 (std::is_convertible_v<Indices, std::size_t> && ...))
    [[nodiscard]] const T& operator()(Indices... indices) const {
        return at(std::vector<std::size_t>{
            static_cast<std::size_t>(indices)...});
    }

    [[nodiscard]] Tensor operator+(const Tensor& other) const {
        return elementwise(other, [](const T& left, const T& right) {
            return left + right;
        });
    }

    [[nodiscard]] Tensor operator-(const Tensor& other) const {
        return elementwise(other, [](const T& left, const T& right) {
            return left - right;
        });
    }

    // Element-wise multiplication; use matmul() for matrix multiplication.
    [[nodiscard]] Tensor operator*(const Tensor& other) const {
        return elementwise(other, [](const T& left, const T& right) {
            return left * right;
        });
    }

    [[nodiscard]] Tensor matmul(const Tensor& other) const {
        const MatrixDimensions dimensions = validate_matmul(other);
        Tensor result({dimensions.rows, dimensions.columns});
        if (dimensions.inner == 0 || result.size() == 0) {
            return result;
        }
        for (std::size_t row = 0; row < dimensions.rows; ++row) {
            for (std::size_t inner = 0; inner < dimensions.inner; ++inner) {
                const T left = data_[row * dimensions.inner + inner];
                for (std::size_t column = 0;
                     column < dimensions.columns; ++column) {
                    result.data_[row * dimensions.columns + column] +=
                        left * other.data_[inner * dimensions.columns + column];
                }
            }
        }
        return result;
    }

    [[nodiscard]] Tensor matmul_blocked(
        const Tensor& other, std::size_t block_size = 64) const {
        const MatrixDimensions dimensions = validate_matmul(other);
        if (block_size == 0) {
            throw std::invalid_argument("Matrix multiplication block size must be positive");
        }

        Tensor result({dimensions.rows, dimensions.columns});
        if (dimensions.inner == 0 || result.size() == 0) {
            return result;
        }
        for (std::size_t row_block = 0; row_block < dimensions.rows;) {
            const std::size_t row_end =
                row_block + std::min(block_size, dimensions.rows - row_block);
            for (std::size_t inner_block = 0;
                 inner_block < dimensions.inner;) {
                const std::size_t inner_end =
                    inner_block +
                    std::min(block_size, dimensions.inner - inner_block);
                for (std::size_t row = row_block; row < row_end; ++row) {
                    for (std::size_t inner = inner_block;
                         inner < inner_end; ++inner) {
                        const T left = data_[row * dimensions.inner + inner];
                        for (std::size_t column = 0;
                             column < dimensions.columns; ++column) {
                            result.data_[row * dimensions.columns + column] +=
                                left * other.data_[inner *
                                                       dimensions.columns +
                                                   column];
                        }
                    }
                }
                inner_block = inner_end;
            }
            row_block = row_end;
        }
        return result;
    }

    [[nodiscard]] Tensor matmul_parallel(
        const Tensor& other, std::size_t block_size = 64) const {
        const MatrixDimensions dimensions = validate_matmul(other);
        if (block_size == 0) {
            throw std::invalid_argument("Matrix multiplication block size must be positive");
        }
#if defined(MINITENSOR_USE_OPENMP)
        Tensor result({dimensions.rows, dimensions.columns});
        if (dimensions.inner == 0 || result.size() == 0) {
            return result;
        }
        if (dimensions.rows >
            static_cast<std::size_t>(
                std::numeric_limits<std::ptrdiff_t>::max())) {
            return matmul(other);
        }
        const bool large_workload =
            dimensions.rows >= 64 && dimensions.columns >= 64 &&
            dimensions.inner >= 64;
#pragma omp parallel for if(large_workload) schedule(static)
        for (std::ptrdiff_t row = 0;
             row < static_cast<std::ptrdiff_t>(dimensions.rows); ++row) {
            const std::size_t row_index = static_cast<std::size_t>(row);
            for (std::size_t inner_block = 0;
                 inner_block < dimensions.inner;) {
                const std::size_t inner_end =
                    inner_block +
                    std::min(block_size, dimensions.inner - inner_block);
                for (std::size_t inner = inner_block; inner < inner_end;
                     ++inner) {
                    const T left =
                        data_[row_index * dimensions.inner + inner];
                    for (std::size_t column = 0;
                         column < dimensions.columns; ++column) {
                        result.data_[row_index * dimensions.columns + column] +=
                            left * other.data_[inner * dimensions.columns +
                                               column];
                    }
                }
                inner_block = inner_end;
            }
        }
        return result;
#else
        return matmul(other);
#endif
    }

    [[nodiscard]] Tensor reshape(
        const std::vector<std::size_t>& new_shape) const {
        const std::size_t new_size = calculate_size(new_shape);
        if (new_size != size()) {
            throw std::invalid_argument(
                "Reshaped tensor must have the same number of elements");
        }
        return Tensor(new_shape, data_);
    }

    [[nodiscard]] Tensor transpose() const {
        if (rank() != 2) {
            throw std::invalid_argument(
                "Transpose is supported only for two-dimensional tensors");
        }

        Tensor result({shape_[1], shape_[0]});
        for (std::size_t row = 0; row < shape_[0]; ++row) {
            for (std::size_t column = 0; column < shape_[1]; ++column) {
                result.data_[column * shape_[0] + row] =
                    data_[row * shape_[1] + column];
            }
        }
        return result;
    }

private:
    struct MatrixDimensions {
        std::size_t rows;
        std::size_t inner;
        std::size_t columns;
    };

    static std::size_t calculate_size(
        const std::vector<std::size_t>& shape) {
        for (const std::size_t dimension : shape) {
            if (dimension == 0) {
                return 0;
            }
        }

        std::size_t total_size = 1;
        for (const std::size_t dimension : shape) {
            if (total_size >
                std::numeric_limits<std::size_t>::max() / dimension) {
                throw std::length_error("Tensor shape product overflows size_t");
            }
            total_size *= dimension;
        }
        return total_size;
    }

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

    [[nodiscard]] MatrixDimensions validate_matmul(
        const Tensor& other) const {
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

    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<T> data_;
};

}  // namespace minitensor

#endif  // MINITENSOR_TENSOR_HPP
