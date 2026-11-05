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
        const TensorMatmulPlan plan = validate_matmul_plan(other);
        Tensor result(plan.result_shape);
        if (plan.inner == 0 || result.size() == 0) {
            return result;
        }

        const std::size_t L = plan.result_batch_shape.size();
        const std::size_t slice_size =
            (plan.has_a_row ? plan.rows : 1) * (plan.has_b_col ? plan.columns : 1);

        for (std::size_t b = 0; b < plan.total_batches; ++b) {
            std::size_t a_batch_off = 0;
            std::size_t b_batch_off = 0;
            std::size_t rem = b;
            for (std::size_t d = L; d > 0; --d) {
                const std::size_t coord = rem % plan.result_batch_shape[d - 1];
                rem /= plan.result_batch_shape[d - 1];
                a_batch_off += coord * plan.a_batch_strides[d - 1];
                b_batch_off += coord * plan.b_batch_strides[d - 1];
            }
            const std::size_t res_batch_off = b * slice_size;

            for (std::size_t row = 0; row < plan.rows; ++row) {
                for (std::size_t inner = 0; inner < plan.inner; ++inner) {
                    const T left =
                        data_[a_batch_off + row * plan.a_row_stride +
                              inner * plan.a_inner_stride];
                    for (std::size_t column = 0; column < plan.columns;
                         ++column) {
                        result.data_[res_batch_off + row * plan.columns +
                                     column] +=
                            left *
                            other.data_[b_batch_off +
                                        inner * plan.b_inner_stride +
                                        column * plan.b_col_stride];
                    }
                }
            }
        }
        return result;
    }

    [[nodiscard]] Tensor matmul_blocked(
        const Tensor& other, std::size_t block_size = 64) const {
        const TensorMatmulPlan plan = validate_matmul_plan(other);
        if (block_size == 0) {
            throw std::invalid_argument(
                "Matrix multiplication block size must be positive");
        }

        Tensor result(plan.result_shape);
        if (plan.inner == 0 || result.size() == 0) {
            return result;
        }

        const std::size_t L = plan.result_batch_shape.size();
        const std::size_t slice_size =
            (plan.has_a_row ? plan.rows : 1) * (plan.has_b_col ? plan.columns : 1);

        for (std::size_t b = 0; b < plan.total_batches; ++b) {
            std::size_t a_batch_off = 0;
            std::size_t b_batch_off = 0;
            std::size_t rem = b;
            for (std::size_t d = L; d > 0; --d) {
                const std::size_t coord = rem % plan.result_batch_shape[d - 1];
                rem /= plan.result_batch_shape[d - 1];
                a_batch_off += coord * plan.a_batch_strides[d - 1];
                b_batch_off += coord * plan.b_batch_strides[d - 1];
            }
            const std::size_t res_batch_off = b * slice_size;

            for (std::size_t row_block = 0; row_block < plan.rows;) {
                const std::size_t row_end =
                    row_block +
                    std::min(block_size, plan.rows - row_block);
                for (std::size_t inner_block = 0;
                     inner_block < plan.inner;) {
                    const std::size_t inner_end =
                        inner_block +
                        std::min(block_size, plan.inner - inner_block);
                    for (std::size_t row = row_block; row < row_end; ++row) {
                        for (std::size_t inner = inner_block; inner < inner_end;
                             ++inner) {
                            const T left =
                                data_[a_batch_off + row * plan.a_row_stride +
                                      inner * plan.a_inner_stride];
                            for (std::size_t column = 0; column < plan.columns;
                                 ++column) {
                                result.data_[res_batch_off +
                                             row * plan.columns + column] +=
                                    left *
                                    other.data_[b_batch_off +
                                                inner * plan.b_inner_stride +
                                                column * plan.b_col_stride];
                            }
                        }
                    }
                    inner_block = inner_end;
                }
                row_block = row_end;
            }
        }
        return result;
    }

    [[nodiscard]] Tensor matmul_parallel(
        const Tensor& other, std::size_t block_size = 64) const {
        const TensorMatmulPlan plan = validate_matmul_plan(other);
        if (block_size == 0) {
            throw std::invalid_argument(
                "Matrix multiplication block size must be positive");
        }
#if defined(MINITENSOR_USE_OPENMP)
        Tensor result(plan.result_shape);
        if (plan.inner == 0 || result.size() == 0) {
            return result;
        }

        const std::size_t L = plan.result_batch_shape.size();
        const std::size_t slice_size =
            (plan.has_a_row ? plan.rows : 1) * (plan.has_b_col ? plan.columns : 1);

        if (plan.total_batches > 1) {
#pragma omp parallel for schedule(static)
            for (std::ptrdiff_t b = 0;
                 b < static_cast<std::ptrdiff_t>(plan.total_batches); ++b) {
                const std::size_t batch_idx = static_cast<std::size_t>(b);
                std::size_t a_batch_off = 0;
                std::size_t b_batch_off = 0;
                std::size_t rem = batch_idx;
                for (std::size_t d = L; d > 0; --d) {
                    const std::size_t coord =
                        rem % plan.result_batch_shape[d - 1];
                    rem /= plan.result_batch_shape[d - 1];
                    a_batch_off += coord * plan.a_batch_strides[d - 1];
                    b_batch_off += coord * plan.b_batch_strides[d - 1];
                }
                const std::size_t res_batch_off = batch_idx * slice_size;

                for (std::size_t row_block = 0; row_block < plan.rows;) {
                    const std::size_t row_end =
                        row_block +
                        std::min(block_size, plan.rows - row_block);
                    for (std::size_t inner_block = 0;
                         inner_block < plan.inner;) {
                        const std::size_t inner_end =
                            inner_block +
                            std::min(block_size, plan.inner - inner_block);
                        for (std::size_t row = row_block; row < row_end;
                             ++row) {
                            for (std::size_t inner = inner_block;
                                 inner < inner_end; ++inner) {
                                const T left =
                                    data_[a_batch_off +
                                          row * plan.a_row_stride +
                                          inner * plan.a_inner_stride];
                                for (std::size_t column = 0;
                                     column < plan.columns; ++column) {
                                    result.data_[res_batch_off +
                                                 row * plan.columns + column] +=
                                        left *
                                        other.data_[b_batch_off +
                                                    inner *
                                                        plan.b_inner_stride +
                                                    column *
                                                        plan.b_col_stride];
                                }
                            }
                        }
                        inner_block = inner_end;
                    }
                    row_block = row_end;
                }
            }
            return result;
        }

        // Single batch: parallelize across rows
        const bool large_workload =
            plan.rows >= 64 && plan.columns >= 64 && plan.inner >= 64;
#pragma omp parallel for if(large_workload) schedule(static)
        for (std::ptrdiff_t row = 0;
             row < static_cast<std::ptrdiff_t>(plan.rows); ++row) {
            const std::size_t row_index = static_cast<std::size_t>(row);
            for (std::size_t inner_block = 0;
                 inner_block < plan.inner;) {
                const std::size_t inner_end =
                    inner_block +
                    std::min(block_size, plan.inner - inner_block);
                for (std::size_t inner = inner_block; inner < inner_end;
                     ++inner) {
                    const T left =
                        data_[row_index * plan.a_row_stride +
                              inner * plan.a_inner_stride];
                    for (std::size_t column = 0; column < plan.columns;
                         ++column) {
                        result.data_[row_index * plan.columns + column] +=
                            left *
                            other.data_[inner * plan.b_inner_stride +
                                        column * plan.b_col_stride];
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

    [[nodiscard]] Tensor tensordot(const Tensor& other,
                                   std::size_t axes = 2) const {
        if (axes == 0) {
            std::vector<std::size_t> res_shape = shape_;
            res_shape.insert(res_shape.end(), other.shape_.begin(),
                             other.shape_.end());
            Tensor result(res_shape);
            for (std::size_t i = 0; i < size(); ++i) {
                for (std::size_t j = 0; j < other.size(); ++j) {
                    result.data_[i * other.size() + j] =
                        data_[i] * other.data_[j];
                }
            }
            return result;
        }
        if (axes > rank() || axes > other.rank()) {
            throw std::invalid_argument(
                "tensordot axes exceed tensor rank");
        }
        std::vector<std::size_t> axes_a(axes);
        std::vector<std::size_t> axes_b(axes);
        for (std::size_t i = 0; i < axes; ++i) {
            axes_a[i] = rank() - axes + i;
            axes_b[i] = i;
        }
        return tensordot(other, axes_a, axes_b);
    }

    [[nodiscard]] Tensor tensordot(
        const Tensor& other,
        const std::vector<std::size_t>& axes_a,
        const std::vector<std::size_t>& axes_b) const {
        if (axes_a.size() != axes_b.size()) {
            throw std::invalid_argument(
                "tensordot axis list lengths must match");
        }
        for (std::size_t i = 0; i < axes_a.size(); ++i) {
            if (axes_a[i] >= rank() || axes_b[i] >= other.rank()) {
                throw std::out_of_range("tensordot axis index out of bounds");
            }
            if (shape_[axes_a[i]] != other.shape_[axes_b[i]]) {
                throw std::invalid_argument(
                    "tensordot contracted dimensions must match");
            }
        }
        std::vector<std::size_t> free_a;
        std::vector<bool> is_contracted_a(rank(), false);
        for (const std::size_t ax : axes_a) is_contracted_a[ax] = true;
        for (std::size_t i = 0; i < rank(); ++i) {
            if (!is_contracted_a[i]) free_a.push_back(i);
        }

        std::vector<std::size_t> free_b;
        std::vector<bool> is_contracted_b(other.rank(), false);
        for (const std::size_t ax : axes_b) is_contracted_b[ax] = true;
        for (std::size_t i = 0; i < other.rank(); ++i) {
            if (!is_contracted_b[i]) free_b.push_back(i);
        }

        std::vector<std::size_t> res_shape;
        for (const std::size_t ax : free_a) res_shape.push_back(shape_[ax]);
        for (const std::size_t ax : free_b) res_shape.push_back(other.shape_[ax]);

        Tensor result(res_shape);
        if (result.size() == 0 || size() == 0 || other.size() == 0) {
            return result;
        }

        for (std::size_t a_idx = 0; a_idx < size(); ++a_idx) {
            std::vector<std::size_t> a_coord(rank());
            std::size_t rem_a = a_idx;
            for (std::size_t d = 0; d < rank(); ++d) {
                a_coord[d] = rem_a / strides_[d];
                rem_a %= strides_[d];
            }
            for (std::size_t b_idx = 0; b_idx < other.size(); ++b_idx) {
                std::vector<std::size_t> b_coord(other.rank());
                std::size_t rem_b = b_idx;
                for (std::size_t d = 0; d < other.rank(); ++d) {
                    b_coord[d] = rem_b / other.strides_[d];
                    rem_b %= other.strides_[d];
                }
                bool match = true;
                for (std::size_t i = 0; i < axes_a.size(); ++i) {
                    if (a_coord[axes_a[i]] != b_coord[axes_b[i]]) {
                        match = false;
                        break;
                    }
                }
                if (!match) continue;

                std::vector<std::size_t> res_coord;
                for (const std::size_t ax : free_a) res_coord.push_back(a_coord[ax]);
                for (const std::size_t ax : free_b) res_coord.push_back(b_coord[ax]);

                result.at(res_coord) += data_[a_idx] * other.data_[b_idx];
            }
        }
        return result;
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
    struct TensorMatmulPlan {
        bool is_dot_product = false;
        bool has_a_row = true;
        bool has_b_col = true;
        std::size_t rows = 1;
        std::size_t inner = 1;
        std::size_t columns = 1;
        std::size_t total_batches = 1;
        std::size_t a_row_stride = 0;
        std::size_t a_inner_stride = 0;
        std::size_t b_inner_stride = 0;
        std::size_t b_col_stride = 0;
        std::vector<std::size_t> result_batch_shape;
        std::vector<std::size_t> result_shape;
        std::vector<std::size_t> a_batch_strides;
        std::vector<std::size_t> b_batch_strides;
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

    [[nodiscard]] TensorMatmulPlan validate_matmul_plan(
        const Tensor& other) const {
        if (rank() == 0 || other.rank() == 0) {
            throw std::invalid_argument(
                "Matrix multiplication requires operands of rank at least 1");
        }

        TensorMatmulPlan plan;

        if (rank() == 1 && other.rank() == 1) {
            if (shape_[0] != other.shape_[0]) {
                throw std::invalid_argument(
                    "Matrix multiplication inner dimensions must match");
            }
            plan.is_dot_product = true;
            plan.has_a_row = false;
            plan.has_b_col = false;
            plan.rows = 1;
            plan.inner = shape_[0];
            plan.columns = 1;
            plan.total_batches = 1;
            plan.a_row_stride = 0;
            plan.a_inner_stride = strides_[0];
            plan.b_inner_stride = other.strides_[0];
            plan.b_col_stride = 0;
            plan.result_shape = {}; // Scalar rank 0
            return plan;
        }

        std::vector<std::size_t> a_batch;
        if (rank() == 1) {
            plan.has_a_row = false;
            plan.rows = 1;
            plan.inner = shape_[0];
            plan.a_row_stride = 0;
            plan.a_inner_stride = strides_[0];
        } else {
            plan.has_a_row = true;
            plan.rows = shape_[rank() - 2];
            plan.inner = shape_[rank() - 1];
            plan.a_row_stride = strides_[rank() - 2];
            plan.a_inner_stride = strides_[rank() - 1];
            a_batch.assign(shape_.begin(), shape_.end() - 2);
        }

        std::vector<std::size_t> b_batch;
        std::size_t b_inner = 0;
        if (other.rank() == 1) {
            plan.has_b_col = false;
            b_inner = other.shape_[0];
            plan.columns = 1;
            plan.b_inner_stride = other.strides_[0];
            plan.b_col_stride = 0;
        } else {
            plan.has_b_col = true;
            b_inner = other.shape_[other.rank() - 2];
            plan.columns = other.shape_[other.rank() - 1];
            plan.b_inner_stride = other.strides_[other.rank() - 2];
            plan.b_col_stride = other.strides_[other.rank() - 1];
            b_batch.assign(other.shape_.begin(), other.shape_.end() - 2);
        }

        if (plan.inner != b_inner) {
            throw std::invalid_argument(
                "Matrix multiplication inner dimensions must match");
        }

        const std::size_t p = a_batch.size();
        const std::size_t q = b_batch.size();
        const std::size_t L = std::max(p, q);

        std::vector<std::size_t> a_padded(L, 1);
        std::vector<std::size_t> b_padded(L, 1);
        plan.result_batch_shape.assign(L, 1);

        for (std::size_t i = 0; i < p; ++i) {
            a_padded[L - p + i] = a_batch[i];
        }
        for (std::size_t i = 0; i < q; ++i) {
            b_padded[L - q + i] = b_batch[i];
        }

        for (std::size_t d = 0; d < L; ++d) {
            const std::size_t dim_a = a_padded[d];
            const std::size_t dim_b = b_padded[d];
            if (dim_a == dim_b) {
                plan.result_batch_shape[d] = dim_a;
            } else if (dim_a == 1) {
                plan.result_batch_shape[d] = dim_b;
            } else if (dim_b == 1) {
                plan.result_batch_shape[d] = dim_a;
            } else {
                throw std::invalid_argument(
                    "Matrix multiplication batch dimensions are incompatible");
            }
        }

        plan.total_batches = 1;
        for (const std::size_t dim : plan.result_batch_shape) {
            plan.total_batches *= dim;
        }

        plan.a_batch_strides.assign(L, 0);
        plan.b_batch_strides.assign(L, 0);

        for (std::size_t i = 0; i < p; ++i) {
            if (a_batch[i] > 1) {
                plan.a_batch_strides[L - p + i] = strides_[i];
            } else {
                plan.a_batch_strides[L - p + i] = 0;
            }
        }
        for (std::size_t i = 0; i < q; ++i) {
            if (b_batch[i] > 1) {
                plan.b_batch_strides[L - q + i] = other.strides_[i];
            } else {
                plan.b_batch_strides[L - q + i] = 0;
            }
        }

        plan.result_shape = plan.result_batch_shape;
        if (plan.has_a_row) {
            plan.result_shape.push_back(plan.rows);
        }
        if (plan.has_b_col) {
            plan.result_shape.push_back(plan.columns);
        }

        return plan;
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
