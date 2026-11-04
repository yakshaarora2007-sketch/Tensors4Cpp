#include <minitensor/minitensor.hpp>

int main() {
    minitensor::Tensor<int> tensor({2, 2}, 1);
    tensor.set({1, 0}, 7);
    return tensor(1, 0) == 7 && tensor.get_strides()[0] == 2 ? 0 : 1;
}
