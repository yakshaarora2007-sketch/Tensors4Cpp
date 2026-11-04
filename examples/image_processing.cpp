#include "minitensor/minitensor.hpp"

#include <cstddef>
#include <iostream>

int main() {
    constexpr std::size_t height = 2;
    constexpr std::size_t width = 2;
    minitensor::Tensor<float> image({height, width, 3});

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            image.set({y, x, 0}, static_cast<float>(y + 1) * 0.1F);
            image.set({y, x, 1}, static_cast<float>(x + 1) * 0.2F);
            image.set({y, x, 2}, 0.3F);
        }
    }

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                image.at({y, x, channel}) += 0.1F;
            }
        }
    }

    const minitensor::Tensor<float> half_intensity({height, width, 3}, 0.5F);
    const auto dimmed_image = image * half_intensity;

    minitensor::Tensor<float> swapped_channels({height, width, 3});
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            swapped_channels.set({y, x, 0}, image.at({y, x, 2}));
            swapped_channels.set({y, x, 1}, image.at({y, x, 1}));
            swapped_channels.set({y, x, 2}, image.at({y, x, 0}));
        }
    }

    std::cout << "Synthetic RGB image shape: {2, 2, 3}\n";
    std::cout << "Brightened, half-intensity pixels (RGB):\n";
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            std::cout << '(' << dimmed_image.at({y, x, 0}) << ", "
                      << dimmed_image.at({y, x, 1}) << ", "
                      << dimmed_image.at({y, x, 2}) << ") ";
        }
        std::cout << '\n';
    }

    std::cout << "First pixel after red/blue channel swap: ("
              << swapped_channels.at({0, 0, 0}) << ", "
              << swapped_channels.at({0, 0, 1}) << ", "
              << swapped_channels.at({0, 0, 2}) << ")\n";
}
