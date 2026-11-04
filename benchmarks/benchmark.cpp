#include "minitensor/minitensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(MINITENSOR_USE_OPENMP)
#include <omp.h>
#endif

#ifndef MINITENSOR_BENCHMARK_BUILD_TYPE
#define MINITENSOR_BENCHMARK_BUILD_TYPE "unknown"
#endif
#ifndef MINITENSOR_BENCHMARK_CXX_FLAGS
#define MINITENSOR_BENCHMARK_CXX_FLAGS ""
#endif
#ifndef MINITENSOR_BENCHMARK_DEBUG_FLAGS
#define MINITENSOR_BENCHMARK_DEBUG_FLAGS ""
#endif
#ifndef MINITENSOR_BENCHMARK_RELEASE_FLAGS
#define MINITENSOR_BENCHMARK_RELEASE_FLAGS ""
#endif
#ifndef MINITENSOR_BENCHMARK_RELWITHDEBINFO_FLAGS
#define MINITENSOR_BENCHMARK_RELWITHDEBINFO_FLAGS ""
#endif
#ifndef MINITENSOR_BENCHMARK_MINSIZEREL_FLAGS
#define MINITENSOR_BENCHMARK_MINSIZEREL_FLAGS ""
#endif

#define MINITENSOR_STRINGIZE_DETAIL(value) #value
#define MINITENSOR_STRINGIZE(value) MINITENSOR_STRINGIZE_DETAIL(value)

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t default_block_size = 64;
constexpr int default_iterations = 3;

const char* compiler_name() {
#if defined(_MSC_VER)
    return "MSVC " MINITENSOR_STRINGIZE(_MSC_VER);
#elif defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#else
    return "Unknown compiler";
#endif
}

minitensor::Tensor<float> make_matrix(std::size_t size, float offset) {
    std::vector<float> values(size * size);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] =
            static_cast<float>((index * 17U + 13U) % 101U) / 101.0F + offset;
    }
    return minitensor::Tensor<float>({size, size}, std::move(values));
}

template <typename Operation>
double measure_average(int iterations, Operation operation,
                       double& checksum) {
    static_cast<void>(operation());
    std::chrono::duration<double> elapsed{};
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = Clock::now();
        const auto result = operation();
        const auto stop = Clock::now();
        elapsed += stop - start;
        if (!result.get_data().empty()) {
            checksum += result.get_data()[result.size() / 2];
        }
    }
    return elapsed.count() / static_cast<double>(iterations);
}

int parse_positive(const char* value, const char* option) {
    const int parsed = std::atoi(value);
    if (parsed <= 0) {
        throw std::invalid_argument(std::string(option) +
                                    " must be a positive integer");
    }
    return parsed;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int iterations = default_iterations;
        std::size_t block_size = default_block_size;
        std::size_t maximum_size = 1024;
        for (int argument = 1; argument < argc; ++argument) {
            const std::string option(argv[argument]);
            if (option == "--iterations" && argument + 1 < argc) {
                iterations = parse_positive(argv[++argument], "--iterations");
            } else if (option == "--block-size" && argument + 1 < argc) {
                block_size = static_cast<std::size_t>(
                    parse_positive(argv[++argument], "--block-size"));
            } else if (option == "--max-size" && argument + 1 < argc) {
                maximum_size = static_cast<std::size_t>(
                    parse_positive(argv[++argument], "--max-size"));
            } else {
                throw std::invalid_argument(
                    "Usage: minitensor_benchmark [--iterations N] "
                    "[--max-size N] [--block-size N]");
            }
        }

        int thread_count = 1;
#if defined(MINITENSOR_USE_OPENMP)
        thread_count = omp_get_max_threads();
#endif
        std::cout << "Compiler: " << compiler_name() << '\n'
                  << "Build configuration: " << MINITENSOR_BENCHMARK_BUILD_TYPE
                  << "\nCMAKE_CXX_FLAGS: " << MINITENSOR_BENCHMARK_CXX_FLAGS
                  << "\nDebug flags: " << MINITENSOR_BENCHMARK_DEBUG_FLAGS
                  << "\nRelease flags: " << MINITENSOR_BENCHMARK_RELEASE_FLAGS
                  << "\nRelWithDebInfo flags: "
                  << MINITENSOR_BENCHMARK_RELWITHDEBINFO_FLAGS
                  << "\nMinSizeRel flags: "
                  << MINITENSOR_BENCHMARK_MINSIZEREL_FLAGS << '\n'
                  << "OpenMP: "
#if defined(MINITENSOR_USE_OPENMP)
                  << "enabled\n"
#else
                  << "disabled (parallel API uses serial fallback)\n"
#endif
                  << "Threads: " << thread_count << ", block size: "
                  << block_size
                  << ", warmups: 1, measured iterations: "
                  << iterations << '\n'
                  << "dimension,serial_ms,blocked_ms,parallel_ms,"
                     "blocked_speedup,parallel_speedup\n";

        constexpr std::size_t dimensions[] = {64, 256, 512, 1024};
        for (const std::size_t dimension : dimensions) {
            if (dimension > maximum_size) {
                continue;
            }
            const auto left = make_matrix(dimension, 0.1F);
            const auto right = make_matrix(dimension, 0.2F);
            const auto reference = left.matmul(right);
            const auto blocked_result =
                left.matmul_blocked(right, block_size);
            const auto parallel_result =
                left.matmul_parallel(right, block_size);
            for (std::size_t index = 0; index < reference.size(); ++index) {
                if (std::abs(reference.get_data()[index] -
                             blocked_result.get_data()[index]) > 1e-4F ||
                    std::abs(reference.get_data()[index] -
                             parallel_result.get_data()[index]) > 1e-4F) {
                    throw std::runtime_error(
                        "Optimized matrix multiplication differs from baseline");
                }
            }

            double checksum = 0.0;
            const double serial = measure_average(
                iterations, [&] { return left.matmul(right); }, checksum);
            const double blocked = measure_average(
                iterations,
                [&] {
                    return left.matmul_blocked(right, block_size);
                },
                checksum);
            const double parallel = measure_average(
                iterations,
                [&] {
                    return left.matmul_parallel(right, block_size);
                },
                checksum);
            const double blocked_speedup = serial / blocked;
            const double parallel_speedup = serial / parallel;

            std::cout << std::fixed << std::setprecision(3) << dimension << ','
                      << serial * 1000.0 << ',' << blocked * 1000.0 << ','
                      << parallel * 1000.0 << ',' << blocked_speedup << ','
                      << parallel_speedup << '\n';
            std::cout << "  result checksum: " << checksum << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Benchmark error: " << error.what() << '\n';
        return 1;
    }
}
