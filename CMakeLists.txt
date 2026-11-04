cmake_minimum_required(VERSION 3.16)
project(MiniTensor VERSION 1.0.0 LANGUAGES CXX)

include(CTest)

option(MINITENSOR_BUILD_EXAMPLES "Build MiniTensor examples" ON)
option(MINITENSOR_BUILD_BENCHMARKS "Build the benchmark executable" ON)
option(MINITENSOR_ENABLE_OPENMP "Enable the OpenMP matrix multiplication path" OFF)
option(MINITENSOR_ENABLE_NATIVE_OPTIMIZATIONS
       "Enable optional hardware-specific compiler optimizations" OFF)
option(MINITENSOR_ENABLE_SANITIZERS
       "Enable supported development sanitizers" OFF)
option(MINITENSOR_USE_GTEST "Use GoogleTest when it is available" ON)

add_library(minitensor INTERFACE)
add_library(MiniTensor::minitensor ALIAS minitensor)
set_target_properties(minitensor PROPERTIES EXPORT_NAME minitensor)
target_include_directories(minitensor INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(minitensor INTERFACE cxx_std_20)

if(MINITENSOR_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(minitensor INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(minitensor INTERFACE MINITENSOR_USE_OPENMP=1)
endif()

if(MINITENSOR_ENABLE_NATIVE_OPTIMIZATIONS)
    if(MSVC)
        target_compile_options(minitensor INTERFACE /arch:AVX2)
    else()
        target_compile_options(minitensor INTERFACE -march=native)
    endif()
endif()

if(MINITENSOR_ENABLE_SANITIZERS)
    if(MSVC)
        add_compile_options(/fsanitize=address)
        add_link_options(/INCREMENTAL:NO)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    else()
        message(FATAL_ERROR "Sanitizers are not configured for this compiler")
    endif()
endif()

function(minitensor_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

if(MINITENSOR_BUILD_EXAMPLES)
    add_executable(basic_tensor examples/basic_tensor.cpp)
    target_link_libraries(basic_tensor PRIVATE MiniTensor::minitensor)
    minitensor_enable_warnings(basic_tensor)

    add_executable(image_processing examples/image_processing.cpp)
    target_link_libraries(image_processing PRIVATE MiniTensor::minitensor)
    minitensor_enable_warnings(image_processing)
endif()

if(BUILD_TESTING)
    if(MINITENSOR_USE_GTEST)
        find_package(GTest QUIET)
    endif()

    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE MiniTensor::minitensor)
    if(GTest_FOUND)
        target_compile_definitions(test_tensor PRIVATE MINITENSOR_USE_GTEST)
        target_link_libraries(test_tensor PRIVATE GTest::gtest_main)
        include(GoogleTest)
        gtest_discover_tests(test_tensor)
    else()
        add_test(NAME minitensor_tests COMMAND test_tensor)
    endif()
    minitensor_enable_warnings(test_tensor)
endif()

if(MINITENSOR_BUILD_BENCHMARKS)
    add_executable(minitensor_benchmark benchmarks/benchmark.cpp)
    target_link_libraries(minitensor_benchmark PRIVATE MiniTensor::minitensor)
    target_compile_definitions(minitensor_benchmark PRIVATE
        MINITENSOR_BENCHMARK_BUILD_TYPE="$<CONFIG>"
        MINITENSOR_BENCHMARK_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
        MINITENSOR_BENCHMARK_DEBUG_FLAGS="${CMAKE_CXX_FLAGS_DEBUG}"
        MINITENSOR_BENCHMARK_RELEASE_FLAGS="${CMAKE_CXX_FLAGS_RELEASE}"
        MINITENSOR_BENCHMARK_RELWITHDEBINFO_FLAGS="${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
        MINITENSOR_BENCHMARK_MINSIZEREL_FLAGS="${CMAKE_CXX_FLAGS_MINSIZEREL}"
    )
    minitensor_enable_warnings(minitensor_benchmark)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS minitensor EXPORT MiniTensorTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES README.md LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(EXPORT MiniTensorTargets
    FILE MiniTensorTargets.cmake
    NAMESPACE MiniTensor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniTensor
)
configure_package_config_file(
    cmake/MiniTensorConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MiniTensorConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniTensor
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MiniTensorConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MiniTensorConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MiniTensorConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniTensor
)
