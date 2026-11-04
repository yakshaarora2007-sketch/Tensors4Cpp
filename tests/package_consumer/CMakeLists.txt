cmake_minimum_required(VERSION 3.16)
project(MiniTensorPackageConsumer LANGUAGES CXX)

find_package(MiniTensor CONFIG REQUIRED)
add_executable(package_consumer main.cpp)
target_link_libraries(package_consumer PRIVATE MiniTensor::minitensor)
