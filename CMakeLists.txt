cmake_minimum_required(VERSION 3.11)
project(pythagorean-demo CXX)

find_package(xtensor CONFIG REQUIRED)
find_package(xsimd CONFIG REQUIRED)
add_executable(main main.cc)
target_link_libraries(main PRIVATE xtensor xsimd)
