cmake_minimum_required(VERSION 3.14)
project(ScratchNeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

# Include GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)

FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/v2.13.6.zip
)

FetchContent_MakeAvailable(googletest pybind11)

enable_testing()

include_directories(include)

# Automatically find all source and test files
file(GLOB_RECURSE SOURCE_FILES src/**/*.cpp)
list(REMOVE_ITEM SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/python/bindings.cpp")
file(GLOB_RECURSE TEST_FILES tests/*.cpp)

# Optionally split utils into their own libraries if needed, but you can also just use object libraries or combine all into one
add_library(core_lib ${SOURCE_FILES})

# pybind11
pybind11_add_module(scratchneuralnetwork src/python/bindings.cpp)
target_link_libraries(scratchneuralnetwork PRIVATE core_lib)

# Create a single test executable from all test files
add_executable(unit_tests ${TEST_FILES})

find_package(OpenMP REQUIRED)
target_link_libraries(unit_tests PUBLIC OpenMP::OpenMP_CXX)

# Link all necessary libraries
target_link_libraries(unit_tests PRIVATE gtest_main)
target_link_libraries(unit_tests PRIVATE core_lib gtest_main)

# Register with CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)
