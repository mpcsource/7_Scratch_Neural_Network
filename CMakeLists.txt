cmake_minimum_required(VERSION 3.14)
project(ScratchNeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

# Include GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(include)

# Automatically find all source and test files
#file(GLOB_RECURSE SOURCE_FILES src/**/*.cpp)
file(GLOB_RECURSE TEST_FILES tests/*.cpp)

# Optionally split utils into their own libraries if needed, but you can also just use object libraries or combine all into one
#add_library(core_lib ${SOURCE_FILES})

# Create a single test executable from all test files
add_executable(unit_tests ${TEST_FILES})

# Link all necessary libraries
target_link_libraries(unit_tests PRIVATE gtest_main)
#target_link_libraries(unit_tests PRIVATE core_lib gtest_main)

# Register with CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)
