cmake_minimum_required(VERSION 3.14)
project(ScratchNeuralNetwork)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Cuda (must be before any targets are defined)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_BIN_DIR}/nvcc)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Include GoogleTest + pybind11
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

# Core library headers live in cpp/core
include_directories(cpp/core)

# --- Core library ---
# C++ sources (none yet besides headers), plus CUDA sources
file(GLOB_RECURSE CPP_SOURCE_FILES cpp/core/*.cpp)
file(GLOB_RECURSE CUDA_SOURCE_FILES cpp/core/*.cu)

add_library(core_lib ${CPP_SOURCE_FILES} ${CUDA_SOURCE_FILES})
set_target_properties(core_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CUDAToolkit_FOUND)
    set_target_properties(core_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(core_lib PRIVATE CUDA::cudart)
endif()

# --- pybind11 module ---
pybind11_add_module(scratchneuralnetwork NO_EXTRAS cpp/bindings/pybind_module.cpp)  
target_link_libraries(scratchneuralnetwork PRIVATE core_lib)