cmake_minimum_required(VERSION 3.18)
project(scratchneuralnetwork_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SCRATCHNEURALNETWORK_ENABLE_CUDA "Build CUDA kernels" ON)

find_package(pybind11 CONFIG REQUIRED)

if(SCRATCHNEURALNETWORK_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_BIN_DIR}/nvcc)
        enable_language(CUDA)
        add_definitions(-DUSE_CUDA)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    else()
        message(FATAL_ERROR "SCRATCHNEURALNETWORK_ENABLE_CUDA=ON but CUDA toolkit was not found")
    endif()
endif()

set(SCRATCHNEURALNETWORK_CORE_SOURCES
    core/tensor.cu
)

add_library(scratchneuralnetwork_backend STATIC ${SCRATCHNEURALNETWORK_CORE_SOURCES})
set_target_properties(scratchneuralnetwork_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(scratchneuralnetwork_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(CUDAToolkit_FOUND)
    target_link_libraries(scratchneuralnetwork_backend PRIVATE CUDA::cudart)
    if(TARGET CUDA::cudadevrt)
        target_link_libraries(scratchneuralnetwork_backend PRIVATE CUDA::cudadevrt)
    endif()
    set_target_properties(scratchneuralnetwork_backend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Python extension must be named _core, package name stays scratchneuralnetwork
pybind11_add_module(_core bindings/pybind_module.cpp)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(_core PRIVATE scratchneuralnetwork_backend)
if(CUDAToolkit_FOUND)
    target_link_libraries(_core PRIVATE CUDA::cudart)
endif()

# Install compiled extension into python package
install(TARGETS _core DESTINATION scratchneuralnetwork)