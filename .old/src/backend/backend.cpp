#include "backend/backend.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

static bool g_cuda_available = false;
static Backend g_backend = Backend::CPU;

bool cuda_available() {
#ifdef USE_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess || count == 0)
        return false;

    return true;
#else
    return false;
#endif
}

void initialize_backend() {
    g_cuda_available = cuda_available();

    if (g_cuda_available)
        g_backend = Backend::CUDA;
    else
        g_backend = Backend::CPU;
}

Backend get_backend() {
    return g_backend;
}