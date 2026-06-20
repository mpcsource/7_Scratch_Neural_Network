#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(ans) do { cuda_assert((ans), __FILE__, __LINE__); } while (0)
inline void cuda_assert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %d: %s at %s:%d\n",
                code, cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define CUDA_KERNEL_CHECK() do {                                           \
    cudaError_t e = cudaGetLastError();                                    \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA kernel launch error %d: %s at %s:%d\n",      \
                e, cudaGetErrorString(e), __FILE__, __LINE__);             \
        exit(e);                                                           \
    }                                                                      \
    e = cudaDeviceSynchronize();                                           \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA kernel sync error %d: %s at %s:%d\n",        \
                e, cudaGetErrorString(e), __FILE__, __LINE__);             \
        exit(e);                                                           \
    }                                                                      \
} while (0)


// CUDA activation kernels
__global__ void sigmoid_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

__global__ void sigmoid_derivative_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 1.0f / (1.0f + expf(-in[i]));
        out[i] = s * (1.0f - s);
    }
}

__global__ void relu_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

__global__ void relu_derivative_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] > 0.0f ? 1.0f : 0.0f;
    }
}

// CUDA matmul: C[M,N] = A[M,K] @ B[K,N]
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int K, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        float sum = 0.0f;
        for (int t = 0; t < K; t++) {
            sum += A[i * K + t] * B[t * N + j];
        }
        C[i * N + j] = sum;
    }
}

// Fused matmul + add_bias: C[M,N] = A[M,K] @ B[K,N] + bias[M,1] (broadcast)
__global__ void dot_add_bias_kernel(const float* A, const float* B, const float* bias,
                                     float* C, int M, int K, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        float sum = 0.0f;
        for (int t = 0; t < K; t++) {
            sum += A[i * K + t] * B[t * N + j];
        }
        C[i * N + j] = sum + bias[i];
    }
}

// Persistent temp buffer pool (avoids cudaMalloc/cudaFree per call)
namespace {
struct TempBuffers {
    float* buf0 = nullptr;  // general purpose (e.g. activation input, matmul A)
    float* buf1 = nullptr;  // general purpose (e.g. activation output, matmul B)
    float* buf2 = nullptr;  // general purpose (e.g. matmul C, bias)
    float* buf3 = nullptr;  // extra (e.g. fused dot+add_bias output)
    int capacity = 0;

    void ensure(int size) {
        if (size > capacity) {
            float* new0 = nullptr, *new1 = nullptr, *new2 = nullptr, *new3 = nullptr;
            CUDA_CHECK(cudaMalloc(&new0, static_cast<size_t>(size) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&new1, static_cast<size_t>(size) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&new2, static_cast<size_t>(size) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&new3, static_cast<size_t>(size) * sizeof(float)));
            if (buf0) CUDA_CHECK(cudaFree(buf0));
            if (buf1) CUDA_CHECK(cudaFree(buf1));
            if (buf2) CUDA_CHECK(cudaFree(buf2));
            if (buf3) CUDA_CHECK(cudaFree(buf3));
            buf0 = new0; buf1 = new1; buf2 = new2; buf3 = new3;
            capacity = size;
        }
    }

    ~TempBuffers() {
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (buf2) cudaFree(buf2);
        if (buf3) cudaFree(buf3);
    }
};
TempBuffers g_temp;

#define LAUNCH_ACTIVATION_KERNEL_PERSISTENT(kernel_name)                     \
    do {                                                                     \
        g_temp.ensure(size);                                                 \
        CUDA_CHECK(cudaMemcpy(g_temp.buf0, h_data.data(),                   \
                              sizeof(float) * size,                          \
                              cudaMemcpyHostToDevice));                      \
        int threads = 256;                                                    \
        int blocks = (size + threads - 1) / threads;                         \
        kernel_name<<<blocks, threads>>>(g_temp.buf1, g_temp.buf0, size);    \
        CUDA_KERNEL_CHECK();                                                  \
        CUDA_CHECK(cudaMemcpy(out.h_data.data(), g_temp.buf1,               \
                              sizeof(float) * size,                          \
                              cudaMemcpyDeviceToHost));                      \
    } while (0)
}

#endif

namespace {

std::vector<int> make_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

int compute_size(const std::vector<int>& shape) {
    if (shape.empty()) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

}

// ============
// Constructors
// ============

Tensor::Tensor(const std::vector<int>& in_shape, const std::vector<float>& in_data)
    : shape(in_shape.begin(), in_shape.end()),
      strides(make_strides(shape)),
      ndim(static_cast<int>(shape.size())),
      size(compute_size(shape)),
      h_data(size, 0.0f),
      h_grad(size, 0.0f) {
    if (in_data.empty()) {
        return;
    }

    if (static_cast<int>(in_data.size()) != size) {
        throw std::invalid_argument("Tensor data size does not match shape product");
    }

    std::copy(in_data.begin(), in_data.end(), h_data.begin());
}

// ===============
// Math operations
// ===============

// Addition
Tensor Tensor::add_tensor(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shape mismatch in add_tensor");
    }

    Tensor out(shape);
    for (int i = 0; i < size; i++) {
        out.h_data[i] = h_data[i] + other.h_data[i];
    }

    return out;
}

// Add bias (column vector) to each column
Tensor Tensor::add_bias(const Tensor& bias) const {
    if (ndim != 2 || bias.ndim != 2) {
        throw std::invalid_argument("add_bias expects 2D tensors");
    }
    if (shape[0] != bias.shape[0] || bias.shape[1] != 1) {
        throw std::invalid_argument("Shape mismatch in add_bias");
    }

    Tensor out(shape);
    for (int j = 0; j < shape[1]; j++) {
        for (int i = 0; i < shape[0]; i++) {
            out.h_data[i * shape[1] + j] = h_data[i * shape[1] + j] + bias.h_data[i];
        }
    }
    return out;
}

// Subtraction
Tensor Tensor::sub_tensor(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shape mismatch in sub_tensor");
    }

    Tensor out(shape);
    for (int i = 0; i < size; i++) {
        out.h_data[i] = h_data[i] - other.h_data[i];
    }

    return out;
}

// Element-wise multiplication
Tensor Tensor::mul_tensor(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shape mismatch in mul_tensor");
    }

    Tensor out(shape);
    for (int i = 0; i < size; i++) {
        out.h_data[i] = h_data[i] * other.h_data[i];
    }

    return out;
}

// Element-wise multiplication by number
Tensor Tensor::mul_tensor_number(float other) const{
    Tensor out(shape);
    for (int i = 0; i < size; i++) {
        out.h_data[i] = h_data[i] * other;
    }

    return out;
}

// Dot product
Tensor Tensor::dot_tensor(const Tensor& other) const{
    if (ndim != 2 || other.ndim != 2) {
        throw std::invalid_argument("dot_tensor expects 2D tensors");
    }
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Shape mismatch in dot_tensor");
    }

    const int m = shape[0];
    const int k = shape[1];
    const int n = other.shape[1];
    Tensor out(std::vector<int>{m, n});

#ifdef USE_CUDA
    int total = m * n;
    int a_size = m * k;
    int b_size = k * n;
    g_temp.ensure(std::max({total, a_size, b_size}));
    CUDA_CHECK(cudaMemcpy(g_temp.buf0, h_data.data(), sizeof(float) * a_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_temp.buf1, other.h_data.data(), sizeof(float) * b_size, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (n + 15) / 16);
    matmul_kernel<<<grid, block>>>(g_temp.buf0, g_temp.buf1, g_temp.buf2, m, k, n);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(out.h_data.data(), g_temp.buf2, sizeof(float) * total, cudaMemcpyDeviceToHost));
#else
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int t = 0; t < k; t++) {
                acc += h_data[i * k + t] * other.h_data[t * n + j];
            }
            out.h_data[i * n + j] = acc;
        }
    }
#endif

    return out;
}

// Fused dot product + add bias
Tensor Tensor::dot_add_bias_tensor(const Tensor& other, const Tensor& bias) const {
    if (ndim != 2 || other.ndim != 2 || bias.ndim != 2) {
        throw std::invalid_argument("dot_add_bias expects 2D tensors");
    }
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Shape mismatch in dot_add_bias");
    }
    if (shape[0] != bias.shape[0] || bias.shape[1] != 1) {
        throw std::invalid_argument("Shape mismatch in dot_add_bias (bias)");
    }

    const int m = shape[0];
    const int k = shape[1];
    const int n = other.shape[1];
    Tensor out(std::vector<int>{m, n});

#ifdef USE_CUDA
    int total = m * n;
    int a_size = m * k;
    int b_size = k * n;
    g_temp.ensure(std::max({total, a_size, b_size, m}));
    CUDA_CHECK(cudaMemcpy(g_temp.buf0, h_data.data(), sizeof(float) * a_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_temp.buf1, other.h_data.data(), sizeof(float) * b_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_temp.buf2, bias.h_data.data(), sizeof(float) * m, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (n + 15) / 16);
    dot_add_bias_kernel<<<grid, block>>>(g_temp.buf0, g_temp.buf1, g_temp.buf2, g_temp.buf3, m, k, n);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(out.h_data.data(), g_temp.buf3, sizeof(float) * total, cudaMemcpyDeviceToHost));
#else
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int t = 0; t < k; t++) {
                sum += h_data[i * k + t] * other.h_data[t * n + j];
            }
            out.h_data[i * n + j] = sum + bias.h_data[i];
        }
    }
#endif

    return out;
}

// ====================
// Activation functions
// ====================

Tensor Tensor::sigmoid_tensor() const {
    Tensor out(shape);
#ifdef USE_CUDA
    LAUNCH_ACTIVATION_KERNEL_PERSISTENT(sigmoid_kernel);
#else
    for (int i = 0; i < size; i++)
        out.h_data[i] = 1.0f / (1.0f + std::exp(-h_data[i]));
#endif
    return out;
}

Tensor Tensor::sigmoid_derivative_tensor() const {
    Tensor out(shape);
#ifdef USE_CUDA
    LAUNCH_ACTIVATION_KERNEL_PERSISTENT(sigmoid_derivative_kernel);
#else
    for (int i = 0; i < size; i++) {
        float s = 1.0f / (1.0f + std::exp(-h_data[i]));
        out.h_data[i] = s * (1.0f - s);
    }
#endif
    return out;
}

Tensor Tensor::relu_tensor() const {
    Tensor out(shape);
#ifdef USE_CUDA
    LAUNCH_ACTIVATION_KERNEL_PERSISTENT(relu_kernel);
#else
    for (int i = 0; i < size; i++)
        out.h_data[i] = h_data[i] > 0.0f ? h_data[i] : 0.0f;
#endif
    return out;
}

Tensor Tensor::relu_derivative_tensor() const {
    Tensor out(shape);
#ifdef USE_CUDA
    LAUNCH_ACTIVATION_KERNEL_PERSISTENT(relu_derivative_kernel);
#else
    for (int i = 0; i < size; i++)
        out.h_data[i] = h_data[i] > 0.0f ? 1.0f : 0.0f;
#endif
    return out;
}

// Transpose (2D only)
Tensor Tensor::transpose_tensor() const {
    if (ndim != 2) {
        throw std::invalid_argument("transpose_tensor expects 2D tensors");
    }
    Tensor out(std::vector<int>{shape[1], shape[0]});
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            out.h_data[j * shape[0] + i] = h_data[i * shape[1] + j];
        }
    }
    return out;
}

// Sum across columns (2D only), returns rows×1
Tensor Tensor::sum_cols_tensor() const {
    if (ndim != 2) {
        throw std::invalid_argument("sum_cols_tensor expects 2D tensors");
    }
    Tensor out(std::vector<int>{shape[0], 1});
    for (int i = 0; i < shape[0]; i++) {
        float sum = 0.0f;
        for (int j = 0; j < shape[1]; j++) {
            sum += h_data[i * shape[1] + j];
        }
        out.h_data[i] = sum;
    }
    return out;
}

// ================
// Gradient methods
// ================

// Zero out the gradient buffer
void Tensor::zero_grad() {
    std::fill(h_grad.begin(), h_grad.end(), 0.0f);
#ifdef USE_CUDA
    if (d_grad != nullptr)
        CUDA_CHECK(cudaMemset(d_grad, 0, static_cast<size_t>(size) * sizeof(float)));
#endif
}

// Accumulate gradient: h_grad += incoming.h_data
void Tensor::accumulate_grad(const Tensor& incoming) {
    if (incoming.size != size) {
        throw std::invalid_argument("Gradient shape mismatch in accumulate_grad");
    }

    for (int i = 0; i < size; i++)
        h_grad[i] += incoming.h_data[i];
}

// Return gradient buffer wrapped as a new Tensor
Tensor Tensor::get_grad() const {
    Tensor out(shape, h_grad);
    return out;
}
