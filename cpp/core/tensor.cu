#include "tensor.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

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

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int t = 0; t < k; t++) {
                acc += h_data[i * k + t] * other.h_data[t * n + j];
            }
            out.h_data[i * n + j] = acc;
        }
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
        cudaMemset(d_grad, 0, static_cast<size_t>(size) * sizeof(float));
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
