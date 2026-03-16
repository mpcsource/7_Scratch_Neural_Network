#include "tensor.hpp"

// ============
// Constructors
// ============

// Empty constructor
Tensor::Tensor(int rows, int cols) : rows_(rows), cols_(cols), h_data() {}

// ===============
// Math operations
// ===============

// Addition
Tensor Tensor::add_tensor(const Tensor& other) const {
    Tensor out(0, 0);

    return out;
}

// Subtraction
Tensor Tensor::sub_tensor(const Tensor& other) const {
    Tensor out(0, 0);

    return out;
}

// Element-wise multiplication
Tensor Tensor::mul_tensor(const Tensor& other) const {
    Tensor out(0, 0);

    return out;
}

// Element-wise multiplication by number
Tensor Tensor::mul_tensor_number(float other) const{
    Tensor out(0, 0);

    return out;
}

// Dot product
Tensor Tensor::dot_tensor(const Tensor& other) const{
    Tensor out(0, 0);

    return out;
}
