#include <vector>

class Tensor 
{

private:
    std::vector<int> shape;
    std::vector<int> strides;

    int ndim, size;

    // CPU
    std::vector<float> h_data; // Internal data
    std::vector<float> h_grad; // Gradient buffer (same shape as h_data)

    // GPU
    float* d_data = nullptr;
    float* d_grad = nullptr;
    mutable bool cpu_dirty = false; // d_data is newer than h_data
    mutable bool gpu_dirty = true;  // h_data is newer than d_data (always true on construction)


public:

    // ============
    // Constructors
    // ============

    // Generic constructor:
    // - in_shape defines dimensions
    // - in_data empty => zero-initialized buffer
    // - in_data provided => must match shape product
    Tensor(
        const std::vector<int>& in_shape = {},
        const std::vector<float>& in_data = {}
    );

    // Transpose (2D only)
    Tensor transpose_tensor() const;

    // Sum across columns (2D only), returns rows×1
    Tensor sum_cols_tensor() const;

    // Read access to shape
    const std::vector<int>& get_shape() const { return shape; }

    // Read access to flat data
    const std::vector<float>& get_data() const { return h_data; }

    // ===============
    // Math operations
    // ===============

    // Addition
    Tensor add_tensor(const Tensor& other) const;

    // Add bias column vector to each column
    Tensor add_bias(const Tensor& bias) const;

    // Subtraction
    Tensor sub_tensor(const Tensor& other) const;

    // Element-wise multiplication
    Tensor mul_tensor(const Tensor& other) const;

    // Element-wise multiplication by number
    Tensor mul_tensor_number(float other) const;

    // Dot product
    Tensor dot_tensor(const Tensor& other) const;

    // Fused dot product + add bias: (this @ other) + bias
    Tensor dot_add_bias_tensor(const Tensor& other, const Tensor& bias) const;

    // ====================
    // Activation functions
    // ====================

    // Sigmoid: 1 / (1 + exp(-x))
    Tensor sigmoid_tensor() const;

    // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    Tensor sigmoid_derivative_tensor() const;

    // ReLU: max(0, x)
    Tensor relu_tensor() const;

    // ReLU derivative: 1 if x > 0 else 0
    Tensor relu_derivative_tensor() const;

    // ================
    // Gradient methods
    // ================

    // Zero out the gradient buffer
    void zero_grad();

    // Accumulate gradient: h_grad += incoming.h_data
    void accumulate_grad(const Tensor& incoming);

    // Return gradient as a new Tensor (wraps h_grad data)
    Tensor get_grad() const;

};