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

    // ===============
    // Math operations
    // ===============

    // Addition
    Tensor add_tensor(const Tensor& other) const;

    // Subtraction
    Tensor sub_tensor(const Tensor& other) const;

    // Element-wise multiplication
    Tensor mul_tensor(const Tensor& other) const;

    // Element-wise multiplication by number
    Tensor mul_tensor_number(float other) const;

    // Dot product
    Tensor dot_tensor(const Tensor& other) const;

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