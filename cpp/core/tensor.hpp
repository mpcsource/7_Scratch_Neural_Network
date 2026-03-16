#include <vector>

class Tensor 
{

private:
    int rows_, cols_;
    


    // CPU
    std::vector<float> h_data; // Internal data
    //std::vector<float> h_grad; // Self gradient
    
    // GPU
    float* d_data = nullptr;
    //float* d_grad = nullptr;
    mutable bool cpu_dirty = false; // d_data is newer than h_data
    mutable bool gpu_dirty = true;  // h_data is newer than d_data (always true on construction)


public:

    // ============
    // Constructors
    // ============

    // Single constructor
    Tensor(int rows, int cols);

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

};