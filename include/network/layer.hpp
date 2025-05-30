#pragma once
#include <vector>
#include <math.h>
#include "math/matrix.hpp"
#include "math/random.hpp"


class Layer
{
protected:
    Matrix weights_;
    Matrix biases_;
    std::string activation_;

public:
    Matrix delta_z;
    Matrix x_;  // Unmodified input
    Matrix a_;  // Output after activation function
    Matrix z_;  // Output before activation function
    Matrix da_; // Derivative of the activation function

    Layer(int nin, int nout, std::string activation = "sigmoid");

    // Perform full pass.
    void pass(Matrix x);
    
    // One iteration of backprop.
    void backward(float learning_rate);

    // Write weights.
    Matrix &weights();

    // Read weights.
    const Matrix &weights() const;

    // Access biases.
    Matrix &biases();
    const Matrix &biases() const;

    // Print weights and biases.
    void debugPrint();
};