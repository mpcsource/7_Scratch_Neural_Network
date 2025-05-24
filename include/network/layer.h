#pragma once
#include <vector>
#include <math.h>
#include "math/matrix.h"
#include "math/random.h"

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>

class Layer
{
protected:
    Matrix<T> weights_;
    Matrix<T> biases_;
    std::string activation_;

public:
    Matrix<T> delta_z;
    Matrix<T> x_;  // Unmodified input
    Matrix<T> a_;  // Output after activation function
    Matrix<T> z_;  // Output before activation function
    Matrix<T> da_; // Derivative of the activation function
    // # Initialises weights as 1 and biases as 0.
    /*
    Layer(int n, int f) : neurons_(n), features_(f), weights_(Matrix<T>(n,f,1)), biases_(Matrix<T>(1,n,0)) {}

    // # Uses one of multiple initialisation functions.
    Layer(int n, int f, std::string init_func) : neurons_(n), features_(f), weights_(Matrix<T>(n,f,1)), biases_(Matrix<T>(1,n,0)) {
        if(init_func == "glorot") {
            T limit = sqrt(6.0f/(this->features_+this->neurons_));
            for(int i = 0; i < n; i++)
                for(int j = 0; j < f; j++) {
                    this->weights_(i,j) = randomRange(-limit, limit); // # This obviously needs to change.
                }
        }
    }
        */

    // NEW CONSTRUCTOR.
    Layer(int nin, int nout, std::string activation = "sigmoid") : weights_(Matrix<T>(nout, nin, 1)),
                                                                   biases_(Matrix<T>(nout, 1, 0.0)),
                                                                   activation_(activation)
    {
        // # Initialise weights (Xavier).
        T limit = sqrt(6.0f / (nin + nout));
        for (int i = 0; i < nout; i++)
            for (int j = 0; j < nin; j++)
            {
                this->weights_(i, j) = randomRange(-limit, limit); // # This obviously needs to change.
            }
    }

    // # Perform full pass.
    void pass(Matrix<T> x)
    {
        this->x_ = x;
        this->z_ = this->weights_.dot(x).add(this->biases_);

        if (this->activation_ == "sigmoid")
        {
            auto sigmoid = [](T x)
            { return 1.0 / (1.0 + std::exp(-x)); };
            auto sigmoid_deriv = [](T x)
            {
                T s = 1.0 / (1.0 + std::exp(-x));
                return s * (1.0 - s);
            };
            this->a_ = this->z_.apply(sigmoid);
            this->da_ = this->z_.apply(sigmoid_deriv);
        }
        else // Linear by default.
        {
            this->a_ = this->z_;
            this->da_ = Matrix<T>(this->z_.rows(), this->z_.cols(), 1.0);
        }
    }

    // # One iteration of backprop.
    void backward(float learning_rate)
    {

        Matrix<T> grad_w = this->delta_z.dot(this->x_.transpose());
        Matrix<T> grad_b = this->delta_z;

        this->weights_ = this->weights_.subtract(grad_w.multiply(learning_rate));
        this->biases_ = this->biases_.subtract(grad_b.multiply(learning_rate));
    }

    // # Update all weights.
    void updateAllWeights(Matrix<T> weights)
    {
        // Guarantee new weights matrix has the same shape as current one.
        assert(this->weights_.rows() == weights.rows());
        assert(this->weights_.cols() == weights.cols());

        // Replace.
        this->weights_ = weights;
    }

    // # Update specific weight.
    void updateWeight(int r, int c, T val)
    {
        this->weights_(r, c) = val;
    }

    // # Access weights.
    Matrix<T> &weights()
    {
        return this->weights_;
    }
    const Matrix<T> &weights() const
    {
        return this->weights_;
    }

    // # Access biases.
    Matrix<T> &biases()
    {
        return this->biases_;
    }
    const Matrix<T> &biases() const
    {
        return this->biases_;
    }

    // # Print weights and biases.
    void debugPrint()
    {
        std::cout << "l1 weights:" << std::endl;
        this->weights().basicPrint();
        std::cout << "l1 biases:" << std::endl;
        this->biases().basicPrint();
    }
};