#pragma once
#include <vector>
#include <math.h>
#include "math/matrix.h"

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>

class Layer {
    private:
        Matrix<T> weights_;
        Matrix<T> biases_;
        // # neurons_: amount of neurons in the layer, also rows of layer output.
        // # features_: amount of features in one input.
        int neurons_, features_; 

        

    public:
        Layer(int n, int f) : neurons_(n), features_(f), weights_(Matrix<T>(n,f,1)), biases_(Matrix<T>(n,1,1)) {} /* Biases might be wrong, check later. */
    
        // # Perform full pass.
        Matrix<T> pass(Matrix<T> x) {
            return calculate_activation(calculate_z(x));
        }   
        
        // # Calculate activation of layer. 
        Matrix<T> calculate_activation(Matrix<T> x) {
            return x; // # Currently linear.
        }

        // # Calculate z of layer.
        Matrix<T> calculate_z(Matrix<T> x) {
            //Matrix<T> z = x * this->weights_;
            //z = z + this->biases_;
            //return z;
            Matrix<T> input = 
        }

        // # One iteration of backprop.
        void backprop(Matrix<float> x, Matrix<float> y, float learning_rate) {
            // # Update weights.
            Matrix<float> a = pass(x);

            Matrix<float> dz_dw = x;

            Matrix<float> da_dz = this->weights_;
            Matrix<float> dE_da = (a - y) * 2;
            Matrix<float> delta = da_dz * dE_da;

            Matrix<float> dE_dw = dz_dw * delta;

            this->weights_ = this->weights_ - dE_dw * learning_rate;

            // # Update biases.

            Matrix<float> dz_db (1,1,1.0f);
            Matrix<float> dE_db = dz_db * delta;

            this->biases_ = this->biases_ - dE_db * learning_rate;

        }

        // # Update all weights.
        void updateAllWeights(Matrix<T> weights) {
            // Guarantee new weights matrix has the same shape as current one.
            assert(this->weights_.rows() == weights.rows());
            assert(this->weights_.cols() == weights.cols());

            // Replace.
            this->weights_ = weights;
        }

        // # Update specific weight.
        void updateWeight(int r, int c, T val) {
            this->weights_(r, c) = val;
        }

        // # Access weights.
        Matrix<T>& weights() {
            return this->weights_;
        }
        const Matrix<T>& weights() const {
            return this->weights_;
        }

        // # Access biases.
        Matrix<T>& biases() {
            return this->biases_;
        }
        const Matrix<T>& biases() const {
            return this->biases_;
        }

    };