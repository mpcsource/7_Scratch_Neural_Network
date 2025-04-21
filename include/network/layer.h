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
        Layer(int n, int f) : neurons_(n), features_(f), weights_(Matrix<T>(n,f,1)), biases_(Matrix<T>(1,n,1)) {} /* Biases might be wrong, check later. */
    
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
            /*Matrix<T> z = this->weights_ * x;
            z = z + this->biases_;
            return z;*/
            return x * this->weights_.transpose() + this->biases_;
        }

        // # One iteration of backprop.
        Matrix<T> backprop(Matrix<float> x, Matrix<float> dE, float learning_rate) {
            // # Forward pass.
            Matrix<float> a = pass(x);

            // # It's linear so doesn't change.
            Matrix<float> dZ = dE;

            // # Loss gradient w.r.t. weights.
            Matrix<float> dE_dw = dZ.transpose() * x;

            // # Loss gradient w.r.t. bias.
            Matrix<float> dE_db = dZ;

            // # Update weights and biases.
            this->weights_ = this->weights_ - dE_dw * learning_rate;
            this->biases_ = this->biases_ - dE_db * learning_rate;

            // # Return 
            return dZ * this->weights_;
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

        // # Print weights and biases.
        void debugPrint() {
            std::cout << "l1 weights:" << std::endl;
            this->weights().basicPrint();
            std::cout << "l1 biases:" << std::endl;
            this->biases().basicPrint();
        }

    };