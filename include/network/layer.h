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
        Layer(int n, int f) : neurons_(n), features_(f), weights_(Matrix<T>(n,f,1)), biases_(Matrix<T>(1,f,1)) {} /* Biases might be wrong, check later. */
    
        // # Perform full pass.
        Matrix<T> pass(Matrix<T> x) {
            return calculate_activation(x);
        }   
        
        // # Calculate activation of layer. 
        Matrix<T> calculate_activation(Matrix<T> x) {
            return calculate_z(x); // # Currently linear.
        }

        // # Calculate z of layer.
        Matrix<T> calculate_z(Matrix<T> x) {
            return x * this->weights_ + this->biases_;
        }

        // # Access weights.
        Matrix<T>& weights() {
            return this->weights_;
        }

        // # Access biases.
        Matrix<T>& biases() {
            return this->biases_;
        }
    };