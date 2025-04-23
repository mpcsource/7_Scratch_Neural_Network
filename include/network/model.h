#pragma once
#include <vector>
#include <iostream>
#include "network/layer.h"
#include "math/matrix.h"

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>

class Model {
    private:
        std::vector<Layer<T>*> layers;

    public:
        Model() {}

        void appendLayer(Layer<T>& layer) { 
            this->layers.push_back(&layer); 
        }
        
        Matrix<T> passOne(Matrix<T> x) {
            for(Layer<T> * layer: this->layers) {
                x = layer->pass(x);
            }
            return x;
        }

        Matrix<T> pass(Matrix<T> x) {
            size_t batch_size = x.rows();
            size_t output_size = (this->layers.back())->weights().rows();

            Matrix<T> y_hat (batch_size, output_size);

            for(size_t i = 0; i < x.rows(); i++) {
                Matrix x_ = x.getRow(i);
                Matrix y_ = this->passOne(x_);
                assert(y_hat.cols() == y_.cols());
                for(size_t j = 0; j < y_.cols(); j++) {
                    y_hat(i,j) = y_(0UL,j);
                }
            } 

            return y_hat;
        }

        void train(Matrix<T> X, Matrix<T> Y, size_t iterations = 1000, float learning_rate = 0.0001f) {
            assert(X.rows() == Y.rows());
            int training_rows = X.rows();

            for(size_t iteration = 0; iteration < iterations; iteration++) {
                std::cout << "Iteration number: " << iteration+1 << std::endl;

                for(size_t train_i = 0; train_i < training_rows; train_i++) {
                    
                    // # Get one sample.
                    Matrix<T> x = X.getRow(train_i);
                    Matrix<T> y = Y.getRow(train_i);

                    // # Forward pass & store activations.
                    std::vector<Matrix<T>> activations;
                    activations.push_back(x);

                    for(auto& layer : this->layers) {
                        x = layer->pass(x);
                        activations.push_back(x);
                    }

                    // # Compute initial gradient from loss.
                    Matrix<T> a_last = activations.back();
                    Matrix<T> dE_da = (a_last - y) * 2;

                    // # Backwards pass.
                    Matrix<T> dE = dE_da;
                    for(int i = layers.size() - 1; i >= 0; i--) {
                        Matrix<T> input = activations[i];
                        dE = this->layers[i]->backprop(input, dE, learning_rate);
                    }
                }
            }
        }
};