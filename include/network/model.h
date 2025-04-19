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
        
        Matrix<T> pass(Matrix<T> x) {
            size_t batch_size = x.rows();
            size_t output_size = (this->layers.back())->weights().cols();

            Matrix<T> y_hat (batch_size, output_size);

            for(size_t i = 0; i < batch_size; i++) {
                Matrix<T> input = x.getRow(i);
                for(Layer<T>* layer: this->layers) {
                    input = layer->pass(input);
                }
                for(size_t col_j = 0; col_j < input.cols(); col_j++) {
                    y_hat(i, col_j) = input(0UL, col_j);
                }
            }
            return y_hat;
        }

        void train(Matrix<T> X, Matrix<T> Y, size_t iterations = 1000, float learning_rate = 0.0001f) {
            assert(X.rows() == Y.rows());
            int training_rows = X.rows();

            for(size_t iteration = 0; iteration < iterations; iteration++) {
                for(size_t train_i = 0; train_i < training_rows; train_i++) {
                    
                    Matrix<T> x = X.getRow(train_i);
                    Matrix<T> y = Y.getRow(train_i);

                    for(int layer_i = this->layers.size()-1; layer_i >= 0; layer_i--) {
                        (this->layers[layer_i])->backprop(x, y, learning_rate);
                        //x = (this->layers[layer_i])->pass(x);
                        x = this->pass(x);
                    }
                }
            }
        }
};