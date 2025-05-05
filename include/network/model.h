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

        Matrix<T> clip_gradients(Matrix<T> grads, T max_value) {
            for (size_t i = 0; i < grads.rows(); i++) {
                for (size_t j = 0; j < grads.cols(); j++) {
                    if (grads(i, j) > max_value) {
                        grads(i, j) = max_value;
                    } else if (grads(i, j) < -max_value) {
                        grads(i, j) = -max_value;
                    }
                }
            }
            return grads;
        }        

        void train(Matrix<T> X, Matrix<T> Y, size_t iterations = 1000, float learning_rate = 0.0001f, int batches_size = 32) {
            assert(X.rows() == Y.rows());

            for(size_t iteration = 0; iteration < iterations; iteration++) {
                //std::cout << "Iteration number: " << iteration+1 << "/" << iterations << " " << ((float)(iteration+1)/(float)iterations)*100.0f << "%" << std::endl;

                auto [batch_x, batch_y] = getBatchOfSize<T>(X, Y, batches_size);
                int batch_size = batch_x.rows();
                

                for(size_t train_i = 0; train_i < batch_size; train_i++) {

                    

                    //std::cout << "Epoch number: " << train_i+1 << "/" << batch_size << std::endl;

                    // # Get one sample.
                    Matrix<T> x = batch_x.getRow(train_i);
                    Matrix<T> y = batch_y.getRow(train_i);

                    for(auto& layer : this->layers) {
                        x = layer->pass(x);
                    }

                    Matrix<T> dE_da = (x - y) * 2;
                    dE_da = clip_gradients(dE_da, 10.0f);

                    float error = 0;
                    for(int i = 0; i < x.rows(); i++)
                        error += (y(i, 0) - x(i, 0)) * (y(i, 0) - x(i, 0));
                    error /= x.rows(); 

                    if(train_i % 10 == 0)
                        std::cout << "Loss at epoch " << train_i+1 << ": " << error << std::endl;


                    // # Backwards pass.
                    Matrix<T> dE = dE_da;
                    for(int i = layers.size() - 1; i >= 0; i--) {

                        


                        dE = this->layers[i]->backprop(dE, learning_rate);
                    }
                }
            }
        }
};