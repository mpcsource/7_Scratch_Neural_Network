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
        std::vector<Layer<T>*> layers_;
        std::string loss_;
        Matrix<T> x_;
        Matrix<T> out_;

    public:
        Model(std::string loss) : loss_(loss) {}

        void appendLayer(Layer<T> * layer) {
            this->layers_.push_back(layer);
        }

        Matrix<T> forward(Matrix<T> x) {
            this->x_ = x;
            for(Layer<T>* layer : this->layers_) {
                layer->pass(x);
                x = layer->a_;
            }
            return x;
        }

        void backward(Matrix<T> label, float learning_rate = 0.01f) {

            Layer<T> * out_layer = this->layers_.back();
            Matrix<T> dloss = out_layer->a_.subtract(label);
            Matrix<T> delta_o = dloss.multiply(out_layer->da_);
            
            out_layer->delta_z = delta_o;
            out_layer->backward(learning_rate);

            for (int i = this->layers_.size() - 1; i > 0; i--) {
                if(i == this->layers_.size()-1) continue; // Skip output layer
                Layer<T> * next_layer = this->layers_.at(i+1);
                Layer<T> * layer = this->layers_.at(i);

                auto delta_h = next_layer->weights().transpose().dot(next_layer->delta_z).multiply(layer->da_);
                layer->delta_z = delta_h;
                layer->backward(learning_rate);

            }
        }

        /**
         * data - in full
         * labels - one hot encoded 
         * */
        void backprop(Matrix<T> data, Matrix<T> labels, int epochs = 10, float learning_rate = 0.01f) {

            int size = data.rows();
            for (int epoch_i = 0; epoch_i < epochs; epoch_i++) { 
                std::cout << "Epoch: " << epoch_i+1 << std::endl;  
                for(int train_i = 0; train_i < size; train_i++) {
                    auto image = data.getRow(train_i).transpose();
                    auto label = labels.getRow(train_i).transpose();

                    auto label_hat = this->forward(image);

                    this->backward(label, learning_rate);
                }
            }
         
        }

        Matrix<T> test(Matrix<T> data, Matrix<T> labels) {
            Matrix<T> labels_hat (labels.rows(), labels.cols(), 0);
            for(int test_i = 0; test_i < data.rows(); test_i++) {
                auto image = data.getRow(test_i).transpose();
                auto label = labels.getRow(test_i).transpose();

                auto label_hat = this->forward(image);
                for(int j = 0; j < labels.cols(); j++)
                    labels_hat(test_i, j) = label_hat(0, j);
            }
            return labels_hat;
        }

        
};