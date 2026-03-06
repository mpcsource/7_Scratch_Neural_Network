#pragma once
#include <vector>
#include <iostream>
#include "network/layer.hpp"
#include "math/matrix.hpp"
#include "data/loader.hpp"

class Model {
    private:
        std::vector<Layer*> layers_;
        std::string loss_;
        Matrix x_;
        Matrix out_;

    public:
        Model(std::string loss);

        void appendLayer(Layer * layer);

        Matrix forward(Matrix x);

        void backward(Matrix label, float learning_rate = 0.01f, int batch_size = 32);

        void backprop(Matrix data, Matrix labels, int epochs = 10, float learning_rate = 0.01f, int batch_size = 32);

        Matrix test(Matrix data, Matrix labels);    
};