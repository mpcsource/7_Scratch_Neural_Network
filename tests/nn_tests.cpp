#include <iostream>
#include "gtest/gtest.h"
#include "network/layer.h"

void printMatrix(Matrix<float> m) {
    for(size_t i = 0; i < m.rows(); i++) {
        for(size_t j = 0; j < m.cols(); j++)
            std::cout << m(i,j) << " ";
        std::cout << std::endl;
    }
}

TEST(NeuralNetworkTests, ForwardPass) {
    Matrix<float> x (1, 2, 2.0f);

    Layer<float> l1 (2, 2);

    Matrix<float> l1_z = l1.pass(x);

    printMatrix(l1_z);
}

TEST(NeuralNetworkTests, ManualBackprop) {
    Matrix<float> x (1, 2, 2.0f);
    Matrix<float> y_true (1, 2, 20.0f);

    Layer<float> l1 (2, 2);
    Layer<float> l2 (2, 2);
    Layer<float> l3 (2, 2);

    std::vector<Layer<float>*> layers = {&l1, &l2, &l3};

    Matrix<float> a1 = l1.pass(x);
    Matrix<float> a2 = l2.pass(a1);
    Matrix<float> a3 = l3.pass(a2);

    for(size_t epoch = 0; epoch < 1000; epoch++) {
        std::vector<Matrix<float>> activations = {x};
        std::vector<Matrix<float>> zs = {};

        for(Layer<float> * layer : layers) {
            Matrix<float> z = activations.back() * layer->weights() + layer->biases();
            Matrix<float> a = layer->calculate_activation(z);
            zs.push_back(z);
            activations.push_back(a);
        }

        Matrix<float> y_hat = activations.back();
        
        Matrix<float> delta = y_hat - y_true;

        for(size_t l = layers.size(); l > 0; l--) {
            Matrix<float> a_prev = activations.at(l);
            Matrix<float> z = zs[l];

            Matrix<float> dL_dW = delta * a_prev.transpose();
            Matrix<float> dL_db = delta;

            //printMatrix(layers.at(l)->weights());
            //printMatrix(dL_dW);
        }
    }
}