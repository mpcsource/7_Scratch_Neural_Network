#include <iostream>
#include "gtest/gtest.h"
#include "network/layer.h"
#include "network/model.h"

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
        std::vector<Matrix<float>> zs;

        for(Layer<float> * layer : layers) {
            Matrix<float> z = activations.back() * layer->weights() + layer->biases();
            Matrix<float> a = layer->calculate_activation(z);
            zs.push_back(z);
            activations.push_back(a);
        }

        Matrix<float> y_hat = activations.back();
        
        Matrix<float> delta = y_hat - y_true;

        for(size_t l = layers.size()-1; l > 0; l) {

            
            Matrix<float> a_prev = activations.at(l);
            Matrix<float> z = zs.at(l);
            
            Matrix<float> dL_dW = delta * a_prev.transpose();
            Matrix<float> dL_db = delta;
            
            std::cout << layers.at(l)->weights().rows() << " " << layers.at(l)->weights().cols() << std::endl;
            std::cout << dL_dW.rows() << " " << dL_dW.cols() << std::endl;
            layers.at(l)->weights() = layers.at(l)->weights() - dL_dW;
            std::cout << "Hello world!" << std::endl;
            
            printMatrix(layers.at(l)->weights());
            printMatrix(dL_dW);
        }
    }
}

TEST(NeuralNetworkTests, ManualBackprop2) {
    
    Matrix<float> l1 (1, 2, 2.0f); // # Input layer.
    Matrix<float> t (1, 2, 20.0f); // # Target output.
    
    
    Layer<float> l2 (2, 2); // # Hidden layer.
    Layer<float> l3 (2, 2); // # Output layer.

    // # Derivative of the loss function w.r.t. output layer.
    auto dE = [](Matrix<float> y, Matrix<float> t) {
        return (t - y) * (t - y);
    };

    // # Derivative of the activation function w.r.t weighted sum.
    auto df = [](Layer<float> a, Matrix<float> z) {
        return 1;
    };

    // # Derivative of the weighted sum w.r.t. weights.
    auto dz = [](Matrix<float> W, float a) {
        Matrix<float> dW (W.rows(), W.cols(), a);
        return dW;
    };

    float learning_rate = 1;

    for(size_t _ = 0; _ < 100000; _++) {
        Matrix<float> z2 = l2.calculate_z(l1);
        Matrix<float> a2 = l2.calculate_activation(z2);

        Matrix<float> z3 = l3.calculate_z(a2);
        Matrix<float> a3 = l3.calculate_activation(z3);

        Matrix<float> E = dE(a3, t);

        Matrix<float> delta3 = dE(a3,  t);
        Matrix<float> dwk = dz(l3.weights(), 1.0f);

        Matrix dW3 (l3.weights().rows(), l3.weights().cols(), 0.0f);


    }
}

TEST(NeuralNetworkTests, OneMatrixLayer) {

    // # NN initialisation.

    Matrix<float> y (1, 1, 57.1243f);

    Matrix<float> x (1, 1, 5.0f);

    Matrix<float> w2 (1, 1, 1.0f);
    Matrix<float> b2 (1, 1, 1.0f);

    // # Backprop.
    float learning_rate = 0.0001f;
    for(size_t i = 0; i < 1000; i++) {
        // Update weight.
        Matrix<float> a2 = x * w2 + b2;
        
        Matrix<float> dz2_dw2 = x;
        Matrix<float> da2_dz2 = w2;
        Matrix<float> dE_da2 = (a2 - y) * 2;

        Matrix<float> dE_dw2 = dz2_dw2 * da2_dz2 * dE_da2;

        w2 = w2 - dE_dw2 * learning_rate;

        // Update bias.
        Matrix<float> a2_ = x * w2 + b2;

        Matrix<float> dz2_db2 (1, 1, 1.0f);
        Matrix<float> da2_dz2_ = w2;
        Matrix<float> dE_da2_ = (a2 * y) * 2;

        Matrix<float> dE_db2 = dz2_db2 * da2_dz2_ * dE_da2_;

        b2 = b2 - dE_dw2 * learning_rate;
    }

    // # Measures.
    
    Matrix<float> y_hat = x * w2 + b2;
    Matrix<float> error = (y - y_hat) * (y - y_hat);
    
    std::cout << "Input: " << std::endl;
    printMatrix(x);
    std::cout << "My guess: " << std::endl;
    printMatrix(y_hat);
    std::cout << "True output: " << std::endl;
    printMatrix(y);
    std::cout << "The weight: " << std::endl;
    printMatrix(w2);
    std::cout << "The bias: " << std::endl;
    printMatrix(b2);
    std::cout << "Cost/error: " << std::endl;
    printMatrix(error);
}

TEST(NeuralNetworkTests, OneLayer) {

    /**
     * INCOMPLETE.
     */

    // # NN initialisation.

    Matrix<float> y (1, 1, 57.1243f);
    Matrix<float> x (1, 1, 5.0f);

    Layer<float> l2 (1, 1);

    // # Backprop.
    float learning_rate = 0.0001f;
    for(size_t i = 0; i < 1000; i++) {

        for(Matrix<float> weight : {l2.weights(), l2.biases()}) {

        }

        // Update weights.
        Matrix<float> a2 = l2.pass(x);

        Matrix<float> dz2_dw2 = x;
        Matrix<float> da2_dz2 = l2.weights();
        Matrix<float> dE_da2 = (a2 - y) * 2;

        Matrix<float> dE_dw2 = dz2_dw2 * da2_dz2 * dE_da2;

        l2.weights() = l2.weights() - dE_dw2 * learning_rate;

        // Update biases.
    }

    // # Measures.

    Matrix<float> y_hat = l2.pass(x);
    Matrix<float> error = (y - y_hat) * (y - y_hat);

    std::cout << "Input: " << std::endl;
    printMatrix(x);
    std::cout << "My guess: " << std::endl;
    printMatrix(y_hat);
    std::cout << "True output: " << std::endl;
    printMatrix(y);
    std::cout << "The weight: " << std::endl;
    printMatrix(l2.weights());
    std::cout << "The bias: " << std::endl;
    printMatrix(l2.biases());
    std::cout << "Cost/error: " << std::endl;
    printMatrix(error);
}

TEST(NeuralNetworkTests, OneLayerBackprop) {
    // # NN initialisation.

    Matrix<float> y (1, 1, 57.1243f);
    Matrix<float> x (1, 1, 5.0f);

    Layer<float> l2 (1, 1);

    // # Backprop.
    float learning_rate = 0.0001f;
    for(size_t i = 0; i < 1000; i++) {
        l2.backprop(x, y, learning_rate);
    }

    // # Measures.

    Matrix<float> y_hat = l2.pass(x);
    Matrix<float> error = (y - y_hat) * (y - y_hat);

    std::cout << "Input: " << std::endl;
    printMatrix(x);
    std::cout << "My guess: " << std::endl;
    printMatrix(y_hat);
    std::cout << "True output: " << std::endl;
    printMatrix(y);
    std::cout << "The weight: " << std::endl;
    printMatrix(l2.weights());
    std::cout << "The bias: " << std::endl;
    printMatrix(l2.biases());
    std::cout << "Cost/error: " << std::endl;
    printMatrix(error);
}

TEST(NeuralNetworkTests, OneLayerMultipleInputsBackprop) {
    
    // # NN initialisation.
    
    std::vector<float> y_data_ = {
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        2.0f,
        2.0f,
        2.0f,
        2.0f,
        2.0f
    };

    Matrix<float> y (10, 1, y_data_);


    std::vector<float> x_data_ = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
        7.0f,
        8.0f,
        9.0f,
        10.0f
    };
    Matrix<float> x (10, 1, x_data_);

    Layer<float> l2 (1, 1);

    printMatrix(y);
    printMatrix(x);

    // # Backprop.

    assert(x.rows() == y.rows());
    float learning_rate = 0.001f;
    int epochs = 100000;

    for(size_t epoch = 0; epoch < epochs; epoch++) {
        for(size_t training_i = 0; training_i < x.rows(); training_i++){

            Matrix<float> x_ = x.getRow(training_i);
            Matrix<float> y_ = y.getRow(training_i);

            l2.backprop(x_, y_, learning_rate);

        }
    }

    // # Measures.

    std::vector<float> y_hat_;
    for(size_t testing_i = 0; testing_i < x.rows(); testing_i++) {
        Matrix<float> x_ = x.getRow(testing_i);
        Matrix<float> y_ = l2.pass(x_);
        y_hat_.push_back(y_(0,0));
    }
    Matrix<float> y_hat (10, 1, y_hat_); 

    std::vector<float> error_data_;
    for(int error_i = 0; error_i < x.rows(); error_i++) {
        float error_ = y(error_i, 0) - y_hat(error_i, 0);
        error_ *= error_;
        error_data_.push_back(error_);
    }
    Matrix<float> error (10, 1, error_data_);

    float cost = 0;
    for(int i = 0; i < x.rows(); i++)
        cost+=error(i,0);
    cost/=x.rows();

    std::cout << "Input: " << std::endl;
    printMatrix(x);
    std::cout << "My guess: " << std::endl;
    printMatrix(y_hat);
    std::cout << "True output: " << std::endl;
    printMatrix(y);
    std::cout << "The weight: " << std::endl;
    printMatrix(l2.weights());
    std::cout << "The bias: " << std::endl;
    printMatrix(l2.biases());
    std::cout << "Loss: " << std::endl;
    printMatrix(error);
    std::cout << "Cost: " << std::endl;
    std::cout << cost << std::endl;
}

TEST(NeuralNetworkTests, TwoLayerMultipleInputsBackprop) {
    // # NN initialisation.

    std::vector<float> y_data_ = {
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        2.0f,
        2.0f,
        2.0f,
        2.0f,
        2.0f
    };

    Matrix<float> y (10, 1, y_data_);


    std::vector<float> x_data_ = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
        7.0f,
        8.0f,
        9.0f,
        10.0f
    };
    Matrix<float> x (10, 1, x_data_);

    Layer<float> l2 (1, 1);
    Layer<float> l3 (1, 1);

    printMatrix(y);
    printMatrix(x);

    // # Backprop.

    assert(x.rows() == y.rows());
    float learning_rate = 0.001f;
    int epochs = 10000;

    for(size_t epoch = 0; epoch < epochs; epoch++) {
        for(size_t training_i = 0; training_i < x.rows(); training_i++){

            Matrix<float> x_ = x.getRow(training_i);
            Matrix<float> y_ = y.getRow(training_i);

            l2.backprop(x_, y_, learning_rate);
            l3.backprop(l2.pass(x_), y_, learning_rate);
        }
    }

    // # Measures.

    std::vector<float> y_hat_;
    for(size_t testing_i = 0; testing_i < x.rows(); testing_i++) {
        Matrix<float> x_ = x.getRow(testing_i);
        Matrix<float> y_ = l3.pass(l2.pass(x_));
        y_hat_.push_back(y_(0,0));
    }
    Matrix<float> y_hat (10, 1, y_hat_); 

    std::vector<float> error_data_;
    for(int error_i = 0; error_i < x.rows(); error_i++) {
        float error_ = y(error_i, 0) - y_hat(error_i, 0);
        error_ *= error_;
        error_data_.push_back(error_);
    }
    Matrix<float> error (10, 1, error_data_);

    float cost = 0;
    for(int i = 0; i < x.rows(); i++)
        cost+=error(i,0);
    cost/=x.rows();

    std::cout << "Input: " << std::endl;
    printMatrix(x);
    std::cout << "My guess: " << std::endl;
    printMatrix(y_hat);
    std::cout << "True output: " << std::endl;
    printMatrix(y);
    std::cout << "The weight: " << std::endl;
    printMatrix(l2.weights());
    std::cout << "The bias: " << std::endl;
    printMatrix(l2.biases());
    std::cout << "Loss: " << std::endl;
    printMatrix(error);
    std::cout << "Cost: " << std::endl;
    std::cout << cost << std::endl;  
}

TEST(NeuralNetworkTests, SimpleModel) {
    std::vector<float> y_data_ = {
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        2.0f,
        2.0f,
        2.0f,
        2.0f,
        2.0f
    };
    std::vector<float> x_data_ = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
        7.0f,
        8.0f,
        9.0f,
        10.0f
    };

    Matrix<float> y (10, 1, y_data_);
    Matrix<float> x (10, 1, x_data_);

    Layer<float> l2 (1,1);
    Layer<float> l3 (1,1);

    Model<float> model;
    model.appendLayer(l2);
    model.appendLayer(l3);


    model.train(x, y, 100UL);



    Matrix x_sample = x.getRow(0);
    x_sample.basicPrint();
    //Matrix y_hat = model.passOne(x_sample);
//
    //std::cout << "y_hat:" << std::endl;
    //y_hat.basicPrint();

}

TEST(NeuralNetworkTests, ModelMultipleNeurons) {
    Matrix<float> x (10, 1, 2.0f);

    Layer<float> l2 (10,1);
    Layer<float> l3 (1,10);

    Model<float> model;
    model.appendLayer(l2);
    model.appendLayer(l3);

    Matrix a = model.pass(x);
    a.basicPrint();
}

TEST(NeuralNetworkTests, ModelMultipleNeuronsTrain) {
    Matrix<float> x (10, 1, 2.0f);
    Matrix<float> y (10, 1, 5.0f);

    Layer<float> l2 (10,1);
    Layer<float> l3 (1,10);

    Model<float> model;
    model.appendLayer(l2);
    model.appendLayer(l3);

    model.train(x, y);

    Matrix a = model.pass(x);
    a.basicPrint();
}

TEST(NeuralNetworkTests, LayersTest) {
    // # Input created.
    Matrix<float> input(1, 1, 5.0f);

    // # Print input.
    std::cout << "Input:" << std::endl;
    input.basicPrint();
    
    // # Layers created.
    Layer<float> l1 (10, 1);
    Layer<float> l2 (1, 10);
    Layer<float> l3 (2, 1);
    
    // # Layer 1 printed.
    std::cout << "Layer 1:" << std::endl;
    l1.debugPrint();

    // # Activation 1 created.
    Matrix a1 = l1.pass(input);

    // # Activation 1 printed.
    std::cout << "a1:" << std::endl;
    a1.basicPrint();

    // # Activation 2 created.
    Matrix a2 = l2.pass(a1);

    // # Activation 2 printed.
    std::cout << "a2:" << std::endl;
    a2.basicPrint();

    // # Activation 3 created.
    Matrix a3 = l3.pass(a2);

    // # Activation 3 printed.
    std::cout << "a3:" << std::endl;
    a3.basicPrint();
}
