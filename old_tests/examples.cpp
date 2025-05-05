#include "gtest/gtest.h"
#include "network/layer.h"
#include "network/relu_layer.h"
#include "network/model.h"
#include "data/loader.h"

TEST(Examples, CaliforniaHousingPrices) {
    // # Load whole data.
    Matrix<double> data = loadData<double>("../tests/data.csv", ',', true);

    // # Split training and testing data.
    auto [x_train, y_train, x_test, y_test] = trainTestSplit<double>(data, 8);

    // # Create layers.

    ReLU_Layer<double> l1 (50, 8, "glorot");
    ReLU_Layer<double> l2 (30, 50, "glorot");
    ReLU_Layer<double> l3 (8, 30, "glorot");
    Layer<double> l4 (1, 8, "glorot");

    Layer<double> l_test (1, 8, "glorot");

    // # Create model and append layers.
    Model<double> model;
    model.appendLayer(l1);
    //model.appendLayer(l4);
    model.appendLayer(l2);
    model.appendLayer(l3);
    model.appendLayer(l4);

    // # Standardize x_train and x_test.
    std::vector<double> mean, deviation;
    std::tie(x_train, mean, deviation) = normalizeData<double>(x_train);
    x_test = normalizeData<double>(x_test, mean, deviation);


    // # Standardize y_train and y_test.
    //y_train = y_train * (0.00001);
    //y_test = y_test * (0.00001);
    std::vector<double> mean_y, std_y;
    std::tie(y_train, mean_y, std_y) = normalizeData<double>(y_train);
    y_test = normalizeData<double>(y_test, mean_y, std_y);


    // # Train model.
    model.train(x_train, y_train, 1000, 0.001, 64);

    // # Make prediction.
    Matrix y_hat = model.pass(x_test);

    // # Unstandardize Y data.
    //y_hat = y_hat * 100000;
    //y_train = y_train * 100000;
    //y_test = y_test * 100000;

    // # y_hat vs y_true.
    std::cout << "Y_hat:" << std::endl;
    y_hat.head().basicPrint();
    std::cout << "Y_true:" << std::endl;
    y_test.head().basicPrint();

    // # Error.
    float error = 0;
    for(int i = 0; i < y_hat.rows(); i++)
        error += (y_test(i,0) - y_hat(i,0)) * (y_test(i,0) - y_hat(i,0));
    error /= y_hat.rows();

    std::cout << "Error:" << error << std::endl;
}

TEST(Examples, MNISTDigitRecognition) {
    // # Load whole data.
    Matrix<float> data = loadData<float>("../tests/mnist_train.csv", ',', false, true, 10000);

    // # Split training and testing data.
    auto [x_train, y_train, x_test, y_test] = trainTestSplit<float>(data, 0);

    // # Standardize x_train and x_test.
    std::vector<float> mean, deviation;
    std::tie(x_train, mean, deviation) = normalizeData<float>(x_train);
    x_test = normalizeData<float>(x_test, mean, deviation);

    // # Create layers.
    Layer<float> hidden2 (16, 28*28, "glorot");
    Layer<float> hidden3 (16, 16, "glorot");
    Layer<float> hidden4 (1, 16, "glorot");

    // # Create model and append layers.
    Model<float> model;
    model.appendLayer(hidden2);
    model.appendLayer(hidden3);
    model.appendLayer(hidden4);

    // # Train model.
    model.train(x_train, y_train, 100, 1.0E-3);

    // # Make prediction.
    Matrix y_hat = model.pass(x_test.head());

    // # y_hat vs y_true.
    std::cout << "Y_hat:" << std::endl;
    y_hat.basicPrint();
    std::cout << "Y_true:" << std::endl;
    y_test.head().basicPrint();
}

TEST(ReLU, MNISTDigitRecognition) {
    // # Load whole data.
    Matrix<float> data = loadData<float>("../tests/mnist_train.csv", ',', false, true, 10000);

    // # Split training and testing data.
    auto [x_train, y_train, x_test, y_test] = trainTestSplit<float>(data, 0);

    // # Standardize x_train and x_test.
    std::vector<float> mean, deviation;
    std::tie(x_train, mean, deviation) = normalizeData<float>(x_train);
    x_test = normalizeData<float>(x_test, mean, deviation);

    // # Create layers.
    ReLU_Layer<float> hidden2 (16, 28*28, "glorot");
    ReLU_Layer<float> hidden3 (16, 16, "glorot");
    ReLU_Layer<float> hidden4 (1, 16, "glorot");

    // # Create model and append layers.
    Model<float> model;
    model.appendLayer(hidden2);
    model.appendLayer(hidden3);
    model.appendLayer(hidden4);

    // # Train model.
    model.train(x_train, y_train, 1000, 1.0E-5, 100);

    // # Make prediction.
    Matrix y_hat = model.pass(x_test.head());

    // # y_hat vs y_true.
    std::cout << "Y_hat:" << std::endl;
    y_hat.basicPrint();
    std::cout << "Y_true:" << std::endl;
    y_test.head().basicPrint();
}