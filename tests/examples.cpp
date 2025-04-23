#include "gtest/gtest.h"
#include "network/layer.h"
#include "network/model.h"
#include "data/loader.h"

TEST(Examples, CaliforniaHousingPrices) {
    // # Load whole data.
    Matrix<double> data = loadData<double>("../tests/data.csv", ',', true);

    // # Split training and testing data.
    auto [x_train, y_train, x_test, y_test] = trainTestSplit<double>(data, 8);

    // # Create layers.
    Layer<double> l1 (10, 8, "glorot");
    Layer<double> l2 (10, 10, "glorot");
    Layer<double> l3 (10, 10, "glorot");
    Layer<double> l4 (1, 10, "glorot");

    // # Create model and append layers.
    Model<double> model;
    model.appendLayer(l1);
    model.appendLayer(l2);
    model.appendLayer(l3);
    model.appendLayer(l4);

    // # Standardize x_train and x_test.
    std::vector<double> mean, deviation;
    std::tie(x_train, mean, deviation) = normalizeData<double>(x_train);
    x_test = normalizeData<double>(x_test, mean, deviation);

    // # Standardize y_train and y_test.
    y_train = y_train * (0.00001);
    y_test = y_test * (0.00001);

    // # Train model.
    model.train(x_train, y_train, 1000, 1.0E-7);

    // # Make prediction.
    Matrix y_hat = model.pass(x_test.head());

    // # Unstandardize Y data.
    y_hat = y_hat * 100000;
    y_train = y_train * 100000;
    y_test = y_test * 100000;

    // # y_hat vs y_true.
    std::cout << "Y_hat:" << std::endl;
    y_hat.basicPrint();
    std::cout << "Y_true:" << std::endl;
    y_test.head().basicPrint();
}