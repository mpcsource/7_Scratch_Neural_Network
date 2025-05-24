#include "gtest/gtest.h"
#include "network/layer.h"
#include "network/model.h"
#include "data/loader.h"

TEST(A, B) {
    Matrix<double> data = loadData<double>("../tests/data.csv", ',', true);

    auto [x_train, y_train, x_test, y_test] = trainTestSplit<double>(data, 8);

    std::vector<double> mean, deviation;
    std::tie(x_train, mean, deviation) = normalizeData<double>(x_train);
    x_test = normalizeData<double>(x_test, mean, deviation);

    std::vector<double> mean_y, std_y;
    std::tie(y_train, mean_y, std_y) = normalizeData<double>(y_train);
    y_test = normalizeData<double>(y_test, mean_y, std_y);

    //Matrix<double> x (8, 1, 1);
    //Matrix<double> y (1, 1, 2);

    Layer<double> l1 (8, 64);
    Layer<double> l2 (64, 64);
    Layer<double> l3 (64, 1, "linear");
    
    Model<double> model ("mse");
    model.appendLayer(&l1);
    model.appendLayer(&l2);
    model.appendLayer(&l3);

    model.backprop(x_train, y_train, 30, 0.01f);

    y_test = unnormalizeData<double>(y_test, mean_y, std_y);
    y_train = unnormalizeData<double>(y_train, mean_y, std_y);

    std::cout << "True:" << std::endl;
    y_test.head().basicPrint();
    std::cout << "Pred:" << std::endl;
    auto y_hat = model.test(x_test, y_test);
    y_hat = unnormalizeData<double>(y_hat, mean_y, std_y);
    y_hat.head().basicPrint();
}