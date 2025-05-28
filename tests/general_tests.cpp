#include "gtest/gtest.h"
#include "network/layer.hpp"
#include "network/model.hpp"
#include "data/loader.hpp"

TEST(A, B) {
    Matrix data = loadData("../tests/data.csv", ',', true);

    auto [x_train, y_train, x_test, y_test] = trainTestSplit(data, 8);

    std::vector<float> mean, deviation;
    std::tie(x_train, mean, deviation) = normalizeData(x_train);
    x_test = normalizeData(x_test, mean, deviation);

    std::vector<float> mean_y, std_y;
    std::tie(y_train, mean_y, std_y) = normalizeData(y_train);
    y_test = normalizeData(y_test, mean_y, std_y);

    Layer l1 (8, 64);
    Layer l2 (64, 64);
    Layer l3 (64, 1, "linear");

    Model model ("mse");
    model.appendLayer(&l1);
    model.appendLayer(&l2);
    model.appendLayer(&l3);

    model.backprop(x_train, y_train, 30, 0.1f);

    y_test = unnormalizeData(y_test, mean_y, std_y);
    y_train = unnormalizeData(y_train, mean_y, std_y);

    std::cout << "True:" << std::endl;
    y_test.head().basicPrint();
    std::cout << "Pred:" << std::endl;
    auto y_hat = model.test(x_test, y_test);
    y_hat = unnormalizeData(y_hat, mean_y, std_y);
    y_hat.head().basicPrint();
}