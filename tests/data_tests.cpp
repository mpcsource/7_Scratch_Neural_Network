#include "gtest/gtest.h"
#include "data/loader.h"

TEST(A, B) {
    Matrix<float> data = loadData<float>("../tests/data.csv", ',', true);
    data.basicPrint();

    auto [x_train, y_train, x_test, y_test] = trainTestSplit<float>(data, 8);
}