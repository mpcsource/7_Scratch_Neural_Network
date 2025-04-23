#include "gtest/gtest.h"
#include "data/loader.h"
#include "math/random.h"

TEST(A, B) {
    Matrix<float> data = loadData<float>("../tests/data.csv", ',', true);
    data.basicPrint();

    auto [x_train, y_train, x_test, y_test] = trainTestSplit<float>(data, 8);
}

TEST(RandomTests, Generation) {
    for(int _ = 0; _ < 10; _++) {
        int number = random10Int();
        std::cout << number << std::endl;
    }
}

TEST(RandomTests, Range) {
    for(int i = 0; i < 10; i++)
        std::cout << randomRange(-1.0f, 2.0f) << std::endl;
}

TEST(Data, Normalization) {
    Matrix<float> data = loadData<float>("../tests/data.csv", ',', true);

    auto [x_train, y_train, x_test, y_test] = trainTestSplit<float>(data, 8);
    x_train.head().basicPrint();
    std::vector<float> a, b;
    std::tie(x_train, a, b) = normalizeData<float>(x_train.head());
    x_train.head().basicPrint();
    Matrix<float>(1,8,b).basicPrint();
}