#include "gtest/gtest.h"
#include "network/layer.hpp"
#include "network/model.hpp"
#include "data/loader.hpp"

TEST(NeuralNetwork, ForwardAndBackprop) {
    // Load data from CSV file.
    Matrix data = loadData("../tests/data.csv", ',', true);

    // Split data into training and testing sets.
    auto [x_train, y_train, x_test, y_test] = trainTestSplit(data, 8);

    // Normalize the data.
    std::vector<float> mean, deviation;
    std::tie(x_train, mean, deviation) = normalizeData(x_train);
    x_test = normalizeData(x_test, mean, deviation);

    // Normalize the target variable.
    std::vector<float> mean_y, std_y;
    std::tie(y_train, mean_y, std_y) = normalizeData(y_train);
    y_test = normalizeData(y_test, mean_y, std_y);

    // Create 3 layers.
    Layer l1 (8, 64); // Sigmoid activation by default.
    Layer l2 (64, 64); // Sigmoid activation by default.
    Layer l3 (64, 1, "linear");

    // Create a model with Mean Squared Error loss.
    // Append the layers to the model.
    Model model ("mse");
    model.appendLayer(&l1);
    model.appendLayer(&l2);
    model.appendLayer(&l3);

    // Backpropagate the model.
    model.backprop(x_train, y_train, 30, 0.1f);

    // Unnormalize the data for testing.
    y_test = unnormalizeData(y_test, mean_y, std_y);
    y_train = unnormalizeData(y_train, mean_y, std_y);

    // Print the results.
    std::cout << "True:" << std::endl;
    y_test.head().basicPrint();
    std::cout << "Pred:" << std::endl;
    auto y_hat = model.test(x_test, y_test);
    y_hat = unnormalizeData(y_hat, mean_y, std_y);
    y_hat.head().basicPrint();
}