#include "gtest/gtest.h"
#include "network/layer.hpp"
#include "network/model.hpp"
#include "data/loader.hpp"

TEST(NeuralNetwork, BasicTest) {
    // Load data from CSV file.
    Matrix data = loadData("../tests/california.csv", ',', true);

    // Split data into training and testing sets.
    auto [x_train, y_train, x_test, y_test] = trainTestSplit(data, 8);

    x_train.head().basicPrint();
    y_train.head().basicPrint();

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
    model.backprop(x_train, y_train, 1, 0.001f);

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
    l1.debugPrint();
}

TEST(ActivationFunctions, Softmax) {
    // Sample data for testing softmax activation.
    Matrix x(3, 1, {1.0f, 2.0f, 3.0f});
    Matrix y(3, 1, {0.0f, 0.0f, 1.0f});

    y = oneHotEncode(y);

    x.basicPrint();
    y.basicPrint();

    // Create 1 layer and model.
    Layer l1(1, 2, "softmax");
    Model model("cross_entropy");
    model.appendLayer(&l1);

    model.backprop(x, y, 100, 0.1f);

    // Print the results.
    std::cout << "True:" << std::endl;
    y.basicPrint();
    std::cout << "Pred:" << std::endl;
    auto y_hat = model.test(x, y);
    y_hat.basicPrint();
    // Print the weights and biases of the layer.
    std::cout << "Weights and biases of the layer:" << std::endl;
    l1.debugPrint();

}

TEST(ActivationFunctions, SoftmaxXOR) {
    Matrix x(4,2, {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    });

    Matrix y(4,1, {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    });

    y = oneHotEncode(y);

    Model model("cross_entropy");
    Layer l1(2, 4, "sigmoid");
    Layer l2(4, 2, "softmax");
    model.appendLayer(&l1);
    model.appendLayer(&l2);

    model.backprop(x, y, 100000, 0.1f);

    // Print the results.
    std::cout << "True:" << std::endl;
    argMax(y).basicPrint();

    std::cout << "Pred:" << std::endl;
    auto y_hat = model.test(x, y);
    argMax(y_hat).basicPrint();

    // Print the weights and biases of the layer.
    std::cout << "Weights and biases of the layer:" << std::endl;
    l1.debugPrint();
}

TEST(ActivationFunctions, SoftmaxMNIST) {
    // Load data from CSV file.
    auto data = loadData("../tests/mnist_train.csv", ',', true);

    // Split data into training and testing sets.
    auto [x_train, y_train, x_test, y_test] = trainTestSplit(data, 0);

    // One-hot encode the labels.
    y_train = oneHotEncode(y_train);
    y_test = oneHotEncode(y_test);

    // Print "finished loading data"
    std::cout << "Finished loading data." << std::endl;

    // Create 2 layers.
    Layer l1(28*28, 128, "sigmoid");
    Layer l2(128, 10, "softmax");

    // Create a model with Cross Entropy loss.
    Model model("cross_entropy");
    model.appendLayer(&l1);
    model.appendLayer(&l2);

    // Backpropagate the model.
    model.backprop(x_train, y_train, 10, 0.01f);

    // Get results.
    auto y_sample = y_test.head();
    auto y_hat = model.test(x_test, y_test);
    auto y_hat_sample = y_hat.head();

    // Print sample of the true and predicted labels.
    std::cout << "Sample of the true and predicted labels:" << std::endl;
    y_sample.basicPrint();
    y_hat_sample.basicPrint();

    // Find the index of the maximum value in each row.
    y_sample = argMax(y_sample);
    y_hat_sample = argMax(y_hat_sample);

    // Reverse the one-hot encoding to get the original labels.
    y_test = argMax(y_test);
    y_hat = argMax(y_hat);

    // Accuracy calculation.
    auto total = y_test.rows();
    auto correct = 0;
    for (int i = 0; i < total; i++) {
        if (y_test(i, 0) == y_hat(i, 0)) {
            correct++;
            std::cout << y_test(i, 0) << " == " << y_hat(i, 0) << std::endl;
        }
    }
    auto accuracy = static_cast<float>(correct) / total * 100.0f;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    std::cout << "Got " << correct << " out of " << total << " correct." << std::endl;


    // Print the results.
    std::cout << "True:" << std::endl;
    y_sample.head().basicPrint();
    std::cout << "Pred:" << std::endl;
    y_hat_sample.head().basicPrint();
    //std::cout << "Weights and biases of the layers:" << std::endl;
    //l1.debugPrint();
    //l2.debugPrint();
}