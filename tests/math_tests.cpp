#include <iostream>
#include "gtest/gtest.h"
#include "math/matrix.h"

void printMatrix(Matrix<int> m) {
    for(size_t i = 0; i < m.rows(); i++) {
        for(size_t j = 0; j < m.cols(); j++)
            std::cout << m(i,j) << " ";
        std::cout << std::endl;
    }
}

TEST(MathTests, Addition1) {
    Matrix<int> m1 (2, 3, 2);
    Matrix<int> m2 (2, 3, 5);  
    Matrix<int> m3 = m1 + m2;
    printMatrix(m3);
}

TEST(MathTests, Addition2) {
    Matrix<int> m1 (4, 4, 9);
    Matrix<int> m2 (4, 4, 0);  
    Matrix<int> m3 = m1 + m2;
    printMatrix(m3);
}

TEST(MathTests, Multiplication1) {
    Matrix<int> m1 (2, 2, 2);
    Matrix<int> m2 (2, 2, 3);  
    Matrix<int> m3 = m1 * m2;
    printMatrix(m3);
}

TEST(MathManualBackprop, OneLayer) {
    float y = 100.0f; // True output.
    
    float x = 5.0f; // Input layer.

    float w2 = 1.0f; // Output layer's weight.
    float b2 = 1.0f; // Output layer's bias.
    
    // # Multiple backprop iteration.
    for(size_t i = 0; i < 100; i++) {
        // Update weight.
        float a2 = x * w2 + b2;
    
        float dz2_dw2 = x;
        float da2_dz2 = w2;
        float dE_da2 = 2 * (a2 - y);
    
        float dE_dw2 = dz2_dw2 * da2_dz2 * dE_da2;
    
        w2 -= 0.0001 * dE_dw2;

        // Update bias.
        float a2_ = x * w2 + b2;
    
        float dz2_db2 = 1;
        float da2_dz2_ = w2;
        float dE_da2_ = 2 * (a2_ - y);
    
        float dE_db2 = dz2_db2 * da2_dz2_ * dE_da2_;
    
        b2 -= 0.0001 * dE_db2;
    }

    // # Measures.

    float y_hat = x * w2 + b2; // NN prediction.
    float error = (y - y_hat) * (y - y_hat); // Error.


    std::cout << "Input: " << x << std::endl;
    std::cout << "My guess: " << y_hat << std::endl;
    std::cout << "True output: " << y << std::endl;
    std::cout << "The weight: " << w2 << std::endl;
    std::cout << "The bias: " << b2 << std::endl;
    std::cout << "Cost/error: " << error << std::endl;
}

TEST(ManualBackprop, OneLayer_NoActivationFunction) {
    float y = 10.0f; // True output.
    
    float x = 5.0f; // Input layer.

    float w2 = 1.0f; // Output layer's weight.
    float b2 = 1.0f; // Output layer's bias.
    
    // # One backprop iteration.
    for(size_t i = 0; i < 100000; i++) {
        // Update weight.
        float a2 = x * w2 + b2;
    
        float dz2_dw2 = x;
        float da2_dz2 = w2;
        float dE_da2 = 2 * (a2 - y);
    
        float dE_dw2 = dz2_dw2 * da2_dz2 * dE_da2;
    
        w2 -= 0.0001 * dE_dw2;

        // Update bias.
        float a2_ = x * w2 + b2;
    
        float dz2_db2 = 1;
        float da2_dz2_ = w2;
        float dE_da2_ = 2 * (a2_ - y);
    
        float dE_db2 = dz2_db2 * da2_dz2_ * dE_da2_;
    
        b2 -= 0.0001 * dE_db2;
    }

    // # Measures.

    float y_hat = x * w2 + b2; // NN prediction.
    float error = (y - y_hat) * (y - y_hat); // Error.


    std::cout << "Input: " << x << std::endl;
    std::cout << "My guess: " << y_hat << std::endl;
    std::cout << "True output: " << y << std::endl;
    std::cout << "The weight: " << w2 << std::endl;
    std::cout << "The bias: " << b2 << std::endl;
    std::cout << "Cost/error: " << error << std::endl;
}