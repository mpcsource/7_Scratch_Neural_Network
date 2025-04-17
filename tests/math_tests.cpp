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

