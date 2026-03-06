/**
 * File: matrix.hpp
 * Author: Miguel Certã
 * Date: 17/4/2025
 * Description: This file is the header and contains the definitions for the Matrix class.
 * Dependencies: [TO ADD]
 */

#pragma once
#include <vector>
#include <type_traits>
#include <iostream>
#include <math.h>
#include <functional>
#include <cassert>



class Matrix
{
private:
    int rows_, cols_; // Amount of rows and columns

    // CPU
    std::vector<float> data_; // Internal data
    std::vector<float> grad_; // Self gradient

    // GPU
    float* d_data = nullptr;
    float* d_grad = nullptr;

public:

    // ============
    // Constructors
    // ============

    Matrix(); // Empty constructor
    Matrix(int r, int c, float fill = 0); // Fill constructor
    Matrix(int r, int c, std::vector<float> data); // Array constructor

    int rows() const; // Row amt getter
    int cols() const; // Col amt getter

    // ===============
    // Math operations
    // ===============

    Matrix add(const Matrix& other) const; // Add two matrices

    Matrix subtract(const Matrix& other) const; // Subtract two matrices

    Matrix multiply(const Matrix& other) const; // Element-wise multiplication

    Matrix multiply(float other) const; // Element-wise multiplication with float

    Matrix dot(const Matrix& other) const; // Dot product

    Matrix transpose() const; // Transpose self

    Matrix exponential() const; // Don't remember

    Matrix apply(std::function<float(float)> func) const; // Apply function to this matrix

    Matrix getRow(int row) const; // Extract row from self

    void basicPrint() const; // Debug print

    // Write access.
    float &operator()(int row, int col);

    // Read access.
    const float &operator()(int row, int col) const;

    Matrix head() const;

    Matrix tail() const;
};