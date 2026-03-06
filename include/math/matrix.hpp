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
    std::vector<float> h_data; // Internal data
    std::vector<float> h_grad; // Self gradient

    // GPU
    float* d_data = nullptr;
    float* d_grad = nullptr;
    mutable bool cpu_dirty = false; // d_data is newer than h_data
    mutable bool gpu_dirty = true;  // h_data is newer than d_data (always true on construction)

public:

    // ============
    // Constructors
    // ============

    Matrix(); // Empty constructor
    Matrix(int r, int c, float fill = 0); // Fill constructor
    Matrix(int r, int c, std::vector<float> data); // Array constructor
    Matrix(const Matrix& other); // Copy constructor
    Matrix(Matrix&& other) noexcept; // Move constructor
    Matrix& operator=(const Matrix& other); // Copy assignment operator

    ~Matrix();

    int rows() const; // Row amt getter
    int cols() const; // Col amt getter

    void toDevice() const;
    void toHost() const;

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

    Matrix addBias(const Matrix& bias) const; // Add nout×1 bias to each column of nout×N matrix

    Matrix sumCols() const; // Sum across columns, returns rows×1 matrix

    Matrix getRow(int row) const; // Extract row from self

    void basicPrint() const; // Debug print



    // Write access.
    float &operator()(int row, int col);

    // Read access.
    const float &operator()(int row, int col) const;

    Matrix head() const;

    Matrix tail() const;
};