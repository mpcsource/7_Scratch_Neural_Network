/**
 * File: matrix.h
 * Author: Miguel Certã
 * Date: 17/4/2025
 * Description: This file is the header and contains the definitions for the Matrix class.
 * Dependencies: vector, type_traits
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
    std::vector<float> data_;
    std::vector<float> grad_;
    int rows_, cols_;

public:
    Matrix();

    Matrix(int r, int c, float fill = 0);

    Matrix(int r, int c, std::vector<float> data);

    int rows() const;
    int cols() const;

    Matrix add(const Matrix& other) const;

    Matrix subtract(const Matrix& other) const;

    Matrix multiply(const Matrix& other) const;

    Matrix multiply(float other) const;

    Matrix dot(const Matrix& other) const;

    Matrix transpose() const;

    Matrix exponential() const;

    Matrix apply(std::function<float(float)> func) const;

    Matrix getRow(int row) const;

    void basicPrint() const;

    // Write access.
    float &operator()(int row, int col);

    // Read access.
    const float &operator()(int row, int col) const;

    Matrix head() const;

    Matrix tail() const;
};