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

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = std::enable_if_t<std::is_arithmetic<T>::value>>

class Matrix
{
private:
    std::vector<T> data_;
    std::vector<T> grad_;
    int rows_, cols_;

public:
    Matrix() : rows_(0), cols_(0), data_() {}

    // # Constructor.
    Matrix(int r, int c, T fill = 0) : rows_(r), cols_(c), data_(r * c, fill) {}

    // # Constructor receiving data.
    Matrix(int r, int c, std::vector<T> data) : rows_(r), cols_(c), data_(data) {}

    // # Get matrix rows or cols.
    int rows() const { return this->rows_; }
    int cols() const { return this->cols_; }

    //
    // Matrix mathematical operations.
    //

    Matrix<T> add(Matrix<T> other)
    {

        // Check for equal dimensions.
        assert(this->rows() == other.rows());
        assert(this->cols() == other.cols());

        // Create output matrix.
        Matrix<T> out(this->rows(), this->cols(), 0);

        // Perform addition.
        for (int i = 0; i < this->rows(); i++)
            for (int j = 0; j < this->cols(); j++)
                out(i, j) = (*this)(i, j) + other(i, j);

        // Return result.
        return out;
    }

    Matrix<T> subtract(Matrix<T> other)
    {

        // Check for equal dimensions.
        assert(this->rows() == other.rows());
        assert(this->cols() == other.cols());

        // Create output matrix.
        Matrix<T> out(this->rows(), this->cols(), 0);

        // Perform subtraction.
        for (int i = 0; i < this->rows(); i++)
            for (int j = 0; j < this->cols(); j++)
                out(i, j) = (*this)(i, j) - other(i, j);

        // Return result.
        return out;
    }

    Matrix<T> multiply(Matrix<T> other)
    {

        // Check for equal dimensions.
        assert(this->rows() == other.rows());
        assert(this->cols() == other.cols());

        // Create output matrix.
        Matrix<T> out(this->rows(), this->cols(), 0);

        // Perform multiplication.
        for (int i = 0; i < this->rows(); i++)
            for (int j = 0; j < this->cols(); j++)
                out(i, j) = (*this)(i, j) * other(i, j);

        // Return result.
        return out;
    }

    Matrix<T> multiply(float other)
    {

        // Create output matrix.
        Matrix<T> out(this->rows(), this->cols(), 0);

        // Perform multiplication.
        for (int i = 0; i < this->rows(); i++)
            for (int j = 0; j < this->cols(); j++)
                out(i, j) = (*this)(i, j) * other;

        // Return result.
        return out;
    }

    Matrix<T> dot(Matrix<T> other)
    {

        // Check this cols are equal to other's rows.
        assert(this->cols() == other.rows());

        // Create output matrix.
        Matrix<T> out(this->rows(), other.cols(), 0);

        // Perform dot product.
        for (int i = 0; i < this->rows(); i++)
            for (int j = 0; j < other.cols(); j++)
            {
                T value = 0;
                for (int k = 0; k < this->cols(); k++)
                    value += (*this)(i, k) * other(k, j);
                out(i, j) = value;
            }

        // Return result.
        return out;
    }

    // # Find transposed matrix.
    Matrix<T> transpose()
    {
        Matrix<T> transposed(this->cols_, this->rows_, 0);

        for (size_t i = 0; i < this->rows_; i++)
            for (size_t j = 0; j < this->cols_; j++)
                transposed(j, i) = (*this)(i, j);

        return transposed;
    }

    Matrix<T> exponential()
    {
        Matrix<T> result(this->rows_, this->cols_, 0);
        for (int i = 0; i < this->rows_; i++)
            for (int j = 0; j < this->cols_; j++)
                result(i, j) = std::exp((*this)(i, j));
        return result;
    }

    Matrix<T> apply(std::function<T(T)> func) const
    {
        Matrix<T> out(this->rows(), this->cols());
        for (int i = 0; i < this->rows(); i++)
            for (int j = 0; j < this->cols(); j++)
                out(i, j) = func((*this)(i, j));
        return out;
    }

    // Everything below is probably useless.

    // # Get n-th row.
    Matrix<T> getRow(int row)
    {
        Matrix<T> result(1, this->cols_, 0);
        for (int i = 0; i < this->cols_; i++)
        {
            result(0, i) = (*this)(row, i);
        }
        return result;
    }

    // # Basic print, should be updated later.
    void basicPrint()
    {
        for (size_t i = 0; i < this->rows_; i++)
        {
            for (size_t j = 0; j < this->cols_; j++)
                std::cout << (*this)(i, j) << " ";
            std::cout << std::endl;
        }
    }

    // # Read-write access. [int parameters]
    T &operator()(int row, int col)
    {
        return this->data_[row * this->cols_ + col];
    }

    // # Read-write access. [size_t parameters]
    T &operator()(size_t row, size_t col)
    {
        return this->data_[(int)row * this->cols_ + (int)col];
    }

    // # Read-only access. [for const's]
    const T &operator()(int row, int col) const
    {
        return this->data_[row * this->cols_ + col];
    }

    // # Read-only access. [for const's]
    const T &operator()(size_t row, size_t col) const
    {
        return this->data_[(int)row * this->cols_ + (int)col];
    }

    Matrix<T> head()
    {
        Matrix<T> result(5, this->cols_);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < this->cols_; j++)
                result(i, j) = (*this)(i, j);
        return result;
    }

    Matrix<T> tail()
    {
        Matrix<T> result(5, this->cols_);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < this->cols_; j++)
                result(i, j) = (*this)(i + this->rows_ - 5, j);
        return result;
    }
};