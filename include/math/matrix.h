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
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>

class Matrix {
    private:
        std::vector<T> data_;
        int rows_, cols_;

    public:
        // # Constructor.
        Matrix(int r, int c, T fill = 0) : rows_(r), cols_(c), data_(r*c, fill) {}

        // # Get matrix rows or cols.
        int rows() const { return this->rows_; }
        int cols() const { return this->cols_; }

        // # Find transposed matrix.
        Matrix<T> transpose() {
            Matrix<T> transposed (this->cols_, this->rows_, 0);

            for(size_t i = 0; i < this->rows_; i++)
                for(size_t j = 0; j < this->cols_; j++)
                    transposed(j,i) = (*this)(i,j);

            return transposed;
        }

        // # Read-write access. [int parameters]
        T& operator() (int row, int col) {
            return this->data_[row * this->cols_ + col];
        }

        // # Read-write access. [size_t parameters]
        T& operator() (size_t row, size_t col) {
            return this->data_[row * this->cols_ + col];
        }

        // # Read-only access. [for const's]
        const T& operator() (size_t row, size_t col) const {
            return this->data_[row * this->cols_ + col];
        }

        // # Matrix<T>-Matrix<T> multiplication.
        Matrix<T> operator* (const Matrix<T>& other) const {

            // # Guarantee valid dimensions for matrix-matrix multiplication.
            assert(this->cols_ == other.rows_);

            // # Create matrix to store result.
            Matrix<T> result (this->rows_, other.cols_, 0);

            // # Matrix-matrix multiplication algorithm.
            for(size_t i = 0; i < this->rows_; i++) {
                for(size_t j = 0; j < other.cols_; j++) {
                    T value = static_cast<T>(0);
                    for(size_t k = 0; k < this->cols_; k++) {
                        value += (*this)(i,k) * other(k,j);
                    }
                    result(i,j) = value;
                }
            }

            // # Return result.
            return result;
        }

        Matrix<T> operator+ (const Matrix<T>& other) const {
            
            // # Guarantee equal dimensions for matrix-matrix addition.
            assert(this->rows_ == other.rows_);
            assert(this->cols_ == other.cols_);

            // # Create matrix to store result.
            Matrix<T> result (this->rows_, this->cols_, 0);

            // # Matrix-matrix addition algorithm.
            for(size_t i = 0; i < this->rows_; i++)
                for(size_t j = 0; j < this->cols_; j++)
                    result(i,j) = (*this)(i,j) + other(i,j);

            // # Return result.
            return result;
        }

        Matrix<T> operator- (const Matrix<T>& other) const {
            
            // # Guarantee equal dimensions for matrix-matrix subtraction.
            assert(this->rows_ == other.rows_);
            assert(this->cols_ == other.cols_);

            // # Create matrix to store result.
            Matrix<T> result (this->rows_, this->cols_, 0);

            // # Matrix-matrix subtraction algorithm.
            for(size_t i = 0; i < this->rows_; i++)
                for(size_t j = 0; j < this->cols_; j++)
                    result(i,j) = (*this)(i,j) - other(i,j);

            // # Return result.
            return result;
        }
        
};