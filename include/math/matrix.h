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
    typename = std::enable_if_t<std::is_arithmetic<T>::value>
>

class Matrix {
    private:
        int rows_, cols_; // Rows and columns of matrix.
        std::vector<T> data_; // Data vector.
        std::vector<Matrix<T>*> children_; // The matrices that made this matrix.
        int op_; // Operation that made this matrix: 
        
        Matrix<T> * other_; // The other matrix in operation with this matrix.
        Matrix<T> * out_; // The result of the operation.
        
    public:
        std::vector<T> grad_; // Gradient of matrix.

        // Empty constructor.
        Matrix() : rows_(1), cols_(1), data_(1, 0), grad_(1, 0), children_(), op_(0) {}

        // Constructor with fill.
        Matrix(int r, int c, T fill = 0) : rows_(r), cols_(c), data_(r*c, fill), grad_(r*c, 0), children_(), op_(0) {}

        // # Constructor with data.
        Matrix(int r, int c, std::vector<T> data) : rows_(r), cols_(c), data_(data), grad_(r*c, 0), children_(), op_(0) {}

        // # Get # of rows/cols.
        int rows() const { return this->rows_; }
        int cols() const { return this->cols_; }

        void backward() {
            switch (this->op_)
            {
            case 1:
                for(int i = 0; i < this->grad_.size(); i++)
                    this->grad_[i] += this->out_->grad_[i];
                for(int i = 0; i < this->other_->grad_.size(); i++)
                    this->other_->grad_[i] += this->out_->grad_[i];
                
                break;
            
            case 2:
                for(int i = 0; i < this->grad_.size(); i++)
                    this->grad_[i] += this->other_->data_[i] * this->out_->grad_[i];
                for(int i = 0; i < this->other_->grad_.size(); i++)
                    this->other_->grad_[i] += this->data_[i] * this->out_->grad_[i];
                    
                break;

            case 3: {
                int M = this->rows_;        // A.rows
                int N = this->cols_;        // A.cols = B.rows
                int P = other_->cols_;      // B.cols
            
                // grad wrt A: A_grad[i,k] += sum_j( C_grad[i,j] * B[k,j] )
                for(int i = 0; i < M; ++i) {
                    for(int k = 0; k < N; ++k) {
                        T acc = 0;
                        for(int j = 0; j < P; ++j) {
                            // index of C[i,j] in flat vector is i*P + j
                            // index of B[k,j] is k*P + j
                            acc += out_->grad_[ i*P + j ] 
                                    * other_->data_[ k*P + j ];
                        }
                        grad_[ i*N + k ] += acc;  // index of A[i,k] is i*N + k
                    }
                }
            
                // grad wrt B: B_grad[k,j] += sum_i( A[i,k] * C_grad[i,j] )
                for(int k = 0; k < N; ++k) {
                    for(int j = 0; j < P; ++j) {
                        T acc = 0;
                        for(int i = 0; i < M; ++i) {
                            // index of A[i,k] is i*N + k
                            // index of C[i,j] is i*P + j
                            acc += this->data_[ i*N + k ]
                                    * out_->grad_[ i*P + j ];
                        }
                        other_->grad_[ k*P + j ] += acc;
                    }
                }
                break;
            }
                

            default:
                break;
            }
        }

        // 
        // Matrix mathematical operations.
        //

        Matrix<T> add(Matrix<T>& other) {

            // Check for equal dimensions.
            assert(this->rows() == other.rows());
            assert(this->cols() == other.cols());

            // Create output matrix.
            Matrix<T> out (this->rows(), this->cols(), 0);
            out.children_ = {this, &other};
            out.op_ = 1;

            // Save other matrix and output by reference.
            this->other_ = &other;
            this->out_ = &out;

            // Perform addition.
            for(int i = 0; i < this->rows(); i++)
                for(int j = 0; j < this->cols(); j++)
                    out(i,j) = (*this)(i,j) + other(i,j);

            // Return result.
            return out;
        }

        Matrix<T> multiply(Matrix<T>& other) {

            // Check for equal dimensions.
            assert(this->rows() == other.rows());
            assert(this->cols() == other.cols());

            // Create output matrix.
            Matrix<T> out (this->rows(), this->cols(), 0);
            out.children_ = {this, &other};
            out.op_ = 2;

            // Save other matrix and output by reference.
            this->other_ = &other;
            this->out_ = &out;

            // Perform multiplication.
            for(int i = 0; i < this->rows(); i++)
                for(int j = 0; j < this->cols(); j++)
                    out(i,j) = (*this)(i,j) * other(i,j);

            // Return result.
            return out;
        }

        Matrix<T> dot(Matrix<T>& other) {

            // Check this cols are equal to other's rows.
            assert(this->cols() == other.rows());

            // Create output matrix.
            Matrix<T> out (this->rows(), other.cols(), 0);
            out.children_ = {this, &other};
            out.op_ = 3;

            // Save other matrix and output by reference.
            this->other_ = &other;
            this->out_ = &out;

            // Perform dot product.
            for(int i = 0; i < this->rows(); i++)
                for(int j = 0; j < other.cols(); j++) {
                    T value = 0;
                    for(int k = 0; k < this->cols(); k++)
                        value += (*this)(i,k) * other(k,j);
                    out(i,j) = value;
                }

            // Return result.
            return out;
        }

        // # Find transposed matrix.
        Matrix<T> transpose() {
            Matrix<T> transposed (this->cols_, this->rows_, 0);

            for(size_t i = 0; i < this->rows_; i++)
                for(size_t j = 0; j < this->cols_; j++)
                    transposed(j,i) = (*this)(i,j);

            return transposed;
        }

        // # Get n-th row.
        Matrix<T> getRow(int row) {
            Matrix<T> result (1, this->cols_, 0);
            for(int i = 0; i < this->cols_; i++) {
                result(0, i) = (*this)(row, i);
            }
            return result;
        }

        // # Basic print, should be updated later.
        void basicPrint() {
            for(size_t i = 0; i < this->rows_; i++) {
                for(size_t j = 0; j < this->cols_; j++)
                    std::cout << (*this)(i,j) << " ";
                std::cout << std::endl;
            }
        }

        // # Read-write access. [int parameters]
        T& operator() (int row, int col) {
            return this->data_[row * this->cols_ + col];
        }

        // # Read-write access. [size_t parameters]
        T& operator() (size_t row, size_t col) {
            return this->data_[(int)row * this->cols_ + (int)col];
        }

        // # Read-only access. [for const's]
        const T& operator() (int row, int col) const {
            return this->data_[row * this->cols_ + col];
        }

        // # Read-only access. [for const's]
        const T& operator() (size_t row, size_t col) const {
            return this->data_[(int)row * this->cols_ + (int)col];
        }

        // # Matrix<T>-Matrix<T> multiplication.
        Matrix<T> operator* (const Matrix<T>& other) const {

            // # Guarantee valid dimensions for matrix-matrix multiplication.
            assert(this->cols_ == other.rows_);

            // # Create matrix to store result.
            Matrix<T> result (this->rows_, other.cols_, 0);

            // # Matrix-matrix multiplication algorithm.
            //#pragma omp parallel for
            for(size_t i = 0; i < this->rows_; i++) {
                for(size_t j = 0; j < other.cols(); j++) {
                    T value = 0;
                    for(size_t k = 0; k < this->cols_; k++) {
                        value += (*this)(i,k) * other(k,j);
                    }
                    result(i,j) = value;
                }
            }

            // # Return result.
            return result;
        }

        // # Matrix<T>-Matrix<T> addition.
        Matrix<T> operator+ (const Matrix<T>& other) const {

            // # Guarantee equal dimensions for matrix-matrix addition.
            assert(this->rows_ == other.rows());
            assert(this->cols_ == other.cols());

            // # Create matrix to store result.
            Matrix<T> result (this->rows_, this->cols_, 0);

            // # Matrix-matrix addition algorithm.
            //#pragma omp parallel for
            for(size_t i = 0; i < this->rows_; i++)
                for(size_t j = 0; j < this->cols_; j++)
                    result(i,j) = (*this)(i,j) + other(i,j);

            // # Return result.
            return result;
        }
        
        

        // # Matrix<T>-Matrix<T> subtraction.
        Matrix<T> operator- (const Matrix<T>& other) const {
            
            // # Guarantee equal dimensions for matrix-matrix subtraction.
            assert(this->rows_ == other.rows());
            assert(this->cols_ == other.cols());

            // # Create matrix to store result.
            Matrix<T> result (this->rows_, this->cols_, 0);

            // # Matrix-matrix subtraction algorithm.
            for(size_t i = 0; i < this->rows_; i++)
                for(size_t j = 0; j < this->cols_; j++)
                    result(i,j) = (*this)(i,j) - other(i,j);

            // # Return result.
            return result;
        }

        Matrix<T> operator* (const T& other) const {
            // # Create matrix to store result.
            Matrix<T> result (this->rows_, this->cols_, 0);

            // # Matrix-float multiplication algorithm.
            for(size_t i = 0; i < this->rows_; i++)
                for(size_t j = 0; j < this->cols_; j++)
                    result(i,j) = (*this)(i,j) * other;

            // # Return result.
            return result;
        }
        
        Matrix<T> head() {
            Matrix<T> result(5, this->cols_);
            for(int i = 0; i < 5; i++)
                for(int j = 0; j < this->cols_; j++)
                    result(i,j) = (*this)(i,j);
            return result;
        }

        Matrix<T> tail() {
            Matrix<T> result(5, this->cols_);
            for(int i = 0; i < 5; i++)
                for(int j = 0; j < this->cols_; j++)
                    result(i,j) = (*this)(i+this->rows_-5,j);
            return result;
        }
};