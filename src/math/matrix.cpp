#include "math/matrix.hpp"


Matrix::Matrix() : rows_(0), cols_(0), data_() {}

Matrix::Matrix(int r, int c, float fill) : rows_(r), cols_(c), data_(r * c, fill) {}

Matrix::Matrix(int r, int c, std::vector<float> data) : rows_(r), cols_(c), data_(std::move(data)) {}

int Matrix::rows() const {
    return this->rows_;
}

int Matrix::cols() const {
    return this->cols_;
}

Matrix Matrix::add(const Matrix& other) const {
    assert(this->rows() == other.rows());
    assert(this->cols() == other.cols());

    Matrix out(this->rows(), this->cols(), 0);

    for (int i = 0; i < this->rows(); i++)
        for (int j = 0; j < this->cols(); j++)
            out(i, j) = (*this)(i, j) + other(i, j);

    return out;
}

Matrix Matrix::subtract(const Matrix& other) const {
    assert(this->rows() == other.rows());
    assert(this->cols() == other.cols());

    Matrix out(this->rows(), this->cols(), 0);

    for (int i = 0; i < this->rows(); i++)
        for (int j = 0; j < this->cols(); j++)
            out(i, j) = (*this)(i, j) - other(i, j);

    return out;
}

Matrix Matrix::multiply(const Matrix& other) const {
    assert(this->rows() == other.rows());
    assert(this->cols() == other.cols());

    Matrix out(this->rows(), this->cols(), 0);

    for (int i = 0; i < this->rows(); i++)
        for (int j = 0; j < this->cols(); j++)
            out(i, j) = (*this)(i, j) * other(i, j);

    return out;
}

Matrix Matrix::multiply(float other) const {
    // Create output matrix.
    Matrix out(this->rows(), this->cols(), 0);

    // Perform multiplication.
    for (int i = 0; i < this->rows(); i++)
        for (int j = 0; j < this->cols(); j++)
            out(i, j) = (*this)(i, j) * other;

    // Return result.
    return out;
}

Matrix Matrix::dot(const Matrix& other) const {
    assert(this->cols() == other.rows());

    Matrix out(this->rows(), other.cols(), 0);

    for (int i = 0; i < this->rows(); i++)
        for (int j = 0; j < other.cols(); j++) {
            float value = 0;
            for (int k = 0; k < this->cols(); k++)
                value += (*this)(i, k) * other(k, j);
            out(i, j) = value;
        }

    return out;
}

Matrix Matrix::transpose() const {
    Matrix transposed(this->cols_, this->rows_, 0);

    for (size_t i = 0; i < this->rows_; i++)
        for (size_t j = 0; j < this->cols_; j++)
            transposed(j, i) = (*this)(i, j);

    return transposed;
}

Matrix Matrix::exponential() const {
    Matrix result(this->rows_, this->cols_, 0);
    for (int i = 0; i < this->rows_; i++)
        for (int j = 0; j < this->cols_; j++)
            result(i, j) = std::exp((*this)(i, j));
    return result;
}

Matrix Matrix::apply(std::function<float(float)> func) const {
    Matrix out(this->rows(), this->cols());
    for (int i = 0; i < this->rows(); i++)
        for (int j = 0; j < this->cols(); j++)
            out(i, j) = func((*this)(i, j));
    return out;
}

Matrix Matrix::getRow(int row) const {
    Matrix result(1, this->cols_, 0);
    for (int i = 0; i < this->cols_; i++) {
        result(0, i) = (*this)(row, i);
    }
    return result;
}

void Matrix::basicPrint() const {
    for (size_t i = 0; i < this->rows_; i++) {
        for (size_t j = 0; j < this->cols_; j++)
            std::cout << (*this)(i, j) << " ";
        std::cout << std::endl;
    }
}

float &Matrix::operator()(int row, int col){
    return this->data_[row * this->cols_ + col];
}

const float &Matrix::operator()(int row, int col) const {
    return this->data_[row * this->cols_ + col];
}

Matrix Matrix::head() const {
    Matrix result(5, this->cols_);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < this->cols_; j++)
            result(i, j) = (*this)(i, j);
    return result;
}

Matrix Matrix::tail() const {
    Matrix result(5, this->cols_);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < this->cols_; j++)
            result(i, j) = (*this)(i + this->rows_ - 5, j);
    return result;
}