#pragma once
#include <string>
#include <fstream>
#include "math/matrix.h"
#include <math.h>

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>
Matrix<T> loadData(const std::string& path, char separator, bool header, bool limitRows = false, int limitRowsAmount = 10000) {
    
    // # Initialise data matrix.
    std::vector<std::vector<T>> data;

    // # Load file.
    std::ifstream file(path);

    // # Guarantee file loaded successfully.
    if(!file)
        throw std::invalid_argument("Failed to open file.");

    // # To store each line.
    std::string line;

    // # Iterate over lines.
    for(int iteration = 0; std::getline(file, line); iteration++) {
        
        if(limitRows && iteration >= limitRowsAmount)
            break;

        // # Ignore header.
        if(iteration == 0 && header)
            continue;

        std::vector<T> row;
        std::stringstream ss (line);
        std::string cell;

        // # Store each line.
        // # (this really needs to be improved.)
        while(std::getline(ss, cell, separator)) {
            try
            {
                row.push_back((T)std::stof(cell));
            }
            catch(const std::exception& e) {}         
        }

        // # Store row.
        data.push_back(row);
    }

    int rows = data.size();
    int cols = data.at(0).size();
    Matrix<T> data_matrix (rows, cols);

    for(int row = 0; row < rows; row++)
        for(int col = 0; col < cols; col++)
            data_matrix(row, col) = data[row][col];

    return data_matrix;
}

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>> trainTestSplit(Matrix<T> data, int y_col) {
    
    // # Shuffle data.
    for(int i = 0; i < data.rows(); i++) {
        int j = (rand() % data.cols()) + 1;

        Matrix<T> a_i = data.getRow(i);
        Matrix<T> a_j = data.getRow(j);

        // Exchange both.
        for(int k = 0; k < a_j.cols(); k++) 
            data(i, k) = a_j(0, k);
        for(int k = 0; k < a_i.cols(); k++)
            data(j, k) = a_i(0, k);
    }

    size_t const half_size = data.rows() % 2;

    Matrix<T> x_train (data.rows() / 2, data.cols()-1);
    Matrix<T> y_train (data.rows() / 2, 1);
    Matrix<T> x_test  (data.rows() / 2 + half_size, data.cols()-1);
    Matrix<T> y_test  (data.rows() / 2 + half_size, 1);

    

    int train_row = 0;
    int test_row = 0;
    for(int i = 0; i < data.rows(); i++) {
        // # Evenly divide train and testing data.
        if(i % 2 == 0) {
            // # Fill training row.
            y_train(train_row, 0) = data(i, y_col);
            for(int j = 0; j < data.cols(); j++) {
                if(j == y_col)
                    continue; // # Skip y_col.
                x_train(train_row, j) = data(i, j);
            }
            train_row++;
        } else {
            // # Fill testing row.
            y_test(test_row, 0) = data(i, y_col);
            for(int j = 0; j < data.cols(); j++) {
                if(j == y_col)
                    continue; // # Skip y_col.
                x_test(test_row, j) = data(i, j);
            }
            test_row++;
        }
    }

    return { x_train, y_train, x_test, y_test };
}

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>
// # Normalize data and return mean and standard deviation.
inline std::tuple<Matrix<T>, std::vector<T>, std::vector<T>> normalizeData(Matrix<T> data) {
    
    std::vector<T> means (data.cols());
    std::vector<T> stds (data.cols());

    // # Calculate mean per column/feature.
    for(int j = 0; j < data.cols(); j++) {
        T mean = 0;
        for(int i = 0; i < data.rows(); i++)
            mean += data(i,j);
        mean /= data.rows();
        means[j] = mean;
    }
        
    // # Calculate standard deviation per column/feature.
    for(int j = 0; j < data.cols(); j++) {
        T std = 0;
        for(int i = 0; i < data.rows(); i++)
            std += (data(i,j) - means[j]) * (data(i,j) - means[j]);
        std /= data.rows();
        std = sqrt(std);
        if(std < 1e-5)
            std = 1e-5;
        stds[j] = std;
    }

    data = normalizeData(data, means, stds);

    return {data, means, stds};
}

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>
// # Normalize data with given mean and standard deviation.
inline Matrix<T> normalizeData(Matrix<T> data, std::vector<T> means, std::vector<T> stds) {
    
    for(int i = 0; i < data.rows(); i++)
        for(int j = 0; j < data.cols(); j++) {
            if (stds[j] != 0)
                data(i,j) = (data(i,j) - means[j]) / stds[j];
            else
                data(i,j) = 0;
        }
    
    return data;
}

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>
inline Matrix<T> unnormalizeData(Matrix<T> data, float mean, float standard) {
    return data;
}

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>
inline std::tuple<Matrix<T>, Matrix<T>> getBatchOfSize(Matrix<T> X, Matrix<T> Y, int batch_size = 32) {
    assert(X.rows() == Y.rows());
    
    Matrix<T> batch_x (batch_size, X.cols());
    Matrix<T> batch_y (batch_size, Y.cols());

    for(int i = 0; i < batch_size; i++) {
        int random_row_index = rand() % X.rows()+1; 
        auto row_x = X.getRow(random_row_index);
        auto row_y = Y.getRow(random_row_index);
        
        for(int j = 0; j < X.cols(); j++) {
            batch_x(i,j) = row_x(0,j);
        }
        for(int j = 0; j < Y.cols(); j++) {
            batch_y(i,j) = row_y(0,j);
        }
    }

    return {batch_x, batch_y};
}