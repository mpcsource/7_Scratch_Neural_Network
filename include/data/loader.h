#pragma once
#include <string>
#include <fstream>
#include "math/matrix.h"

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>
Matrix<T> loadData(const std::string& path, char separator, bool header) {
    
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
        int j = 1; // Should be random.

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