#include "data/loader.hpp"

Matrix loadData(const std::string& path, char separator, bool header, bool limitRows, int limitRowsAmount) {
    
    // # Initialise data matrix.
    std::vector<std::vector<float>> data;

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

        std::vector<float> row;
        std::stringstream ss (line);
        std::string cell;

        // # Store each line.
        // # (this really needs to be improved.)
        while(std::getline(ss, cell, separator)) {
            try
            {
                row.push_back(std::stof(cell));
            }
            catch(const std::exception& e) {}         
        }

        // # Store row.
        data.push_back(row);
    }

    int rows = data.size();
    int cols = data.at(0).size();
    Matrix data_matrix (rows, cols);

    for(int row = 0; row < rows; row++)
        for(int col = 0; col < cols; col++)
            data_matrix(row, col) = data[row][col];

    return data_matrix;
}

std::tuple<Matrix, Matrix, Matrix, Matrix> trainTestSplit(Matrix data, int y_col) {

    // # Shuffle data.
    /*for(int i = 0; i < data.rows(); i++) {
        int j = (rand() % data.rows());

        if(i == j) continue;

        Matrix a_i = data.getRow(i);
        Matrix a_j = data.getRow(j);

        // Exchange both.
        for(int k = 0; k < a_j.cols(); k++) 
            data(i, k) = a_j(0, k);
        for(int k = 0; k < a_i.cols(); k++)
            data(j, k) = a_i(0, k);
    }
    */
    for(int i = data.rows() -1; i > 0; i--) {
        int j = rand() % (i + 1);

        if(i == j) continue;

        auto row_i = data.getRow(i);
        auto row_j = data.getRow(j);

        // # Swap rows i and j.
        for(int k = 0; k < data.cols(); k++) {
            data(i, k) = row_j(0, k);
            data(j, k) = row_i(0, k);
        }
    }

    size_t const half_size = data.rows() % 2;

    Matrix x_train (data.rows() / 2, data.cols()-1);
    Matrix y_train (data.rows() / 2, 1);
    Matrix x_test  (data.rows() / 2 + half_size, data.cols()-1);
    Matrix y_test  (data.rows() / 2 + half_size, 1);

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

Matrix normalizeData(Matrix data, std::vector<float> means, std::vector<float> stds) {

    for(int i = 0; i < data.rows(); i++)
        for(int j = 0; j < data.cols(); j++) {
            if (stds[j] != 0)
                data(i,j) = (data(i,j) - means[j]) / stds[j];
            else
                data(i,j) = 0;
        }
    
    return data;
}

std::tuple<Matrix, std::vector<float>, std::vector<float>> normalizeData(Matrix data) {

    std::vector<float> means (data.cols());
    std::vector<float> stds (data.cols());

    // # Calculate mean per column/feature.
    for(int j = 0; j < data.cols(); j++) {
        float mean = 0;
        for(int i = 0; i < data.rows(); i++)
            mean += data(i,j);
        mean /= data.rows();
        means[j] = mean;
    }
        
    // # Calculate standard deviation per column/feature.
    for(int j = 0; j < data.cols(); j++) {
        float std = 0;
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

Matrix unnormalizeData(Matrix data, std::vector<float> means, std::vector<float> stds) {
    for(int i = 0; i < data.rows(); i++)
        for(int j = 0; j < data.cols(); j++)
            data(i, j) = data(i, j) * stds[j] + means[j];
    return data;
}

std::tuple<Matrix, Matrix> getBatchOfSize(Matrix X, Matrix Y, int batch_size) {
    assert(X.rows() == Y.rows());

    Matrix batch_x (batch_size, X.cols());
    Matrix batch_y (batch_size, Y.cols());

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

Matrix oneHotEncode(Matrix data) {
    // Assure it's a single column matrix.
    if(data.cols() != 1)
        throw std::invalid_argument("Data must be a single column matrix for one-hot encoding.");

    // Get unique values in the column.
    std::set<float> unique_values;
    for(int i = 0; i < data.rows(); i++) {
        unique_values.insert(data(i, 0));
    }

    Matrix encoded_data(data.rows(), unique_values.size(), 0);
    for(int i = 0; i < data.rows(); i++) {
        auto it = unique_values.find(data(i, 0));
        if(it != unique_values.end()) {
            int index = std::distance(unique_values.begin(), it);
            encoded_data(i, index) = 1.0f; // Set the corresponding column to 1.
        }
    }

    return encoded_data;
}

Matrix argMax(Matrix data) {

    Matrix result(data.rows(), 1, 0);

    for(int i = 0; i < data.rows(); i++) {
        float max_value = data(i, 0);
        int max_index = 0;

        for(int j = 1; j < data.cols(); j++) {
            if(data(i, j) > max_value) {
                max_value = data(i, j);
                max_index = j;
            }
        }

        result(i, 0) = max_index; // Store the column index of the maximum value for each row in the result matrix.
    }
    
    return result;
}