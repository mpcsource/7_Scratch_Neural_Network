#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include "math/matrix.hpp"
#include <math.h>


Matrix loadData(const std::string& path, char separator, bool header, bool limitRows = false, int limitRowsAmount = 10000);


std::tuple<Matrix, Matrix, Matrix, Matrix> trainTestSplit(Matrix data, int y_col);

// # Normalize data with given mean and standard deviation.
Matrix normalizeData(Matrix data, std::vector<float> means, std::vector<float> stds);

std::tuple<Matrix, std::vector<float>, std::vector<float>> normalizeData(Matrix data);



Matrix unnormalizeData(Matrix data, std::vector<float> means, std::vector<float> stds);

std::tuple<Matrix, Matrix> getBatchOfSize(Matrix X, Matrix Y, int batch_size = 32);