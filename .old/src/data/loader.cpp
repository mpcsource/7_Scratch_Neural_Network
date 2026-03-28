#include "data/loader.hpp"
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

// ─── helpers ──────────────────────────────────────────────────────────────────

// Read every cell as a raw string, optionally capped at limitRowsAmount data rows.
// Returns {column_names, raw_rows}.
static std::pair<std::vector<std::string>, std::vector<std::vector<std::string>>>
readRaw(const std::string& path, char separator, bool header,
        bool limitRows, int limitRowsAmount)
{
    std::vector<std::string> col_names;
    std::vector<std::vector<std::string>> raw;

    std::ifstream file(path);
    if (!file)
        throw std::invalid_argument("Failed to open file: " + path);

    std::string line;
    for (int iteration = 0; std::getline(file, line); iteration++) {
        if (limitRows && (int)raw.size() >= limitRowsAmount)
            break;

        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, separator))
            row.push_back(cell);

        if (iteration == 0 && header) {
            col_names = row;
            continue;
        }
        if (!row.empty())
            raw.push_back(row);
    }
    return {col_names, raw};
}

// ─── loadData ─────────────────────────────────────────────────────────────────

Matrix loadData(const std::string& path, char separator, bool header,
                bool limitRows, int limitRowsAmount)
{
    auto [col_names, raw] = readRaw(path, separator, header, limitRows, limitRowsAmount);

    if (raw.empty())
        throw std::invalid_argument("No data rows found in: " + path);

    int rows = (int)raw.size();
    int cols = (int)raw[0].size();

    // ── Pass 1: classify each column and build string→float label encodings ──
    std::vector<bool>                              is_string(cols, false);
    std::vector<std::unordered_map<std::string,float>> encodings(cols);
    std::vector<float> col_sum(cols, 0.f);
    std::vector<int>   col_count(cols, 0);

    for (int j = 0; j < cols; j++) {
        std::unordered_map<std::string,float> enc;
        float next_label = 0.f;

        for (int i = 0; i < rows; i++) {
            if (j >= (int)raw[i].size() || raw[i][j].empty())
                continue;   // missing – handled in pass 2

            try {
                float v = std::stof(raw[i][j]);
                col_sum[j]   += v;
                col_count[j] += 1;
            } catch (...) {
                is_string[j] = true;
                if (enc.find(raw[i][j]) == enc.end()) {
                    std::cout << "[loadData] encoding column "
                              << (col_names.empty() ? std::to_string(j) : col_names[j])
                              << ": \"" << raw[i][j] << "\" → " << next_label << "\n";
                    enc[raw[i][j]] = next_label++;
                }
            }
        }
        if (is_string[j])
            encodings[j] = std::move(enc);
    }

    // ── Compute per-column mean for numeric imputation ─────────────────────
    std::vector<float> col_mean(cols, 0.f);
    for (int j = 0; j < cols; j++)
        if (!is_string[j] && col_count[j] > 0)
            col_mean[j] = col_sum[j] / (float)col_count[j];

    // ── Pass 2: fill matrix ────────────────────────────────────────────────
    int imputed = 0;
    Matrix data_matrix(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            bool missing = j >= (int)raw[i].size() || raw[i][j].empty();

            if (missing) {
                data_matrix(i, j) = is_string[j] ? 0.f : col_mean[j];
                imputed++;
            } else if (is_string[j]) {
                data_matrix(i, j) = encodings[j].at(raw[i][j]);
            } else {
                try {
                    data_matrix(i, j) = std::stof(raw[i][j]);
                } catch (...) {
                    // Non-numeric cell in a numeric column – treat as missing.
                    data_matrix(i, j) = col_mean[j];
                    imputed++;
                }
            }
        }
    }

    if (imputed > 0)
        std::cout << "[loadData] imputed " << imputed
                  << " missing value(s) with column means.\n";

    return data_matrix;
}

std::tuple<Matrix, Matrix, Matrix, Matrix> trainTestSplit(Matrix data, int y_col) {

    // # Shuffle data.
    for(int i = 0; i < data.rows(); i++) {
        int j = rand() % data.rows();

        Matrix a_i = data.getRow(i);
        Matrix a_j = data.getRow(j);

        // Exchange both.
        for(int k = 0; k < a_j.cols(); k++) 
            data(i, k) = a_j(0, k);
        for(int k = 0; k < a_i.cols(); k++)
            data(j, k) = a_i(0, k);
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
            int x_col = 0;
            for(int j = 0; j < data.cols(); j++) {
                if(j == y_col)
                    continue; // # Skip y_col.
                x_train(train_row, x_col++) = data(i, j);
            }
            train_row++;
        } else {
            // # Fill testing row.
            y_test(test_row, 0) = data(i, y_col);
            int x_col = 0;
            for(int j = 0; j < data.cols(); j++) {
                if(j == y_col)
                    continue; // # Skip y_col.
                x_test(test_row, x_col++) = data(i, j);
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
        int random_row_index = rand() % X.rows();
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

// ─── inspectData ──────────────────────────────────────────────────────────────

void inspectData(const std::string& path, char separator, bool header)
{
    auto [col_names, raw] = readRaw(path, separator, header, false, 0);

    if (raw.empty()) {
        std::cout << "[inspectData] No data rows found.\n";
        return;
    }

    int rows = (int)raw.size();
    int cols = (int)raw[0].size();

    // Pad col_names if header was absent.
    while ((int)col_names.size() < cols)
        col_names.push_back("col_" + std::to_string(col_names.size()));

    // Per-column accumulators.
    struct ColStats {
        int  missing  = 0;
        bool is_str   = false;
        float vmin    = std::numeric_limits<float>::max();
        float vmax    = std::numeric_limits<float>::lowest();
        double sum    = 0.0;
        int  count    = 0;
        std::unordered_map<std::string,int> value_counts;
    };
    std::vector<ColStats> stats(cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j >= (int)raw[i].size() || raw[i][j].empty()) {
                stats[j].missing++;
                continue;
            }
            const std::string& cell = raw[i][j];
            try {
                float v = std::stof(cell);
                stats[j].sum   += v;
                stats[j].count += 1;
                stats[j].vmin   = std::min(stats[j].vmin, v);
                stats[j].vmax   = std::max(stats[j].vmax, v);
            } catch (...) {
                stats[j].is_str = true;
                stats[j].value_counts[cell]++;
            }
        }
    }

    // ── Print report ──────────────────────────────────────────────────────
    const int W = 20;
    std::cout << "\n" << std::string(80, '=') << " Data Inspection " << std::string(80, '=') << "\n";
    std::cout << "File  : " << path << "\n";
    std::cout << "Rows  : " << rows << "   Columns: " << cols << "\n\n";

    std::cout << std::left
              << std::setw(W) << "Column"
              << std::setw(9)  << "Missing"
              << std::setw(10) << "Type"
              << std::setw(12) << "Min"
              << std::setw(12) << "Max"
              << "Mean / Categories\n";
    std::cout << std::string(80, '-') << "\n";

    for (int j = 0; j < cols; j++) {
        const ColStats& s = stats[j];
        std::cout << std::left << std::setw(W) << col_names[j];
        std::cout << std::setw(9) << s.missing;

        if (s.is_str) {
            std::cout << std::setw(10) << "string"
                      << std::setw(12) << "-"
                      << std::setw(12) << "-";
            // List unique categories.
            std::cout << s.value_counts.size() << " unique: ";
            for (auto& [k, cnt] : s.value_counts)
                std::cout << k << "(" << cnt << ") ";
        } else {
            float mean = s.count > 0 ? (float)(s.sum / s.count) : 0.f;
            std::cout << std::setw(10) << "numeric"
                      << std::setw(12) << (s.count > 0 ? std::to_string(s.vmin) : "N/A")
                      << std::setw(12) << (s.count > 0 ? std::to_string(s.vmax) : "N/A")
                      << mean;
        }
        std::cout << "\n";
    }
    std::cout << std::string(80, '=') << "\n\n";
}