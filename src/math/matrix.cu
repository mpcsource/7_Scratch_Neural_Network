#include "math/matrix.hpp"
#include "backend/backend.hpp"

#ifdef USE_CUDA
#include "matrix_gpu.cu"
#endif

// ============
// Constructors
// ============

// Empty constructor
Matrix::Matrix() : rows_(0), cols_(0), h_data() {}

// Fill constructor
Matrix::Matrix(int r, int c, float fill) : rows_(r), cols_(c), h_data(r * c, fill) {
#ifdef USE_CUDA
    if(get_backend() == Backend::CUDA) {
        cudaMalloc(&d_data, r * c * sizeof(float));
        toDevice();
    }
#endif
}

// Array constructor
Matrix::Matrix(int r, int c, std::vector<float> data) : rows_(r), cols_(c), h_data(std::move(data)) {
#ifdef USE_CUDA
    if(get_backend() == Backend::CUDA) {
        cudaMalloc(&d_data, r * c * sizeof(float));
        toDevice();
    }
#endif
}

// Copy constructor
Matrix::Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), h_data() {
#ifdef USE_CUDA
    other.toHost(); // ensure h_data is up to date before copying
#endif
    h_data = other.h_data;
#ifdef USE_CUDA
    if (get_backend() == Backend::CUDA) {
        cudaMalloc(&d_data, rows_ * cols_ * sizeof(float));
        toDevice();
    }
#endif
}

// Move constructor
Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), h_data(std::move(other.h_data)) {
#ifdef USE_CUDA
    d_data = other.d_data;
    cpu_dirty = other.cpu_dirty;
    gpu_dirty = other.gpu_dirty;
    other.d_data = nullptr;
    other.cpu_dirty = false;
    other.gpu_dirty = false;
#endif
    other.rows_ = 0;
    other.cols_ = 0;
}

// Copy assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;

    rows_ = other.rows_;
    cols_ = other.cols_;
#ifdef USE_CUDA
    other.toHost(); // ensure h_data is up to date before copying
#endif
    h_data = other.h_data;

#ifdef USE_CUDA
    if (d_data != nullptr) {
        cudaFree(d_data);
        d_data = nullptr; 
    }
    if (get_backend() == Backend::CUDA) {
        cudaMalloc(&d_data, rows_ * cols_ * sizeof(float));
        gpu_dirty = true;
        cpu_dirty = false;
        toDevice();
    }
#endif

    return *this;
}

// Destructor
Matrix::~Matrix() {
#ifdef USE_CUDA
    if (d_data != nullptr) {
        cudaFree(d_data);
        d_data = nullptr;
    }
#endif
}

// Row amt getter
int Matrix::rows() const {
    return this->rows_;
}

// Col amt getter
int Matrix::cols() const {
    return this->cols_;
}

// Copy data from host to device
void Matrix::toDevice() const {
    if(get_backend() == Backend::CUDA && d_data != nullptr && gpu_dirty) {
        cudaMemcpy(
            d_data,
            h_data.data(),
            rows_ * cols_ * sizeof(float),
            cudaMemcpyHostToDevice
        );
        gpu_dirty = false;
    }
};

// Copy data from device to host
void Matrix::toHost() const {
    if(get_backend() == Backend::CUDA && d_data != nullptr && cpu_dirty) {
        cudaMemcpy(
            const_cast<float*>(h_data.data()),
            d_data,
            rows_ * cols_ * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        cpu_dirty = false;
    }
};

__global__ void matrix_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

Matrix Matrix::add(const Matrix& other) const {
    assert(this->rows() == other.rows());
    assert(this->cols() == other.cols());

    Matrix out(this->rows(), this->cols(), 0);

#ifdef USE_CUDA

    // Use CUDA-based implementation if allowed and possible
    if (get_backend() == Backend::CUDA) {
    
        this->toDevice();
        other.toDevice();

        int threads = 256;
        int size = this->rows() * this->cols();
        int blocks = (size + threads - 1) / threads;

        matrix_add_kernel<<<blocks, threads>>>(this->d_data, other.d_data, out.d_data, size);

        cudaDeviceSynchronize();
        out.cpu_dirty = true;

        return out;
    }
#endif

    // Fallback CPU Implementation
    for (int i = 0; i < this->rows(); i++) {
        for (int j = 0; j < this->cols(); j++) {
                out(i, j) = (*this)(i, j) + (other)(i, j);
        }
    }

    return out;
}

Matrix Matrix::subtract(const Matrix& other) const {
    toHost();
    assert(this->rows() == other.rows());
    assert(this->cols() == other.cols());

    Matrix out(this->rows(), this->cols(), 0);

    for (int i = 0; i < this->rows(); i++)
        for (int j = 0; j < this->cols(); j++)
            out(i, j) = (*this)(i, j) - other(i, j);

    return out;
}

Matrix Matrix::multiply(const Matrix& other) const {
    toHost();
    assert(this->rows() == other.rows());
    assert(this->cols() == other.cols());

    Matrix out(this->rows(), this->cols(), 0);

    for (int i = 0; i < this->rows(); i++)
        for (int j = 0; j < this->cols(); j++)
            out(i, j) = (*this)(i, j) * other(i, j);

    return out;
}

Matrix Matrix::multiply(float other) const {
    toHost();
    // Create output matrix.
    Matrix out(this->rows(), this->cols(), 0);

    // Perform multiplication.
    for (int i = 0; i < this->rows(); i++)
        for (int j = 0; j < this->cols(); j++)
            out(i, j) = (*this)(i, j) * other;

    // Return result.
    return out;
}

#define TILE 16

__global__ void matrix_dot_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {

        if (row < M && t*TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] =
                A[row*K + t*TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        if (col < N && t*TILE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] =
                B[(t*TILE + threadIdx.y)*N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int i = 0; i < TILE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row*N + col] = sum;
}

Matrix Matrix::dot(const Matrix& other) const {
    assert(this->cols() == other.rows());

    Matrix out(this->rows(), other.cols(), 0);

#ifdef USE_CUDA

    // Use CUDA-based implementation if allowed and possible
    if (get_backend() == Backend::CUDA) {

        this->toDevice();
        other.toDevice();

        int M = rows();
        int K = cols();
        int N = other.cols();

        dim3 block(TILE, TILE);
        
        dim3 grid(
            (N + TILE - 1) / TILE,
            (M + TILE - 1) / TILE
        );

        matrix_dot_kernel<<<grid, block>>>(this->d_data, other.d_data, out.d_data, M, K, N);

        cudaDeviceSynchronize();
        out.cpu_dirty = true;

        return out;
    }
#endif

    // Fallback CPU Implementation
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
    toHost();
    Matrix transposed(this->cols_, this->rows_, 0);

    for (size_t i = 0; i < this->rows_; i++)
        for (size_t j = 0; j < this->cols_; j++)
            transposed(j, i) = (*this)(i, j);

    return transposed;
}

Matrix Matrix::exponential() const {
    toHost();
    Matrix result(this->rows_, this->cols_, 0);
    for (int i = 0; i < this->rows_; i++)
        for (int j = 0; j < this->cols_; j++)
            result(i, j) = std::exp((*this)(i, j));
    return result;
}

Matrix Matrix::apply(std::function<float(float)> func) const {
    toHost();
    Matrix out(this->rows(), this->cols());
    for (int i = 0; i < this->rows(); i++)
        for (int j = 0; j < this->cols(); j++)
            out(i, j) = func((*this)(i, j));
    return out;
}

Matrix Matrix::getRow(int row) const {
    toHost();
    Matrix result(1, this->cols_, 0);
    for (int i = 0; i < this->cols_; i++) {
        result(0, i) = (*this)(row, i);
    }
    return result;
}

void Matrix::basicPrint() const {
    toHost();

    for (size_t i = 0; i < this->rows_; i++) {
        for (size_t j = 0; j < this->cols_; j++)
            std::cout << (*this)(i, j) << " ";
        std::cout << std::endl;
    }
}

float &Matrix::operator()(int row, int col){
    gpu_dirty = true;
    return this->h_data[row * this->cols_ + col];
}

const float &Matrix::operator()(int row, int col) const {
    toHost();
    return this->h_data[row * this->cols_ + col];
}

Matrix Matrix::head() const {
    toHost();
    Matrix result(5, this->cols_);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < this->cols_; j++)
            result(i, j) = (*this)(i, j);
    return result;
}

Matrix Matrix::tail() const {
    toHost();
    Matrix result(5, this->cols_);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < this->cols_; j++)
            result(i, j) = (*this)(i + this->rows_ - 5, j);
    return result;
}