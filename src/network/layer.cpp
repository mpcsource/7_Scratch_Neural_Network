#include "network/layer.hpp"

Layer::Layer(int nin, int nout, std::string activation) : weights_(Matrix(nout, nin, 1)),
                                                          biases_(Matrix(nout, 1, 0.0)),
                                                          activation_(activation)
{
    // Initialise weights.
    float limit = 0;
    if (activation == "sigmoid")
        // Glorot/Xavier initialization.
        limit = sqrt(6.0f / (nin + nout));
    else if (activation == "relu")
        // He initialization.
        limit = sqrt(2.0f / nin);
    else
        // Default initialization.
        limit = 1.0f;
    // Fill weights with random values in range [-limit, limit].
    for (int i = 0; i < nout; i++)
        for (int j = 0; j < nin; j++)
            this->weights_(i, j) = randomRange(-limit, limit);
}

void Layer::pass(Matrix x)
{
    this->x_ = x;
    this->z_ = this->weights_.dot(x).add(this->biases_);

    if (this->activation_ == "sigmoid")
    {
        // Sigmoid function.
        auto sigmoid = [](float x)
        { return 1.0 / (1.0 + std::exp(-x)); };

        // Sigmoid derivative function.
        auto sigmoid_deriv = [](float x)
        {
            float s = 1.0 / (1.0 + std::exp(-x));
            return s * (1.0 - s);
        };

        // Apply both.
        this->a_ = this->z_.apply(sigmoid);
        this->da_ = this->z_.apply(sigmoid_deriv);
    }
    else if (this->activation_ == "relu")
    {
        // ReLU function.
        auto relu = [](float x)
        { return x > 0 ? x : 0; };

        // ReLU derivative function.
        auto relu_deriv = [](float x)
        { return x > 0 ? 1.0 : 0.0; };

        // Apply both.
        this->a_ = this->z_.apply(relu);
        this->da_ = this->z_.apply(relu_deriv);
    }
    else // Linear by default.
    {
        this->a_ = this->z_;
        this->da_ = Matrix(this->z_.rows(), this->z_.cols(), 1.0);
    }
}

void Layer::backward(float learning_rate)
{

    Matrix grad_w = this->delta_z.dot(this->x_.transpose());
    Matrix grad_b = this->delta_z;

    this->weights_ = this->weights_.subtract(grad_w.multiply(learning_rate));
    this->biases_ = this->biases_.subtract(grad_b.multiply(learning_rate));
}

Matrix &Layer::weights()
{
    return this->weights_;
}

const Matrix &Layer::weights() const
{
    return this->weights_;
}

Matrix &Layer::biases()
{
    return this->biases_;
}

const Matrix &Layer::biases() const
{
    return this->biases_;
}

void Layer::debugPrint()
{
    std::cout << "l1 weights:" << std::endl;
    this->weights().basicPrint();
    std::cout << "l1 biases:" << std::endl;
    this->biases().basicPrint();
}