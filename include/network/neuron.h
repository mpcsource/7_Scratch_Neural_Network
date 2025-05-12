#include "math/scalar.h"

template <
    // Generic type name.
    typename T,
    // Restrict type to numbers only.
    typename = std::enable_if_t<std::is_arithmetic<T>::value>
>
struct Neuron: public std::__enable_shared_from_this<Neuron<T>>
{
    std::vector<std::shared_ptr<Scalar<T>>> weights;
    std::shared_ptr<Scalar<T>> bias;

    Neuron(int in_features) : weights(in_features, Scalar<T>(1)), bias(0) {}

    // Forward pass.
    std::shared_ptr<Scalar<T>> operator () (std::vector<std::shared_ptr<Scalar<T>>> x) {
        // w * x + b
        T out = this->bias;
        for(int i = 0; i < this->weights.size(); i++)
            out += w.at(i) * x.at(i);
        out = out.tanh();

        return out;
    }

    // Get parameters.
    std::vector<std::shared_ptr<Scalar<T>>> getParameters() {
        std::vector<Scalar<T>> parameters = weights;
        parameters.push_back(bias);
        return parameters;
    }
};
