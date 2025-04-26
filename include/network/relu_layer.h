#pragma once
#include "network/layer.h"

template <
    // # Generic type name.
    typename T,
    // # Restrict type to numbers only.
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type 
>
class ReLU_Layer : public Layer<T>
{
    public:
        using Layer<T>::Layer;

        Matrix<T> calculate_activation(Matrix<T> x) override {
            for(int i = 0; i < x.rows(); i++)
                for(int j = 0; j < x.cols(); j++)
                    x(i,j) = std::max(x(i,j), (T)0);

            return x;
        }

        Matrix<T> dZ(Matrix<T> dE) override {
            for(int i = 0; i < dE.rows(); i++)
                for(int j = 0; j < dE.cols(); j++)
                    dE(i,j) = (dE(i,j) > 0) ? (T)1 : (T)0;
                    
            return dE;
        }
};