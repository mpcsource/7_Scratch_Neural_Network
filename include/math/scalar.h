#include <type_traits>
#include <vector>
#include <memory>
#include <cmath>



template <
    // Generic type name.
    typename T,
    // Restrict type to numbers only.
    typename = std::enable_if_t<std::is_arithmetic<T>::value>
>


struct Scalar: public std::enable_shared_from_this<Scalar<T>> {
    T data;
    T grad;
    std::function<void()> _backward;
    
    Scalar(T data_) : data(data_), grad(0), _backward([](){}) {}

    T getData() { return this->data; }

    std::shared_ptr<Scalar<T>> operator + (const std::shared_ptr<Scalar<T>>& other) { 
        // Initialise & calculate output scalar. 
        auto out = std::make_shared<Scalar<T>>(data + other->data);

        // Define _backward lambda.
        out->_backward = [
            self = this->shared_from_this(),
            other,
            out 
        ](){
            self->grad += out->grad;
            other->grad += out->grad;
        };

        // Return output scalar.
        return out; 
    }

    
    std::shared_ptr<Scalar<T>> operator * (const std::shared_ptr<Scalar<T>>& other) { 
        // Initialise & calculate output scalar.
        auto out = std::make_shared<Scalar<T>>(data * other->data);

        // Define _backward lambda.
        out->_backward = [
            self = this->shared_from_this(),
            other,
            out 
        ](){
            self->grad += other->data * out->grad;
            other->grad += self->data * out->grad;
        };

        return out; 
    }

    std::shared_ptr<Scalar<T>> power (const std::shared_ptr<Scalar<T>>& other) { 
        // Initialise & calculate output scalar.
        auto out = std::make_shared<Scalar<T>>(data * other->data);

        // Define _backward lambda.
        out->_backward = [
            self = this->shared_from_this(),
            other,
            out 
        ](){
            self->grad += other->data * pow(self->data, other->data-1) * out->grad;
        };

        return out; 
    }

    std::shared_ptr<Scalar<T>> exponent (const std::shared_ptr<Scalar<T>>& other) { 
        // Initialise & calculate output scalar.
        auto out = std::make_shared<Scalar<T>>(exp(this->data));

        // Define _backward lambda.
        out->_backward = [
            self = this->shared_from_this(),
            other,
            out 
        ](){
            self->grad += other->data * out->grad;
        };

        return out; 
    }

    

    std::shared_ptr<Scalar<T>> tanh (const std::shared_ptr<Scalar<T>>& other) { 
        // Initialise & calculate output scalar.
        auto x = this->data;
        auto t = (exp(2*x) - 1)/(exp(2*x) + 1);
        auto out = std::make_shared<Scalar<T>>(t);

        // Define _backward lambda.
        out->_backward = [
            self = this->shared_from_this(),
            other,
            out,
            t
        ](){
            self->grad += (1 - pow(t, 2)) * out->grad;
        };

        return out; 
    }

};