#include "network/model.hpp"

Model::Model(std::string loss) : loss_(loss) {}

void Model::appendLayer(Layer *layer)
{
    this->layers_.push_back(layer);
}

Matrix Model::forward(Matrix x)
{
    this->x_ = x;
    for (Layer *layer : this->layers_)
    {
        layer->pass(x);
        x = layer->a_;
    }
    return x;
}

void Model::backward(Matrix label, float learning_rate)
{

    Layer *out_layer = this->layers_.back();
    
    Matrix dloss, delta_o;
    // Handle different loss functions.
    if(this->loss_ == "cross_entropy") {
        dloss = out_layer->a_.subtract(label);
        delta_o = dloss;
    }
    else { // Default to MSE.
        dloss = out_layer->a_.subtract(label);
        delta_o = dloss.multiply(out_layer->da_);
    }

    out_layer->delta_z = delta_o;
    out_layer->backward(learning_rate);

    for (int i = this->layers_.size() - 1; i > 0; i--)
    {
        if (i == this->layers_.size() - 1)
            continue; // Skip output layer
        Layer *next_layer = this->layers_.at(i + 1);
        Layer *layer = this->layers_.at(i);

        auto delta_h = next_layer->weights().transpose().dot(next_layer->delta_z).multiply(layer->da_);
        layer->delta_z = delta_h;
        layer->backward(learning_rate);
    }
}

void Model::backprop(Matrix data, Matrix labels, int epochs, float learning_rate)
{

    int size = data.rows();
    for (int epoch_i = 0; epoch_i < epochs; epoch_i++)
    {
        std::cout << "Epoch: " << epoch_i + 1 << std::endl;
        for (int train_i = 0; train_i < size; train_i++)
        {
            auto image = data.getRow(train_i).transpose();
            auto label = labels.getRow(train_i).transpose();

            auto label_hat = this->forward(image);

            this->backward(label, learning_rate);
        }
    }
}

Matrix Model::test(Matrix data, Matrix labels)
{
    Matrix labels_hat(labels.rows(), labels.cols(), 0);
    for (int test_i = 0; test_i < data.rows(); test_i++)
    {
        auto image = data.getRow(test_i).transpose();
        auto label = labels.getRow(test_i).transpose();

        auto label_hat = this->forward(image);
        for (int j = 0; j < labels.cols(); j++)
            labels_hat(test_i, j) = label_hat(0, j);
    }
    return labels_hat;
}