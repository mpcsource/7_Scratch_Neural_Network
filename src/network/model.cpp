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

void Model::backward(Matrix label, float learning_rate, int batch_size)
{
    // ---- Pass 1: compute all delta_z values using the CURRENT (pre-update) weights ----
    Layer *out_layer = this->layers_.back();
    Matrix dloss = out_layer->a_.subtract(label);
    Matrix delta_o = dloss.multiply(out_layer->da_).multiply(1.0f / batch_size);
    out_layer->delta_z = delta_o;

    for (int i = (int)this->layers_.size() - 2; i >= 0; i--)
    {
        Layer *next_layer = this->layers_.at(i + 1);
        Layer *layer     = this->layers_.at(i);
        layer->delta_z = next_layer->weights().transpose().dot(next_layer->delta_z).multiply(layer->da_);
    }

    // ---- Pass 2: update weights now that all deltas are correct ----
    for (Layer *layer : this->layers_)
        layer->backward(learning_rate);
}

void Model::backprop(Matrix data, Matrix labels, int epochs, float learning_rate, int batch_size)
{
    /*
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
    }*/

    for (int epoch_i = 0; epoch_i < epochs; epoch_i++) {
        std::cout << "Epoch: " << epoch_i + 1 << std::endl;

        int steps = data.rows() / batch_size;
        for (int step = 0; step < steps; step++) {
            auto [batch_x, batch_y] = getBatchOfSize(data, labels, batch_size);

            auto label_hat = this->forward(batch_x.transpose());
            this->backward(batch_y.transpose(), learning_rate, batch_size);
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