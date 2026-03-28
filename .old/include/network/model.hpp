#pragma once
#include <vector>
#include <iostream>
#include "network/layer.hpp"
#include "math/matrix.hpp"
#include "data/loader.hpp"

enum class Decay { NONE, EXPONENTIAL, STEP, COSINE };

class Model {
    private:
        std::vector<Layer*> layers_;
        std::string loss_;
        Decay   decay_;
        float   decay_rate_;   // multiplier per epoch/step (EXPONENTIAL, STEP)
        int     decay_steps_;  // epoch interval for STEP decay
        float   lr_min_;       // floor for COSINE annealing
        Matrix x_;
        Matrix out_;

    public:
        Model(std::string loss,
              Decay decay      = Decay::NONE,
              float decay_rate = 0.96f,
              int   decay_steps = 10,
              float lr_min     = 1e-6f);

        void appendLayer(Layer * layer);

        Matrix forward(Matrix x);

        void backward(Matrix label, float learning_rate = 0.01f, int batch_size = 32);

        void backprop(Matrix data, Matrix labels, int epochs = 10, float learning_rate = 0.01f, int batch_size = 32);

        Matrix test(Matrix data, Matrix labels);    
};