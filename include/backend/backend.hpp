#pragma once

enum class Backend {
    CPU,
    CUDA
};

Backend get_backend();
bool cuda_available();
void initialize_backend();
