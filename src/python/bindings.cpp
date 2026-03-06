#include "python/bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(scratchneuralnetwork, m) {
    m.doc() = "Add something here.";
}