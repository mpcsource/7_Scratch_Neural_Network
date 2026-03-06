#include "python/bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(scratchneuralnetwork, m) {
    m.doc() = "Add something here.";

    // matrix.hpp
    py::class_<Matrix>(m, "Matrix")

        // ============
        // Constructors
        // ============

        .def(py::init<>()) // Empty constructor
        .def(py::init<int, int, float>()) // Fill constructor

        // ===============
        // Math operations
        // ===============

        // Add function
        .def("add", &Matrix::add)
        // Override + operator
        .def("__add__", [](const Matrix &a, const Matrix &b) {
            return a.add(b);
        })

        // Subtract function
        .def("subtract", &Matrix::subtract)
        // Override - operator
        .def("__sub__", [](const Matrix &a, const Matrix &b) {
            return a.subtract(b);
        })

        // Element-wise multiplication function
        .def("multiply", static_cast<Matrix (Matrix::*)(const Matrix&) const>(&Matrix::multiply))
        .def("multiply", static_cast<Matrix (Matrix::*)(float) const>(&Matrix::multiply))
        // Override * operator
        .def("__mul__", [](const Matrix &a, const Matrix &b) {
            return a.multiply(b);
        })
        .def("__mul__", [](const Matrix &a, float b) {
            return a.multiply(b);
        })

        // Dot product function
        .def("dot", &Matrix::dot)
        // Override @ operator
        .def("__matmul__", [](const Matrix &a, const Matrix &b) {
            return a.dot(b);
        })

        // ===============
        // Other functions
        // ===============
        
        // Debug printing
        //.def("__str__", &Matrix::basicPrint)
        .def("print", &Matrix::basicPrint)
    ;
}

