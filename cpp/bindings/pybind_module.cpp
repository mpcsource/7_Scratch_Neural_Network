#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../core/tensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    
    m.doc() = "This really needs to be something.";

    py::class_<Tensor>(m, "CTensor")
        .def(py::init<int, int>(),
            py::arg("rows") = 0,
            py::arg("cols") = 0
        )
        .def(py::init<const Tensor&>())
        .def("add_tensor", &Tensor::add_tensor)
        .def("sub_tensor", &Tensor::sub_tensor)
        .def("mul_tensor", &Tensor::mul_tensor)
        .def("mul_tensor_number", &Tensor::mul_tensor_number)
        .def("dot_tensor", &Tensor::dot_tensor)
    ;
}