#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../core/tensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    
    m.doc() = "This really needs to be something.";

    py::class_<Tensor>(m, "CTensor")
        .def(py::init<std::vector<int>, std::vector<float>>(),
            py::arg("shape") = std::vector<int>{},
            py::arg("data") = std::vector<float>{}
        )
        .def_property_readonly("shape", [](const Tensor &t) {
            return t.get_shape();
        })
        .def_property_readonly("flat", [](const Tensor &t) {
            return t.get_data();
        })
        .def("add_tensor", &Tensor::add_tensor)
        .def("add_bias", &Tensor::add_bias)
        .def("transpose_tensor", &Tensor::transpose_tensor)
        .def("sum_cols_tensor", &Tensor::sum_cols_tensor)
        .def("sub_tensor", &Tensor::sub_tensor)
        .def("mul_tensor", &Tensor::mul_tensor)
        .def("mul_tensor_number", &Tensor::mul_tensor_number)
        .def("dot_tensor", &Tensor::dot_tensor)
        .def("zero_grad", &Tensor::zero_grad)
        .def("accumulate_grad", &Tensor::accumulate_grad)
        .def("get_grad", &Tensor::get_grad)
    ;
}