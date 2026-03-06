#include "python/bindings.hpp"



namespace py = pybind11;

PYBIND11_MODULE(scratchneuralnetwork, m) {
    initialize_backend(); // Initialize CUDA if possible

    m.doc() = "Add something here.";

    // loader.hpp
    m.def("load_data", &loadData,
        py::arg("path"),
        py::arg("separator"),
        py::arg("header"),
        py::arg("limit_rows") = false,
        py::arg("limit_rows_amount") = 10000,
        py::return_value_policy::reference
    );

    m.def("train_test_split", &trainTestSplit,
        py::arg("data"),
        py::arg("y_col"),
        py::return_value_policy::reference
    );

    m.def("normalize_data", static_cast<Matrix (*)(Matrix, std::vector<float>, std::vector<float>)>(&normalizeData),
        py::arg("data"),
        py::arg("means"),
        py::arg("stds"),
        py::return_value_policy::reference
    );

    m.def("normalize_data", static_cast<std::tuple<Matrix, std::vector<float>, std::vector<float>> (*)(Matrix)>(&normalizeData),
        py::arg("data"),
        py::return_value_policy::reference
    );

    m.def("unnormalize_data", &unnormalizeData,
        py::arg("data"),
        py::arg("means"),
        py::arg("stds"),
        py::return_value_policy::reference
    );

    // matrix.hpp
    py::class_<Matrix>(m, "Matrix")

        // ============
        // Constructors
        // ============

        .def(py::init<>()) // Empty constructor
    
        .def(py::init<int, int, float>(), // Fill constructor
            py::arg("rows"),
            py::arg("columns"),
            py::arg("fill")
        ) 

        // Getter
        .def_property_readonly("rows",
            &Matrix::rows
        )

        .def_property_readonly("cols",
            &Matrix::cols
        )

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

        .def("head", &Matrix::head, py::return_value_policy::reference)
        .def("tail", &Matrix::tail, py::return_value_policy::reference)

        // Element access: matrix[row, col]
        .def("__getitem__", [](const Matrix &m, std::tuple<int, int> idx) {
            return m(std::get<0>(idx), std::get<1>(idx));
        })

        // Element write: matrix[row, col] = value
        .def("__setitem__", [](Matrix &m, std::tuple<int, int> idx, float val) {
            m(std::get<0>(idx), std::get<1>(idx)) = val;
        })
    ;

    // layer.hpp
    py::class_<Layer>(m, "Layer")
        
        // ============
        // Constructors
        // ============

        .def(py::init<int, int, std::string>(),
            py::arg("in"),
            py::arg("out"),
            py::arg("activation") = "sigmoid"
        )
    ;

    // model.hpp
    py::class_<Model>(m, "Model")
    
        // ============
        // Constructors
        // ============

        .def(py::init<std::string>(),
            py::arg("loss")
        )

        .def("append_layer", &Model::appendLayer)
        .def("backprop", &Model::backprop)
        .def("test", &Model::test)

    ;
}

