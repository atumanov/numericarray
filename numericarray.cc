#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>

namespace py = pybind11;

void add_arrays(py::array_t<float> input1, py::array_t<float> input2) {
  auto a = input1.mutable_unchecked<1>();
  auto b = input1.unchecked<1>();
  
  if (input1.ndim() != 1 || input1.ndim() != 1)
    throw std::runtime_error("Number of dimensions must be one");

  for (size_t idx = 0; idx < input1.shape(0); idx++) {
    a(idx) += b(idx);
  }
}

PYBIND11_MODULE(numericarray, m) {
    m.def("increment", &add_arrays, "Increment first NumPy array by the second one");
}
