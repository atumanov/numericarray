#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
    auto buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    double *ptr1 = (double *) buf1.ptr;
    double *ptr2 = (double *) buf2.ptr;

    double new_val;
    for (size_t idx = 0; idx < buf1.shape[0]; idx++) {
      double old_val = ptr1[idx];
      do {
	new_val = ptr1[idx] + ptr2[idx];
      } while (__sync_bool_compare_and_swap(reinterpret_cast<uint64_t*>(&ptr1[idx]),
					    *reinterpret_cast<uint64_t*>(&old_val),
					    *reinterpret_cast<uint64_t*>(&new_val)));
    }
}

PYBIND11_MODULE(atomicarray, m) {
    m.def("increment", &add_arrays, "Increment first NumPy array by the second one");
}
