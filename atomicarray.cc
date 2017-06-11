#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
    auto buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    volatile double *ptr1 = (double *) buf1.ptr;
    double *ptr2 = (double *) buf2.ptr;

    union bits { double f; int64_t i; };
    bits old_val, new_val;
    for (size_t idx = 0; idx < buf1.shape[0]; idx++) {
      do {
	old_val.f = ptr1[idx];
	new_val.f = old_val.f + ptr2[idx];
	// On IA64/x64, adding a PAUSE instruction in compare/exchange loops
        // is recommended to improve performance.  (And it does!)
#if (defined(__i386__) || defined(__amd64__))
        __asm__ __volatile__ ("pause\n");
#endif
      } while (!__sync_bool_compare_and_swap(reinterpret_cast<volatile int64_t*>(&ptr1[idx]),
					     old_val.i,
					     new_val.i));
    }
}

PYBIND11_MODULE(atomicarray, m) {
    m.def("increment", &add_arrays, "Increment first NumPy array by the second one");
}
