#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <x86intrin.h>
#include <omp.h>
#define TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512 4352

#include <iostream>

namespace py = pybind11;

void adds_AVX512(float *y, const float *x, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)
  for (i=0; i<=((n)-16); i+=16) {
    __m512 YMM0;
    YMM0 = _mm512_loadu_ps(x+i);
    _mm512_storeu_ps(y+i, YMM0);
  }
  off = (n) - ((n)%16);
  for (i=off; i<(n); i++) {
    y[i] = x[i];
  }
}

void add_arrays(py::array_t<float> input1, py::array_t<float> input2) {
  auto buf1 = input1.request(), buf2 = input2.request();

  if (input1.ndim() != 1 || input1.ndim() != 1)
    throw std::runtime_error("Number of dimensions must be one");


  float *ptr1 = (float *) buf1.ptr;
  float *ptr2 = (float *) buf2.ptr;

  adds_AVX512(ptr1, ptr2, buf1.shape[0]);
}

PYBIND11_MODULE(numericarray, m) {
    m.def("increment", &add_arrays, "Increment first NumPy array by the second one");
}
