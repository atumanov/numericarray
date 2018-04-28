cmake_minimum_required(VERSION 2.8.12)
project(numericarray)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mavx512vl")

add_subdirectory(pybind11)
pybind11_add_module(numericarray numericarray.cc)
