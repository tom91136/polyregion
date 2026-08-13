cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(spectra LANGUAGES CXX Fortran)

add_library(spectra_cpp_consumer OBJECT test/cpp/check_surface.cpp)
target_compile_features(spectra_cpp_consumer PRIVATE cxx_std_17)
target_include_directories(spectra_cpp_consumer PRIVATE generated/cpp/include)

add_library(
  spectra_fortran_consumer OBJECT
  generated/fortran/spectra_api.f90
  test/fortran/check_surface.f90
)
