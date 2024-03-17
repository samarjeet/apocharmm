// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Andrew Simmonett, Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include <cuda_runtime.h>

#include <iostream>
template <typename T> class deviceVector {
private:
  size_t size_ = 0;
  T *array_;

public:
  deviceVector() {}
  //  ~deviceVector();
  deviceVector(size_t size) {
    std::cout << "in dv cons\n";
    resize(size);
  };
  deviceVector(unsigned long long size);
  void resize(size_t size);
  T *data();
  void set(T *devPtr) { array_ = devPtr; }
};

template class deviceVector<float>;
template class deviceVector<int>;
template class deviceVector<double>;
template class deviceVector<float4>;
template class deviceVector<int4>;
template class deviceVector<int2>;
template class deviceVector<double4>;
