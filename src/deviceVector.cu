// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Andrew Simmonett, Samarjeet Prasad
//
// ENDLICENSE

#include "cuda_utils.h"
#include "deviceVector.h"

/*
template<typename T>
deviceVector<T>::deviceVector(){

}

template<typename T>
deviceVector<T>::deviceVector(unsigned long long size){
    resize((size_t)size);
}
template<typename T>
deviceVector<T>::deviceVector(size_t size){
    resize(size);
}
*/

/*
template <typename T> deviceVector<T>::~deviceVector() {
  // std::cout << "I am in the dV destructor\n";
  if (array_ != NULL)
    cudaFree(array_);
}
*/

template <typename T> void deviceVector<T>::resize(size_t size) {
  if (size_ < size) {
    T *newArray;
    cudaCheck(cudaMalloc(&newArray, sizeof(T) * size));
    if (size_)
      cudaCheck(cudaMemcpy(newArray, array_, sizeof(T) * size_,
                           cudaMemcpyDeviceToDevice));
    if (size_ > 0)
      cudaCheck(cudaFree(array_));
    array_ = newArray;
    size_ = size;
  } else if (size_ > size) {
    size_ = size;
  }
}

template <typename T> T *deviceVector<T>::data() { return array_; }
