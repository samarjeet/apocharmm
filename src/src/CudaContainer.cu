// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "CudaContainer.h"

#include "cuda_utils.h"
#include <iostream>
// template <typename T> CudaContainer<T>::CudaContainer() {}
// template <typename T> CudaContainer<T>::~CudaContainer() {}

template <typename T> void CudaContainer<T>::allocate(size_t size) {
  hostArray = std::vector<T>(size);
  // deviceArray = deviceVector<T>(size);
  // deviceArray = deviceVector<T>();
  deviceArray.resize(size);
}

template <typename T>
void CudaContainer<T>::setHostArray(const std::vector<T> &array) {
  // assert(array.size() == size());
  hostArray = std::vector<T>(array.cbegin(), array.cend());
}

template <typename T> void CudaContainer<T>::setDeviceArray(T *devPtr) {
  deviceArray.set(devPtr);
}

template <typename T> const std::vector<T> &CudaContainer<T>::getHostArray() {
  return hostArray;
}

template <typename T> void CudaContainer<T>::transferToDevice() {
  deviceArray.resize(hostArray.size());
  cudaCheck(cudaMemcpy(deviceArray.data(), hostArray.data(), sizeof(T) * size(),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaDeviceSynchronize());
}

template <typename T> void CudaContainer<T>::transferFromDevice() {
  cudaMemcpy(hostArray.data(), deviceArray.data(), sizeof(T) * size(),
             cudaMemcpyDeviceToHost);
}

template <typename T> void CudaContainer<T>::transferToHost() {
  transferFromDevice();
}
template <typename T> void CudaContainer<T>::transferFromHost() {
  transferToDevice();
}

// specialize this for the T=float, float4, double, doule4, int etc
template <typename T>
__global__ void printKernel(const T *deviceArray, int size) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < size) {
    // printf("%d\n", deviceArray[index]);
  }
}

template <typename T> void CudaContainer<T>::printDevice() {
  int nthreads = 128;
  int nblocks = hostArray.size() / nthreads + 1;

  std::cout << "nblocks : " << nblocks << " nthreads : " << nthreads << "\n";
  printKernel<<<nblocks, nthreads>>>(deviceArray.data(), size());
  cudaDeviceSynchronize();
}
/*
template <typename T>
void CudaContainer<T>::printHost() {
  for (int i=0; i < size(); ++i){
    std::cout << hostArray[i] << "\n";
  }
}

template<>
void CudaContainer<float4>::printHost() {
  for (int i=0; i < size(); ++i){
    std::cout << hostArray[i].x  << " " <<
        hostArray[i].y  << " " <<  hostArray[i].z <<  " "  << hostArray[i].w <<
std::endl;
  }
}
*/
template <typename T> void CudaContainer<T>::set(const std::vector<T> &inpVec) {
  setHostArray(inpVec);
  transferToDevice();
}

template <typename T> T &CudaContainer<T>::operator[](size_t pos) {
  return hostArray[pos];
}

template <typename T> T &CudaContainer<T>::at(size_t pos) {
  if (pos > hostArray.size())
    throw std::out_of_range("Out of range");
  return hostArray[pos];
}

template <typename T> void CudaContainer<T>::setToValue(const T &inpVal) {
  for (auto &val : hostArray) {
    val = inpVal;
  }
  transferToDevice();
}