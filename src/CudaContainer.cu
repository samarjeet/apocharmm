// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CudaContainer.h"
#include "cuda_utils.h"

#include <stdexcept>

template <typename T>
CudaContainer<T>::CudaContainer(void) : m_HostArray(), m_DeviceArray() {}

template <typename T>
CudaContainer<T>::CudaContainer(const std::vector<T> &array) : CudaContainer() {
  this->set(array);
}

template <typename T>
CudaContainer<T>::CudaContainer(const CudaContainer<T> &other)
    : CudaContainer() {
  m_HostArray = std::vector<T>(other.getHostArray().cbegin(),
                               other.getHostArray().cend());
  m_DeviceArray.resize(other.size());
  cudaCheck(cudaMemcpy(static_cast<void *>(m_DeviceArray.data()),
                       static_cast<const void *>(other.getDeviceArray().data()),
                       this->size() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
CudaContainer<T>::CudaContainer(const CudaContainer<T> &&other)
    : CudaContainer() {
  m_HostArray = std::vector<T>(other.getHostArray().cbegin(),
                               other.getHostArray().cend());
  m_DeviceArray.resize(other.size());
  cudaCheck(cudaMemcpy(static_cast<void *>(m_DeviceArray.data()),
                       static_cast<const void *>(other.getDeviceArray().data()),
                       this->size() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T> void CudaContainer<T>::allocate(const size_t size) {
  m_HostArray = std::vector<T>(size);
  m_DeviceArray.resize(size);
  return;
}

template <typename T> size_t CudaContainer<T>::size(void) const {
  return m_HostArray.size();
}

template <typename T>
void CudaContainer<T>::setHostArray(const std::vector<T> &array) {
  m_HostArray = std::vector<T>(array.cbegin(), array.cend());
  return;
}

template <typename T> void CudaContainer<T>::setDeviceArray(T *devPtr) {
  m_DeviceArray.set(devPtr);
  return;
}

template <typename T>
const std::vector<T> &CudaContainer<T>::getHostArray(void) const {
  return m_HostArray;
}

template <typename T>
const DeviceVector<T> &CudaContainer<T>::getDeviceArray(void) const {
  return m_DeviceArray;
}

template <typename T> std::vector<T> &CudaContainer<T>::getHostArray(void) {
  return m_HostArray;
}

template <typename T> DeviceVector<T> &CudaContainer<T>::getDeviceArray(void) {
  return m_DeviceArray;
}

template <typename T> void CudaContainer<T>::set(const std::vector<T> &array) {
  this->setHostArray(array);
  this->transferToDevice();
  return;
}

template <typename T> void CudaContainer<T>::setToValue(const T &val) {
  m_HostArray.assign(this->size(), val);
  this->transferToDevice();
  return;
}

template <typename T>
const T &CudaContainer<T>::operator[](const size_t pos) const {
  return m_HostArray[pos];
}

template <typename T> const T &CudaContainer<T>::at(const size_t pos) const {
  if (pos > m_HostArray.size())
    throw std::out_of_range("Out of range");
  return m_HostArray[pos];
}

template <typename T>
CudaContainer<T> &CudaContainer<T>::operator=(const CudaContainer<T> &other) {
  m_HostArray = other.getHostArray();
  m_DeviceArray = other.getDeviceArray();
  return *this;
}

template <typename T>
CudaContainer<T> &CudaContainer<T>::operator=(const CudaContainer<T> &&other) {
  m_HostArray = other.getHostArray();
  m_DeviceArray = other.getDeviceArray();
  return *this;
}

template <typename T> T &CudaContainer<T>::operator[](const size_t pos) {
  return m_HostArray[pos];
}

template <typename T> T &CudaContainer<T>::at(const size_t pos) {
  if (pos > m_HostArray.size())
    throw std::out_of_range("Out of range");
  return m_HostArray[pos];
}

template <typename T> void CudaContainer<T>::transferToDevice(void) {
  m_DeviceArray.resize(m_HostArray.size());
  cudaCheck(cudaMemcpy(static_cast<void *>(m_DeviceArray.data()),
                       static_cast<const void *>(m_HostArray.data()),
                       this->size() * sizeof(T), cudaMemcpyHostToDevice));
  cudaCheck(cudaDeviceSynchronize());
  return;
}

template <typename T> void CudaContainer<T>::transferToHost(void) {
  cudaCheck(cudaMemcpy(static_cast<void *>(m_HostArray.data()),
                       static_cast<const void *>(m_DeviceArray.data()),
                       this->size() * sizeof(T), cudaMemcpyDeviceToHost));
  return;
}

template <typename T> void CudaContainer<T>::transferFromDevice(void) {
  this->transferToHost();
  return;
}

template <typename T> void CudaContainer<T>::transferFromHost(void) {
  this->transferToDevice();
  return;
}

__global__ void printKernel(const int *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %d\n", index, data[index]);
  return;
}

__global__ void printKernel(const int2 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %d, %d\n", index, data[index].x, data[index].y);
  return;
}

__global__ void printKernel(const int3 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %d, %d, %d\n", index, data[index].x, data[index].y,
           data[index].z);
  return;
}

__global__ void printKernel(const int4 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %d, %d, %d, %d\n", index, data[index].x, data[index].y,
           data[index].z, data[index].w);
  return;
}

__global__ void printKernel(const unsigned int *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %u\n", index, data[index]);
  return;
}

__global__ void printKernel(const float *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e\n", index, data[index]);
  return;
}

__global__ void printKernel(const float2 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e, %e\n", index, data[index].x, data[index].y);
  return;
}

__global__ void printKernel(const float3 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e, %e, %e\n", index, data[index].x, data[index].y,
           data[index].z);
  return;
}

__global__ void printKernel(const float4 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e, %e, %e, %e\n", index, data[index].x, data[index].y,
           data[index].z, data[index].w);
  return;
}

__global__ void printKernel(const double *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e\n", index, data[index]);
  return;
}

__global__ void printKernel(const double2 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e, %e\n", index, data[index].x, data[index].y);
  return;
}

__global__ void printKernel(const double3 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e, %e, %e\n", index, data[index].x, data[index].y,
           data[index].z);
  return;
}

__global__ void printKernel(const double4 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e, %e, %e, %e\n", index, data[index].x, data[index].y,
           data[index].z, data[index].w);
  return;
}

template <typename T> void CudaContainer<T>::printDeviceArray(void) const {
  const unsigned int blockDim = 32;
  const unsigned int gridDim = (this->size() + blockDim - 1) / blockDim;
  printKernel<<<gridDim, blockDim>>>(m_DeviceArray.data(), this->size());
  cudaCheck(cudaDeviceSynchronize());
  return;
}
