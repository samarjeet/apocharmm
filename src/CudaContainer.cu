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
CudaContainer<T>::CudaContainer(const std::size_t count)
    : m_HostArray(count), m_DeviceArray(count) {}

template <typename T>
CudaContainer<T>::CudaContainer(const std::vector<T> &other)
    : m_HostArray(other), m_DeviceArray(other.size()) {
  this->transferToDevice();
}

template <typename T>
CudaContainer<T>::CudaContainer(const DeviceVector<T> &other)
    : m_HostArray(other.size()), m_DeviceArray(other) {
  this->transferToHost();
}

template <typename T>
CudaContainer<T>::CudaContainer(const CudaContainer<T> &other)
    : m_HostArray(other.getHostArray()), m_DeviceArray(other.getDeviceArray()) {
}

template <typename T>
CudaContainer<T>::CudaContainer(const CudaContainer<T> &&other)
    : m_HostArray(other.getHostArray()), m_DeviceArray(other.getDeviceArray()) {
}

template <typename T>
CudaContainer<T> &CudaContainer<T>::operator=(const std::vector<T> &other) {
  m_HostArray = other;
  m_DeviceArray.resize(other.size());
  this->transferToDevice();
  return *this;
}

template <typename T>
CudaContainer<T> &CudaContainer<T>::operator=(const DeviceVector<T> &other) {
  m_DeviceArray = other;
  m_HostArray.resize(other.size());
  this->transferToHost();
  return *this;
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

template <typename T>
const T &CudaContainer<T>::at(const std::size_t pos) const {
  return m_HostArray.at(pos);
}

template <typename T> T &CudaContainer<T>::at(const std::size_t pos) {
  return m_HostArray.at(pos);
}

template <typename T>
const T &CudaContainer<T>::operator[](const std::size_t pos) const {
  return m_HostArray[pos];
}

template <typename T> T &CudaContainer<T>::operator[](const std::size_t pos) {
  return m_HostArray[pos];
}

template <typename T>
const std::vector<T> &CudaContainer<T>::getHostArray(void) const {
  return m_HostArray;
}

template <typename T> std::vector<T> &CudaContainer<T>::getHostArray(void) {
  return m_HostArray;
}

template <typename T>
const DeviceVector<T> &CudaContainer<T>::getDeviceArray(void) const {
  return m_DeviceArray;
}

template <typename T> DeviceVector<T> &CudaContainer<T>::getDeviceArray(void) {
  return m_DeviceArray;
}

template <typename T> const T *CudaContainer<T>::getHostData(void) const {
  return m_HostArray.data();
}

template <typename T> T *CudaContainer<T>::getHostData(void) {
  return m_HostArray.data();
}

template <typename T> const T *CudaContainer<T>::getDeviceData(void) const {
  return m_DeviceArray.data();
}

template <typename T> T *CudaContainer<T>::getDeviceData(void) {
  return m_DeviceArray.data();
}

template <typename T> std::size_t CudaContainer<T>::size(void) const {
  return m_HostArray.size();
}

template <typename T> void CudaContainer<T>::clear(void) {
  m_HostArray.clear();
  m_DeviceArray.clear();
  return;
}

template <typename T> void CudaContainer<T>::resize(const std::size_t count) {
  m_HostArray.resize(count);
  m_DeviceArray.resize(count);
  return;
}

template <typename T> void CudaContainer<T>::set(const std::vector<T> &values) {
  m_HostArray = values;
  this->transferToDevice();
  return;
}

template <typename T>
void CudaContainer<T>::set(const DeviceVector<T> &values) {
  m_DeviceArray = values;
  this->transferToHost();
  return;
}

template <typename T> void CudaContainer<T>::setToValue(const T value) {
  m_HostArray.assign(m_HostArray.size(), value);
  this->transferToDevice();
  return;
}

template <typename T> void CudaContainer<T>::transferToDevice(void) {
  cudaCheck(cudaMemcpy(static_cast<void *>(m_DeviceArray.data()),
                       static_cast<const void *>(m_HostArray.data()),
                       m_HostArray.size() * sizeof(T), cudaMemcpyHostToDevice));
  cudaCheck(cudaDeviceSynchronize());
  return;
}

template <typename T> void CudaContainer<T>::transferToHost(void) {
  cudaCheck(cudaMemcpy(static_cast<void *>(m_HostArray.data()),
                       static_cast<const void *>(m_DeviceArray.data()),
                       m_DeviceArray.size() * sizeof(T),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaDeviceSynchronize());
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

__global__ void printKernel(const long long int *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %lld\n", index, data[index]);
  return;
}

__global__ void printKernel(const longlong2 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %lld, %lld\n", index, data[index].x, data[index].y);
  return;
}

__global__ void printKernel(const longlong3 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %lld, %lld, %lld\n", index, data[index].x, data[index].y,
           data[index].z);
  return;
}

__global__ void printKernel(const longlong4 *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %lld, %lld, %lld, %lld\n", index, data[index].x, data[index].y,
           data[index].z, data[index].w);
  return;
}

__global__ void printKernel(const std::size_t *data, const size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %llu\n", index, data[index]);
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
  const unsigned int blockDim = 256;
  const unsigned int gridDim = (this->size() + blockDim - 1) / blockDim;
  printKernel<<<gridDim, blockDim>>>(m_DeviceArray.data(), this->size());
  cudaCheck(cudaDeviceSynchronize());
  return;
}
