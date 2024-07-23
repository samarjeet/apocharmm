// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Andrew Simmonett, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "cuda_utils.h"

#include "DeviceVector.h"

template <typename T>
DeviceVector<T>::DeviceVector(void) : m_Size(0), m_Data(nullptr) {}

template <typename T>
DeviceVector<T>::DeviceVector(const size_t size) : DeviceVector() {
  this->resize(size);
}

template <typename T>
DeviceVector<T>::DeviceVector(const DeviceVector<T> &other)
    : DeviceVector(other.size()) {
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       m_Size * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
DeviceVector<T>::DeviceVector(const DeviceVector<T> &&other)
    : DeviceVector(other.size()) {
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       m_Size * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T> DeviceVector<T>::~DeviceVector(void) {
  this->deallocate();
}

template <typename T>
DeviceVector<T> &DeviceVector<T>::operator=(const DeviceVector<T> &other) {
  this->resize(other.size());
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       m_Size * sizeof(T), cudaMemcpyDeviceToDevice));
  return *this;
}

template <typename T>
DeviceVector<T> &DeviceVector<T>::operator=(const DeviceVector<T> &&other) {
  this->resize(other.size());
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       m_Size * sizeof(T), cudaMemcpyDeviceToDevice));
  return *this;
}

template <typename T> void DeviceVector<T>::resize(const size_t size) {
  if (m_Size < size) {
    T *data = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void **>(&data), size * sizeof(T)));

    if (m_Size > 0) {
      cudaCheck(cudaMemcpy(static_cast<void *>(data),
                           static_cast<const void *>(m_Data),
                           m_Size * sizeof(T), cudaMemcpyDeviceToDevice));
      cudaCheck(cudaFree(static_cast<void *>(m_Data)));
    }

    m_Data = data;
    m_Size = size;
  } else if (m_Size > size) {
    m_Size = size;
  }

  return;
}

template <typename T> void DeviceVector<T>::deallocate(void) {
  m_Size = 0;
  if (m_Data != nullptr) {
    cudaCheck(cudaFree(static_cast<void *>(m_Data)));
    m_Data = nullptr;
  }
  return;
}

template <typename T> const T *DeviceVector<T>::data(void) const {
  return m_Data;
}

template <typename T> T *DeviceVector<T>::data(void) { return m_Data; }

template <typename T> void DeviceVector<T>::set(T *data) {
  m_Data = data;
  return;
}

template <typename T> size_t DeviceVector<T>::size(void) const {
  return m_Size;
}
