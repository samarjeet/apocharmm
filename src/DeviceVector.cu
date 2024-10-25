// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Andrew Simmonett, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "DeviceVector.h"

#include "cuda_utils.h"

template <typename T>
DeviceVector<T>::DeviceVector(void)
    : m_Size(0), m_Capacity(0), m_Data(nullptr) {}

template <typename T>
DeviceVector<T>::DeviceVector(const std::size_t count) : DeviceVector() {
  this->allocate(count);
  m_Size = count;
}

template <typename T>
DeviceVector<T>::DeviceVector(const DeviceVector<T> &other)
    : DeviceVector(other.size()) {
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
DeviceVector<T>::DeviceVector(const DeviceVector<T> &&other)
    : DeviceVector(other.size()) {
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T> DeviceVector<T>::~DeviceVector(void) {
  this->deallocate();
}

template <typename T>
DeviceVector<T> &DeviceVector<T>::operator=(const DeviceVector<T> &other) {
  this->reallocate(other.capacity());
  m_Size = other.size();
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));
  return *this;
}

template <typename T>
DeviceVector<T> &DeviceVector<T>::operator=(const DeviceVector<T> &&other) {
  this->reallocate(other.capacity());
  m_Size = other.size();
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));
  return *this;
}

template <typename T> const T *DeviceVector<T>::data(void) const {
  return m_Data;
}

template <typename T> T *DeviceVector<T>::data(void) { return m_Data; }

template <typename T> bool DeviceVector<T>::empty(void) const {
  return (m_Size == 0);
}

template <typename T> std::size_t DeviceVector<T>::size(void) const {
  return m_Size;
}

template <typename T> std::size_t DeviceVector<T>::capacity(void) const {
  return m_Capacity;
}

template <typename T> void DeviceVector<T>::shrink_to_fit(void) {
  this->reallocate(m_Size);
  return;
}

template <typename T> void DeviceVector<T>::clear(void) {
  this->deallocate();
  return;
}

// template <typename T> void DeviceVector<T>::push_back(const T &value) {
//   if (m_Size >= m_Capacity) // Increase size of memory block by 50%
//     this->reallocate(m_Capacity + m_Capacity / 2);
//   return;
// }

template <typename T> void DeviceVector<T>::resize(const std::size_t count) {
  if (m_Capacity == 0)
    this->allocate(count);
  else if (count > m_Capacity)
    this->reallocate(count);
  m_Size = count;
  return;
}

template <typename T> void DeviceVector<T>::swap(DeviceVector<T> &other) {
  // Copy properties and data of this DeviceVector
  std::size_t size = m_Size;
  std::size_t capacity = m_Capacity;
  T *data = nullptr;
  cudaCheck(
      cudaMalloc(reinterpret_cast<void **>(&data), m_Capacity * sizeof(T)));
  cudaCheck(cudaMemcpy(static_cast<void *>(data),
                       static_cast<const void *>(m_Data),
                       m_Capacity * sizeof(T), cudaMemcpyDeviceToDevice));

  // Copy properties and data from other DeviceVector to this DeviceVector
  this->reallocate(other.capacity());
  m_Size = other.size();
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));

  // Copy properties and data from copy of this DeviceVector to other
  // DeviceVector
  other.reallocate(capacity);
  other.resize(size);
  cudaCheck(cudaMemcpy(static_cast<void *>(other.data()),
                       static_cast<const void *>(data), size * sizeof(T),
                       cudaMemcpyDeviceToDevice));

  // Free temporary memory used for copying data
  cudaCheck(cudaFree(static_cast<void *>(data)));

  return;
}

template <typename T> void DeviceVector<T>::allocate(const std::size_t count) {
  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&m_Data), count * sizeof(T)));
  m_Capacity = count;
  return;
}

template <typename T>
void DeviceVector<T>::reallocate(const std::size_t count) {
  if (count == m_Capacity) // No need for a new memory block
    return;

  // Allocate new memory block
  std::size_t oldSize = m_Size;
  T *data = nullptr;
  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));

  // Copy relevant data to new memory block
  cudaCheck(cudaMemcpy(static_cast<void *>(data),
                       static_cast<const void *>(m_Data),
                       ((count < m_Size) ? count : m_Size) * sizeof(T),
                       cudaMemcpyDeviceToDevice));

  // Free old memory block
  this->deallocate();

  // Assign new memory block
  m_Size = (count < oldSize) ? count : oldSize;
  m_Capacity = count;
  m_Data = data;

  return;
}

template <typename T> void DeviceVector<T>::deallocate(void) {
  m_Size = 0;
  m_Capacity = 0;
  if (m_Data != nullptr) {
    cudaCheck(cudaFree(static_cast<void *>(m_Data)));
    m_Data = nullptr;
  }
  return;
}
