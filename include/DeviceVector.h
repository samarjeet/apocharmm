// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Andrew Simmonett, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>

template <typename T> class DeviceVector {
public: // Member functions
  DeviceVector(void);
  DeviceVector(const std::size_t count);
  DeviceVector(const DeviceVector<T> &other);
  DeviceVector(const DeviceVector<T> &&other);

  ~DeviceVector(void);

  DeviceVector<T> &operator=(const DeviceVector<T> &other);
  DeviceVector<T> &operator=(const DeviceVector<T> &&other);

public: // Element access
  const T *data(void) const;
  T *data(void);

  void assignData(T *data);

public: // Capacity
  bool empty(void) const;
  std::size_t size(void) const;
  std::size_t capacity(void) const;
  void shrink_to_fit(void);

public: // Modifiers
  void clear(void);
  // void push_back(const T &value);
  void resize(const std::size_t count);
  void swap(DeviceVector<T> &other);

private:
  void allocate(const std::size_t count);
  void reallocate(const std::size_t count);
  void deallocate(void);

private:
  std::size_t m_Size;
  std::size_t m_Capacity;
  T *m_Data;
};

template class DeviceVector<int>;
template class DeviceVector<int2>;
template class DeviceVector<int3>;
template class DeviceVector<int4>;
template class DeviceVector<unsigned int>;
template class DeviceVector<float>;
template class DeviceVector<float2>;
template class DeviceVector<float3>;
template class DeviceVector<float4>;
template class DeviceVector<long long int>;
template class DeviceVector<longlong2>;
template class DeviceVector<longlong3>;
template class DeviceVector<longlong4>;
template class DeviceVector<std::size_t>;
template class DeviceVector<double>;
template class DeviceVector<double2>;
template class DeviceVector<double3>;
template class DeviceVector<double4>;
