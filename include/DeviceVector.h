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

#include <cuda_runtime.h>
#include <iostream>
#include <stddef.h>

template <typename T> class DeviceVector {
public:
  DeviceVector(void);
  DeviceVector(const size_t size);
  DeviceVector(const DeviceVector<T> &other);
  DeviceVector(const DeviceVector<T> &&other);
  ~DeviceVector(void);

public:
  DeviceVector<T> &operator=(const DeviceVector<T> &other);
  DeviceVector<T> &operator=(const DeviceVector<T> &&other);

public:
  void resize(const size_t size);
  void deallocate(void);
  const T *data(void) const;
  T *data(void);
  void set(T *data);
  size_t size(void) const;

private:
  size_t m_Size;
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
template class DeviceVector<double>;
template class DeviceVector<double2>;
template class DeviceVector<double3>;
template class DeviceVector<double4>;
