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
#include "DeviceVector.h"

#include <vector>

/**
 * @brief Templated container for host/device vectors
 *
 * @tparam T
 *
 * Right now, double-precision vectors containing positions and charges are
 * stored as CudaContainer, but single-precision ones are using the
 * (deprecated) XYZQ class.
 *
 * @todo Remove the XYZQ class, replace it with instances of CudaContainers
 * using single precision
 */
template <typename T> class CudaContainer {
public:
  /**
   * @brief Construct a new Cuda Container object
   *
   */
  CudaContainer(void);

  /**
   * @brief Construct a new Cuda Container object with the same contents as the
   * vector passed
   *
   * @param const std::vector<T> &
   */
  CudaContainer(const std::vector<T> &array);

  /**
   * @brief Copy constructor for new Cuda Container object
   *
   * @param const CudaContainer<T> &
   */
  CudaContainer(const CudaContainer<T> &other);

  /**
   * @brief Copy constructor for new Cuda Container object using an rvalue
   *
   * @param const CudaContainer<T> &&
   */
  CudaContainer(const CudaContainer<T> &&other);

public:
  /**
   * @brief Allocate memory on host and device
   *
   * @param size
   */
  void allocate(size_t size);

  /**
   * @brief Size of the container
   *
   * @return size_t
   */
  size_t size(void) const;

public:
  /**
   * @brief Set the Host Array object
   *
   * @param const std::vector<T> &
   */
  void setHostArray(const std::vector<T> &array);

  /**
   * @brief Set the deviceArray with the `same` pointer
   * Use with caution!
   *
   * @param T*
   */
  void setDeviceArray(T *devPtr);

  const std::vector<T> &getHostArray(void) const;
  const DeviceVector<T> &getDeviceArray(void) const;

  std::vector<T> &getHostArray(void);
  DeviceVector<T> &getDeviceArray(void);

  /**
   * @brief Sets the host array and transfers it to device
   *
   * @param[in] inpVec passed as reference
   */
  void set(const std::vector<T> &inpVec);

  /**
   * @brief Set all values of the host array to the input value, transfer to
   * device.
   */
  void setToValue(const T &inpVal);

  CudaContainer<T> &operator=(const CudaContainer<T> &other);
  CudaContainer<T> &operator=(const CudaContainer<T> &&other);

  // Returns a reference to the element at specified location pos.
  // No bounds checking is performed.
  const T &operator[](const size_t pos) const;

  // Returns a reference to the element at specified location pos, with bounds
  // checking. If pos is not within the range of the container, an exception of
  // type std::out_of_range is thrown.
  const T &at(const size_t pos) const;

  // Returns a reference to the element at specified location pos.
  // No bounds checking is performed.
  T &operator[](const size_t pos);

  // Returns a reference to the element at specified location pos, with bounds
  // checking. If pos is not within the range of the container, an exception of
  // type std::out_of_range is thrown.
  T &at(const size_t pos);

  void transferToDevice(void);
  void transferToHost(void);
  void transferFromDevice(void);
  void transferFromHost(void);

public:
  void printDeviceArray(void) const;

  // TODO : enable a range-based iterator
private:
  std::vector<T> m_HostArray;
  DeviceVector<T> m_DeviceArray;
};

template class CudaContainer<int>;
template class CudaContainer<int2>;
template class CudaContainer<int3>;
template class CudaContainer<int4>;
template class CudaContainer<unsigned int>;
template class CudaContainer<float>;
template class CudaContainer<float2>;
template class CudaContainer<float3>;
template class CudaContainer<float4>;
template class CudaContainer<double>;
template class CudaContainer<double2>;
template class CudaContainer<double3>;
template class CudaContainer<double4>;
