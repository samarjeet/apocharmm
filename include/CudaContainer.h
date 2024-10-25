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
public: // Member functions
  /**
   * @brief Construct a new Cuda Container object
   *
   */
  CudaContainer(void);

  CudaContainer(const std::size_t count);

  /**
   * @brief Construct a new Cuda Container object with the same contents as the
   * vector passed
   *
   * @param const std::vector<T> &
   */
  CudaContainer(const std::vector<T> &other);

  /**
   * @brief Construct a new Cuda Container object with the same contents as the
   * vector passed
   *
   * @param const DeviceVector<T> &
   */
  CudaContainer(const DeviceVector<T> &other);

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

  CudaContainer<T> &operator=(const std::vector<T> &other);
  CudaContainer<T> &operator=(const DeviceVector<T> &other);
  CudaContainer<T> &operator=(const CudaContainer<T> &other);
  CudaContainer<T> &operator=(const CudaContainer<T> &&other);

public: // Element access
  /**
   * @brief Returns a constant reference to the element at specified location
   * pos, with bounds checking. If pos is not within the range of the container,
   * an exception of type std::out_of_range is thrown.
   *
   * @param[in] pos
   */
  const T &at(const std::size_t pos) const;

  /**
   * @brief Returns a reference to the element at specified location
   * pos, with bounds checking. If pos is not within the range of the container,
   * an exception of type std::out_of_range is thrown.
   *
   * @param[in] pos
   */
  T &at(const std::size_t pos);

  /**
   * @brief Returns a constant reference to the element at specified location
   * pos. No bounds checking is performed.
   *
   * @param[in] pos
   */
  const T &operator[](const std::size_t pos) const;

  /**
   * @brief Returns a reference to the element at specified location pos. No
   * bounds checking is performed.
   *
   * @param[in] pos
   */
  T &operator[](const std::size_t pos);

  const std::vector<T> &getHostArray(void) const;
  std::vector<T> &getHostArray(void);

  const DeviceVector<T> &getDeviceArray(void) const;
  DeviceVector<T> &getDeviceArray(void);

  const T *getHostData(void) const;
  T *getHostData(void);

  const T *getDeviceData(void) const;
  T *getDeviceData(void);

public: // Capacity
  /**
   * @brief Size of the container
   *
   * @return size_t
   */
  std::size_t size(void) const;

public: // Modifiers
  void clear(void);
  void resize(const std::size_t count);

  /**
   * @brief Sets the host array and transfers it to device
   *
   * @param[in] values passed as reference
   */
  void set(const std::vector<T> &values);

  /**
   * @brief Sets the device array and transfers it to host
   *
   * @param[in] values passed as reference
   */
  void set(const DeviceVector<T> &values);

  /**
   * @brief Set all values of the host array to the input value, transfer to
   * device.
   */
  void setToValue(const T value);

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
template class CudaContainer<long long int>;
template class CudaContainer<longlong2>;
template class CudaContainer<longlong3>;
template class CudaContainer<longlong4>;
template class CudaContainer<std::size_t>;
template class CudaContainer<double>;
template class CudaContainer<double2>;
template class CudaContainer<double3>;
template class CudaContainer<double4>;
