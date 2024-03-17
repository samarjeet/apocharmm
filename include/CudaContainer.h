// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Andrew Simmonett, Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include "deviceVector.h"
#include <stdexcept>
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
  //  /**
  //   * @brief Construct a new Cuda Container object
  //   *
  //   */
  //  // CudaContainer();
  //~CudaContainer();

  /**
   * @brief Size of the container
   *
   * @return size_t
   */
  size_t size() const { return hostArray.size(); }

  /**
   * @brief Allocate memory on host and device
   *
   * @param size
   */
  void allocate(size_t size);

  /**
   * @brief Set the Host Array object
   *
   * @param array
   */
  void setHostArray(const std::vector<T> &array);

  /**
   * @brief Set the deviceArray with the `same` pointer
   * Use with caution!
   *
   * @param devPtr
   */
  void setDeviceArray(T *devPtr);

  // remove this one
  const std::vector<T> &getHostArray();
  // const deviceVector<T>& getDeviceArray(){return deviceArray;};
  deviceVector<T> &getDeviceArray() { return deviceArray; };

  void transferToDevice();
  void transferFromDevice();

  void transferToHost();
  void transferFromHost();

  void printDevice();
  // void printHost();

  /**
   * @brief Sets the host array and transfers it to device
   *
   * @param[in] inpVec passed as reference
   */
  void set(const std::vector<T> &inpVec);

  // Returns a reference to the element at specified location pos.
  // No bounds checking is performed.
  T &operator[](size_t pos);

  // Returns a reference to the element at specified location pos, with bounds
  // checking. If pos is not within the range of the container, an exception of
  // type std::out_of_range is thrown.
  T &at(size_t pos);

  /** @brief Set all values of the host array to the input value, transfer to
   * device. */
  void setToValue(const T &inpVal);

  // TODO : enable a range-based iterator
private:
  std::vector<T> hostArray;
  // T* deviceArray;
  deviceVector<T> deviceArray;
};

template class CudaContainer<float>;
template class CudaContainer<double>;
template class CudaContainer<float4>;
template class CudaContainer<double4>;
template class CudaContainer<int>;
template class CudaContainer<int2>;
template class CudaContainer<int4>;
