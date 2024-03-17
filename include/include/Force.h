// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#ifndef NOCUDAC
#ifndef FORCE_H
#define FORCE_H

#include "cuda_utils.h"
#include <cuda.h>
#include <iostream>

/**
 * @brief Storage class for force *values*. Should be replaced by
 * CudaContainer.
 * @todo Replace by CudaContainers
 * 
 *
 * The Force object contains the *values* of forces (as vectors of size 3)
 *
 * Contains _size force vectors, stored at _xyz.
 * Stride of the force data:
 * x data is in data[0...size-1];
 * y data is in data[stride...stride+size-1];
 * z data is in data[stride*2...stride*2+size-1];
 */
template <typename T> class Force {
private:
  // Stride of the force data:
  // x data is in data[0...size-1];
  // y data is in data[stride...stride+size-1];
  // z data is in data[stride*2...stride*2+size-1];
  // int stride;

  // Force data
  // int data_len;
  // T *data;

  /** @brief Number of force vectors stored */
  int _size;
  int _stride;
  int _capacity;
  T *_xyz;

  // cudaXYZ<T> xyz;

public:
  /**
   * @brief Base constructor. Sets evthg to 0/NULL.
   */
  Force() {
    _size = 0;
    _stride = 0;
    _capacity = 0;
    _xyz = NULL;
  }

  /**
   * @brief Constructor, takes size only to allocate space
   *
   *
   * @param size 
   */
  Force(const int size) {
    _size = 0;
    _stride = 0;
    _capacity = 0;
    _xyz = NULL;
    this->realloc(size);
  }

  /**
   * @brief Force object from file.
   */
  Force(const char *filename);

  ~Force() {
    if (_xyz != NULL)
      deallocate<T>(&_xyz);
  }

  /** @brief Given a cudaStream_t, clear the associated GPU array */
  void clear(cudaStream_t stream = 0) {
    clear_gpu_array<T>(this->_xyz, 3 * this->_stride, stream);
  }

  /**
   * @brief Compares two force arrays, returns true if the difference is within tolerance
   * 
   * NOTE: Comparison is done in double precision
   */
  bool compare(Force<T> &force, const double tol, double &max_diff);

  /**
   * @brief Re-allocates array, does not preserve content. 
   *
   * Recomputes _stride that aligns with 256 bytes boundaries.
   */
  void realloc(int size, float fac = 1.0f) {
    this->_size = size;
    // Returns stride that aligns with 256 byte boundaries
    this->_stride =
        (((size - 1 + 32) * sizeof(T) - 1) / 256 + 1) * 256 / sizeof(T);
    int new_capacity = (int)((double)(3 * this->_stride) * (double)fac);
    reallocate<T>(&this->_xyz, &this->_capacity, new_capacity, fac);
  }

  /**
   * @brief Storage stride 
   *
   * Stride = _size * nbytesperentry, size of all the x components of the stored vector.
   * Force::realloc recomputes _stride to align with 256 byte boundaries.
   */
  int stride() { return _stride; }
  /**
   * @brief Number of vectors stored
   */
  int size() { return _size; }

  /**
   * @brief Pointer to the forces array
   */
  T *xyz() { return _xyz; }
  /**
   * @brief Pointer to the array containing forces x components
   */
  T *x() { return &_xyz[0]; }
  T *y() { return &_xyz[_stride]; }
  T *z() { return &_xyz[_stride * 2]; }

 /**
  * @brief Gets forces to host
  */
  void getXYZ(T *h_x, T *h_y, T *h_z);

  /**
   * @brief Returns _stride
   */
  int stride() const { return _stride; }
  /**
   * @brief Returns _size
   */
  int size() const { return _size; }
  const T *xyz() const { return _xyz; }
  const T *x() const { return &_xyz[0]; }
  const T *y() const { return &_xyz[_stride]; }
  const T *z() const { return &_xyz[_stride * 2]; }

  template <typename T2>
  void convert(Force<T2> &force, cudaStream_t stream = 0);
  template <typename T2> void convert(cudaStream_t stream = 0);
  template <typename T2, typename T3>
  void convert_to(Force<T3> &force, cudaStream_t stream = 0);
  template <typename T2, typename T3>
  void convert_add(Force<T3> &force, cudaStream_t stream = 0);
  template <typename T2, typename T3>
  void add(Force<T3> &force, cudaStream_t stream = 0);
  template <typename T2>
  void add(float3 *force_data, int force_n, cudaStream_t stream = 0);
  void add(const Force<T> &force, cudaStream_t stream = 0);
  /**
   * @brief Save to file
   */
  template <typename T2> void save(const char *filename);
  void saveFloat(const char *fileName);
};

#endif // FORCE_H
#endif // NOCUDAC
