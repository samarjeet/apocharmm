// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#ifndef NOCUDAC
#ifndef FORCE_H
#define FORCE_H

#include "cuda_utils.h"
#include <cuda.h>

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
public:
  /**
   * @brief Base constructor. Sets evthg to 0/NULL.
   */
  Force(void);

  /**
   * @brief Constructor, takes size only to allocate space
   *
   * @param size
   */
  Force(const int size);

  /**
   * @brief Force object from file.
   *
   * @param filename
   */
  Force(const char *filename);

  ~Force(void);

  /**
   * @brief Given a cudaStream_t, clear the associated GPU array
   *
   * @param stream
   */
  void clear(cudaStream_t stream = 0);

  /**
   * @brief Compares two force arrays, returns true if the difference is within
   * tolerance
   *
   * NOTE: Comparison is done in double precision
   */
  bool compare(Force<T> &force, const double tol, double &max_diff);

public:
  /**
   * @brief Storage stride
   *
   * Stride = _size * nbytesperentry, size of all the x components of the stored
   * vector. Force::realloc recomputes _stride to align with 256 byte
   * boundaries.
   */
  int stride(void);
  int stride(void) const;

  /**
   * @brief Number of vectors stored
   */
  int size(void);
  int size(void) const;

  /**
   * @brief Pointer to the forces array
   */
  T *xyz(void);
  const T *xyz(void) const;

  /**
   * @brief Pointer to the array containing forces x components
   */
  T *x(void);
  const T *x(void) const;

  T *y(void);
  const T *y(void) const;

  T *z(void);
  const T *z(void) const;

  /**
   * @brief Gets forces to host
   */
  void getXYZ(T *h_x, T *h_y, T *h_z);

public:
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

public:
  /**
   * @brief Save to file
   */
  template <typename T2> void save(const char *filename);

public:
  /**
   * @brief Re-allocates array, does not preserve content.
   *
   * Recomputes _stride that aligns with 256 bytes boundaries.
   */
  void realloc(int size, float fac = 1.0f);

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
  int m_Size;
  int m_Stride;
  int m_Capacity;
  T *m_XYZ;
};

#endif // FORCE_H
#endif // NOCUDAC
