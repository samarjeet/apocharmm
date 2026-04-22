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
#ifndef XYZQ_H
#define XYZQ_H

//
// XYZQ class
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
//
#include "CudaContainer.h"
#include "cudaXYZ.h"
// #include "gpu_utils.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

static const int warpsize_local = 32;

  /**
   * @brief GPU-adapted simple precision storage of position and charge
   *
   * Each entry contains x, y, z position and charge value for a given atom.
   * Everything is kept in simple precision, as these quantities are used for
   * force computation.
   * Double-precision version is kept in CudaContainer objects.
   *
   * @deprecated In the future, XYZQ objects should be replaced by
   * CudaContainer objects, which are templates and thus flexible in precision.
   *
   */
class XYZQ {
private:
  /**
   * @brief Get optimal size for "GPU storage"
   */
  int get_xyzq_len(const int ncoord_in);

public:
  // Members - should probably be private ?
  // ======================================
  /**
   * @brief GPU memory size. 
   *
   * By default, will be set to warpsize_local=32.
   */
  int align;
  /**
   * @brief Number of coordinates to store
   */
  int ncoord;
  /**
   * @brief Size required to properly vectorize
   */
  int xyzq_len;

  /** @brief Pointer to self ? */
  float4 *xyzq;

  /**
   * @brief (shared) pointer to data on Host
   */
  std::shared_ptr<std::vector<float4>> host_xyzq;

  // Constructors
  //=============
  /**
   * @brief Base class creator. Everything 0 or NULL.
   */
  XYZQ();
  /**
   * @brief Class creator. Allocates XYZQ, resizes host_XYZQ.
   */
  XYZQ(int ncoord, int align = warpsize_local);
  /**
   * @brief File based class creator.
   *
   * Reads coordinates from basic txt file, containing on each line x, y, z and
   * charge. Allocates XYZQ, fills in with parsed values.
   *
   */
  XYZQ(const char *filename, int align = warpsize_local);
  /**
   * @brief Destructor. Deallocates if not NULL.
   */
  ~XYZQ();

  // Methods
  // ========

  /**
   * @brief Change ncoord (reallocates xyzq, recreates host_xyzq)
   */
  void set_ncoord(const int ncrd);
  /**
   * @brief Reallocates array, does not preserve content
   */
  void realloc(int ncoord_new, float fac = 1.0f);
  /** 
   * @brief Re-sizes array, preserves content
   */
  void resize(int ncoord_new, float fac = 1.0f);

  /**
   * @brief Copies xyzq from host
   *
   * NOTE: Does not reallocate xyzq
   *
   * @param[in] ncopy Length of array to copy
   * @param[in] h_xyzq Pointer to host XYZQ
   * @param[in] offset First entry to copy
   * @param[in] stream cudaStream_t to use
   */
  void set_xyzq(int ncopy, const float4 *h_xyzq, size_t offset = 0,
                cudaStream_t stream = 0);

  void set_xyzq(const cudaXYZ<double> &coord, const float *q,
                cudaStream_t stream = 0);
  void set_xyzq(const cudaXYZ<double> &coord, const float *q,
                const int *loc2glo, const float3 *xyz_shift, const double boxx,
                const double boxy, const double boxz, cudaStream_t stream = 0);

  /** @brief Copies x,y,z (on device) into the coordinate slots */
  void set_xyz(const cudaXYZ<double> &coord, cudaStream_t stream = 0);
  /** @brief Copies x,y,z (on device) into coord slots with a shift */
  void set_xyz(const cudaXYZ<double> &coord, const int start, const int end,
               const float3 *xyz_shift, const double boxx, const double boxy,
               const double boxz, cudaStream_t stream = 0);
  /** @brief Sets x,y,z from a float vector into host xyzq */
  void set_xyz(const std::vector<float> &coords);

  /** @brief Compares two XYZQ arrays 
   *
   * Compares each entry (x,y,z and q) for each atom of the input XYZQ with the current one. 
   */
  bool compare(XYZQ &xyzq_in, const double tol, double &max_diff);

  /** @brief Print to ostream */
  void print(const int start, const int end, std::ostream &out);
  /** @brief Save to file */
  void save(const char *filename);

  /** @brief Returns pointer to Device xyzq  */
  float4 *getDeviceXYZQ();

  /** @brief Copy xyzq content from GPU to Host (cudaMemcpy) 
   *  
   * Copies xyzq into host_xyzq
   */
  void transferFromDevice();
  /** @brief Copy host_xyzq content from Host to Device
   * (cudaMemcpyHostToDevice)
   *
   * Copies host_xyzq into xyzq (on device)
   */
  void transferToDevice();

  /** @brief Transfrers xyzq data from device, returns pointer to newly copied
   * data on host 
   *
   * Calls transferFromDevice, returns host_xyzq
   */
  std::shared_ptr<std::vector<float4>> getHostXYZQ();
  void setDeviceXYZQ(float4 *in);

  /** @brief Returns float vector of **positions only**, copied from GPU
   *
   * Uses transferFromDevice, then extracts x,y and z components from host_xyzq
   */
  std::vector<float> get_xyz();

  /** @brief Returns float vector of **charges only**, copied from GPU
   *
   * Similar to get_xyz(), but only for the charges (debug tool)
   */
  std::vector<float> get_q();

  void imageCenter(const std::vector<float> &boxSize,
                   CudaContainer<int2> &groups);
  void orient(bool massWeighting = true, bool rotation = true);
};

#endif // XYZQ_H

#endif // NOCUDAC
