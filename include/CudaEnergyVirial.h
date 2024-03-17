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
#ifndef CUDAENERGYVIRIAL_H
#define CUDAENERGYVIRIAL_H
//
// Storage class for direct-space energies and virials in CUDA
// (c) Antti-Pekka Hynninen, February 2015
// aphynninen@hotmail.com
//
#include "CudaContainer.h"
#include "EnergyVirial.h"
#include "cuda_utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Structure into which virials are stored
// NOTE: This structure is only used for computing addresses
struct Virial_t {
  double sforce_dp[27][3];
  long long int sforce_fp[27][3];
  double virmat[9];
  // Energies start here ...
};



/** @brief Storage class for energy terms and virial for a force group
 *
 * Energy and virial terms are mapped with their names (e.g, \c
 * getEnergy("bond") returns the bond energy, \c getEnergyPointer("angle")
 * returns a device pointer to the angular energy)
 */
class CudaEnergyVirial : public EnergyVirial {
private:
  // Host and device arrays for storing energies and sforce -arrays
  int h_buffer_len;
  char *h_buffer;

  int d_buffer_len;
  char *d_buffer;

  void reallocateBuffer();

public:
  /** @brief Base constructor */
  CudaEnergyVirial();
  ~CudaEnergyVirial();

  /** @brief Clears (sets to zero) energies and virials */
  void clear(cudaStream_t stream = 0);
  /** @brief  Clears (sets to zero) energies */
  void clearEnergy(cudaStream_t stream = 0);
  /** @brief Clears (sets to zero) a specified energy 
   *
   * @param[in] &nameEterm Name of energy term to be cleared
   * @param[in] stream cudaStream_t
   */
  void clearEtermDevice(std::string &nameEterm, cudaStream_t stream = 0);
  /** @brief Clears (sets to zero) virials */
  void clearVirial(cudaStream_t stream = 0);
  /** @brief Calculates virial 
   * @todo put this somewhere else (ForceManager for example)
   * @deprecated We should not be computing things using this class! 
   */
  void calcVirial(const int ncoord, const float4 *xyzq, const double boxx,
                  const double boxy, const double boxz, const int stride,
                  const double *force, cudaStream_t stream = 0);
  /** @brief  Copies energy and virial values to host */
  void copyToHost(cudaStream_t stream = 0);


  /** @brief  Return device pointer to the Virial_t -structure */
  Virial_t *getVirialPointer();
  /** @brief Returns device pointer to energy term "name" */
  double *getEnergyPointer(std::string &name);
  /** @brief Returns device pointer to energy term "name" */
  double *getEnergyPointer(const char *name);

  /** @brief  Return value of energy called "name" */
  double getEnergy(std::string &name);
  /** @brief  Return value of energy called "name" */
  double getEnergy(const char *name);
  /** @brief  Return the virial 3x3 matrix */
  void getVirial(double *virmat);
  /** @brief  Return the virial 3x3 matrix, from GPU to GPU  (using
   * CudaContainer) 
   */
  void getVirial(CudaContainer<double> &virial);

  // std::vector<double> calculateVirial();
  /** @brief  Return sforce 27*3 array */
  void getSforce(double *sforce);

  void addPotentialEnergies(const CudaEnergyVirial &other);
};

#endif // CUDAENERGYVIRIAL_H
#endif // NOCUDAC
