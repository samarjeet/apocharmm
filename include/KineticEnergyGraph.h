// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Nathan Zimmerberg, Samarjeet Prasad
//
// ENDLICENSE

/** \file
 * \author Nathan Zimmerberg (nhz2@cornell.edu)
 * \date
 * \brief Class and kernels to calculate the total kinetic energy
 */
#pragma once
#include <CudaIntegratorGraph.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/** calculate the total kinetic energy*/
__global__ void kineticEnergySimpleKernel(
    // inputs
    const double4
        *__restrict__ vel_mass, /**< The device pointer to the array of velocity
                                   and mass of every atom.*/
    // outputs
    double *__restrict__ total_kinetic_energy, /**< The total kinetic energy.*/
    // parameters
    int numAtoms /**< number of atoms.*/);

/** Create a graph that calculates the kinetic energy.*/
class KineticEnergyGraph : public CudaIntegratorGraph {
public:
  /** Create a graph that calculate the kinetic energy.
   *      All units are SI compatible.
   */
  KineticEnergyGraph(
      // inputs
      const double4 *__restrict__ vel_mass,
      // outputs
      double *__restrict__ total_kinetic_energy,
      // parameters
      int numAtoms);

  ~KineticEnergyGraph();

private:
  void initializeGraph(void);
  cudaKernelNodeParams myparams;
  std::array<void *, 3> kernelArgs;
  cudaGraphNode_t mynode;
  // inputs
  const double4 *__restrict__ vel_mass;
  // outputs
  double *__restrict__ total_kinetic_energy;
  // parameters
  int numAtoms;
};
