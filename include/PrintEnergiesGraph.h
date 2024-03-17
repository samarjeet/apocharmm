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
 * \brief graphs and kernels to print the kinetic energy.
 */
#pragma once
#include <CudaIntegratorGraph.h>
#include <VolumePiston.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/** input for the graph that prints out the components of the total system
 * energy.*/
struct PrintEnergiesGraphInputs {
  // input
  /** Device pointer to Velocity and mass of each atom.*/
  const double4 *__restrict__ velmass;
  /** Device pointer to potential of all atoms.*/
  const double *__restrict__ potential_energy;
  /** Device pointer to box dimensions.*/
  const double3 *__restrict__ box;
  /** Device pointer to time derivative of box dimensions.*/
  const double3 *__restrict__ box_dot;
  // parameters
  /** Piston Parameters.*/
  VolumePiston piston;
  /** Number of Atoms.*/
  int numAtoms;
};

/** Print out Energies.
 *      Example:
 *          Potential Energy,0,Kinetic Energy,0,Box Piston Potential
 * Energy,0,Box Piston Kinetic Energy,0
 *
 *          Units are SI compatible.
 */
__global__ void PrintEnergiesGraphKernel(PrintEnergiesGraphInputs in);

/**Create a graph to print the system energies.
 *      Example:
 *          Potential Energy,0,Kinetic Energy,0,Box Piston Potential
 * Energy,0,Box Piston Kinetic Energy,0
 */
class PrintEnergiesGraph : public CudaIntegratorGraph {
public:
  PrintEnergiesGraph(PrintEnergiesGraphInputs in) : inputs(in) {
    myparams = {0};
    kernelArgs = {(void *)&inputs};
    myparams.func = (void *)PrintEnergiesGraphKernel;
    myparams.gridDim = dim3(1, 1, 1);
    myparams.blockDim = dim3(1, 1, 1);
    myparams.sharedMemBytes = 0;
    myparams.kernelParams = (void **)(kernelArgs.data());
    myparams.extra = NULL;
    // Add nodes
    cudaCheck(cudaGraphAddKernelNode(&mynode, graph, NULL, 0, &myparams));
  }

private:
  cudaKernelNodeParams myparams;
  std::array<void *, 1> kernelArgs;
  cudaGraphNode_t mynode;
  PrintEnergiesGraphInputs inputs;
};
