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
 * \brief Graph with a simple leapfrog integrator, no constriants or pistons
 */
#pragma once
#include <CudaIntegratorGraph.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/**Struct for Simple leapfrog kernel input*/
struct SimpleLeapfrogGraphInputs {
  // input
  const double4 *old_xyzq;    /**< Device pointer to old positions.*/
  const double4 *old_velmass; /**< Device pointer to old velocity.*/
  const double4
      *__restrict__ force_invmass; /**<Device pointer to force inverse mass.*/
  // output
  double4 *new_xyzq;    /**<Device pointer to new positions.*/
  double4 *new_velmass; /**<Device pointer to new velocities.*/
  // parameters
  int numAtoms;    /**<Number of atoms.*/
  double timestep; /**<Time step.*/
};

/**Update the on step positions and previous half step velocities using the leap
 * frog algorithm.*/
__global__ void SimpleLeapfrogGraphKernel(SimpleLeapfrogGraphInputs in);

/**Create a graph to do the simple leapfrog integration.*/
class SimpleLeapfrogGraph : public CudaIntegratorGraph {
public:
  SimpleLeapfrogGraph(SimpleLeapfrogGraphInputs in) : inputs(in) {
    int nThreads = 128;
    int nBlocks = (inputs.numAtoms - 1) / nThreads + 1;
    myparams = {0};
    kernelArgs = {(void *)&inputs};
    myparams.func = (void *)SimpleLeapfrogGraphKernel;
    myparams.gridDim = dim3(nThreads, 1, 1);
    myparams.blockDim = dim3(nBlocks, 1, 1);
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
  SimpleLeapfrogGraphInputs inputs;
};

__global__ void SimpleLeapfrogGraphKernel(SimpleLeapfrogGraphInputs in);
