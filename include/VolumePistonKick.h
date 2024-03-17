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
 * \brief Classes and kernels to update box_dot based on the virial.
 */
#pragma once
#include <CudaIntegratorGraph.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/** Update the box time dirivative based on the virial.*/
__global__ void volumePistonKickKernel(
    // inputs
    const double3 *__restrict__ box, /**< device pointer to box dimensions.*/
    const double3 *old_box_dot, /**< device pointer to box time dirivative.*/
    const double3
        *__restrict__ virial, /**< device pointer to Corrected virial.*/
    // outputs
    double3 *new_box_dot, /**< device pointer to new box time dirivatives.*/
    // parameters
    double timestep,   /**< Time step. */
    double piston_mass /**< Piston mass, dimensions of mass/length^4. */
);

/** Create graph that updates the box time dirivative.*/
class VolumePistonKick : public CudaIntegratorGraph {
public:
  /** Create graph that updates the box time dirivative.*/
  VolumePistonKick(
      // inputs
      const double3 *__restrict__ box, /**< device pointer to box dimensions.*/
      const double3 *old_box_dot, /**< device pointer to box time dirivative.*/
      const double3
          *__restrict__ virial, /**< device pointer to Corrected virial.*/
      // outputs
      double3 *new_box_dot, /**< device pointer to new box time dirivatives.*/
      // parameters
      double timestep,   /**< Time step. */
      double piston_mass /**< Piston mass, dimensions of mass/length^4. */
  );

private:
  void initializeGraph(void);
  cudaKernelNodeParams myparams;
  std::array<void *, 6> kernelArgs;
  cudaGraphNode_t mynode;
  // inputs
  const double3 *__restrict__ box; /**< device pointer to box dimensions.*/
  const double3 *old_box_dot;      /**< device pointer to box time dirivative.*/
  const double3 *__restrict__ virial; /**< device pointer to Corrected virial.*/
  // outputs
  double3 *new_box_dot; /**< device pointer to new box time dirivatives.*/
  // parameters
  double timestep;    /**< Time step. */
  double piston_mass; /**< Piston mass, dimensions of mass/length^4. */
};
