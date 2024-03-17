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
 * \date 17 Jul 2019
 * \brief Functions and kernals to Calculate the correction to the virial to
 * account for pressure groups. \details
 */

#pragma once
#include <CudaIntegratorGraph.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/** virial gets the sum of the element wise product of each force and its
 * position.
 */
__global__ void
pressureGroupVirialSimpleKernel(const double *__restrict__ forcex,
                                const double *__restrict__ forcey,
                                const double *__restrict__ forcez,
                                const double *__restrict__ relative_coordsx,
                                const double *__restrict__ relative_coordsy,
                                const double *__restrict__ relative_coordsz,
                                int numAtoms, double3 *__restrict__ virial);

/** Creates a graph that calculates the virial correction for pressure groups*/
class PressureGroupVirialGraph : public CudaIntegratorGraph {
public:
  /** Create a graph that calculates and accumulates the pressure group internal
   virial.
   * @param fx,fy,fz the device pointers to the first element in an numAtoms
   length array. The x,y,and z components of the total force on each atom, in
   the same order as relative_coordsy. The forces should be directly from the
   same potential as the homogenous virial, no contraint force modifications.
      @param x,y,z the device pointers to the first element in an numAtoms
   length array. The x,y and z components of the relative coordinates of each
   atom. These coordinates are relative to the center of mass of their pressure
   group. The relative coordinates must never be tranformed by the boundary
   conditions during a simulation.
      @param numAtoms The number of atoms, and length of the forces and coords
   arrays.
      @param virial A device pointer to the virial to be written to by executing
   the graph. virial->x is set as the dot product of forcex and relative_coordsx
   when the graph executes, the same for y and z.
      TODO Add parameter to control which components of the full 3x3 virial are
   calculated, possible external virial struct and enum.
   */
  PressureGroupVirialGraph(const double *__restrict__ fx,
                           const double *__restrict__ fy,
                           const double *__restrict__ fz,
                           const double *__restrict__ x,
                           const double *__restrict__ y,
                           const double *__restrict__ z, int numAtoms,
                           double3 *__restrict__ virial);
  /** Destroy the graph, its nodes, and free internal memory.*/
  ~PressureGroupVirialGraph();

private:
  void initializeGraph(void);
  cudaKernelNodeParams myparams;
  std::array<void *, 8> kernelArgs;
  cudaGraphNode_t mynode;
  const double *__restrict__ fx;
  const double *__restrict__ fy;
  const double *__restrict__ fz;
  const double *__restrict__ x;
  const double *__restrict__ y;
  const double *__restrict__ z;
  double3 *__restrict__ virial;
  int numAtoms;
};
