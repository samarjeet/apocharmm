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
 * \brief The class and kernels to update the net momentums of the pressure
 * groups.
 */
#pragma once
#include <CudaIntegratorGraph.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>
/** Type to store an array of atom ids*/
typedef struct AtomIdLists {
  int numAtoms; /**< Number of atom ids in the array. */
  int *idarray; /**< Pointer to first id in the array of atom ids. */
} AtomIdList_t;

/** Set the new momentums to the old momentums + the net force* timestep. */
__global__ void pressureGroupMomentumUpdateSimpleKernel(
    const double *__restrict__ forcex, const double *__restrict__ forcey,
    const double *__restrict__ forcez, const double4 *old_momentum_invmass,
    double4 *new_momentum_invmass,
    const AtomIdList_t *__restrict__ group_atom_ids, double timestep,
    int numGroups);

/** Create a graph that gets the net force on each pressure group and calc the
 * new momentum.*/
class PressureGroupMomentumUpdate : public CudaIntegratorGraph {
public:
  /** Create a graph that gets the net force on each pressure group and updates
   * the group net momentum. All units are SI compatible, defining constants
   * (planks constant, speed of light, charge of electron..., and such) may be
   * different.
   *  @param fx,fy,fz the device pointers to the array of atom forces.
   *      These must be in the same order as the atom ids.
   *  @param old_momentum_invmass The device pointer to the array of previous
   * net momentums, and the inverse of the net masses, numGroups long.
   *  @param new_momentum_invmass [out] The device pointer to the array of new
   * net momentums, and the inverse of the net masses, numGroups long.
   *  @param group_atom_ids The device pointer to the array of what atoms are in
   * each pressure group.
   *  @param timestep Time step, how long the force should be applied.
   *  @param numGroups Number of groups.
   */
  PressureGroupMomentumUpdate(const double *__restrict__ fx,
                              const double *__restrict__ fy,
                              const double *__restrict__ fz,
                              const double4 *old_momentum_invmass,
                              double4 *new_momentum_invmass,
                              const AtomIdList_t *__restrict__ group_atom_ids,
                              double timestep, int numGroups);

  /** Destroy the graph, its nodes, and free internal memory.*/
  ~PressureGroupMomentumUpdate();

private:
  void initializeGraph(void);
  cudaKernelNodeParams myparams;
  std::array<void *, 8> kernelArgs;
  cudaGraphNode_t mynode;
  const double *__restrict__ fx;
  const double *__restrict__ fy;
  const double *__restrict__ fz;
  const double4 *old_momentum_invmass;
  const double4 *new_momentum_invmass;
  const AtomIdList_t *__restrict__ group_atom_ids;
  double timestep;
  int numGroups;
};
