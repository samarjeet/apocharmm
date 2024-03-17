// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include <PressureGroupMomentumUpdate.h>
__global__ void pressureGroupMomentumUpdateSimpleKernel(
    const double *__restrict__ forcex, const double *__restrict__ forcey,
    const double *__restrict__ forcez, const double4 *old_momentum_invmass,
    double4 *new_momentum_invmass,
    const AtomIdList_t *__restrict__ group_atom_ids, double timestep,
    int numGroups) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid == 0) {
    for (int i = 0; i < numGroups; i++) {
      int *groupids = group_atom_ids[i].idarray;
      int numatoms = group_atom_ids[i].numAtoms;
      double group_fx = 0.0;
      double group_fy = 0.0;
      double group_fz = 0.0;
      for (int j = 0; j < numatoms; j++) {
        int atomid = groupids[j];
        group_fx += forcex[atomid];
        group_fy += forcey[atomid];
        group_fz += forcez[atomid];
      }
      double4 momentum = old_momentum_invmass[i];
      momentum.x = momentum.x + group_fx * timestep;
      momentum.y = momentum.y + group_fy * timestep;
      momentum.z = momentum.z + group_fz * timestep;
      new_momentum_invmass[i] = momentum;
    }
  }
}

PressureGroupMomentumUpdate::PressureGroupMomentumUpdate(
    const double *__restrict__ fx, const double *__restrict__ fy,
    const double *__restrict__ fz, const double4 *old_momentum_invmass,
    double4 *new_momentum_invmass,
    const AtomIdList_t *__restrict__ group_atom_ids, double timestep,
    int numGroups)
    : fx(fx), fy(fy), fz(fz), old_momentum_invmass(old_momentum_invmass),
      new_momentum_invmass(new_momentum_invmass),
      group_atom_ids(group_atom_ids), timestep(timestep), numGroups(numGroups) {
  initializeGraph();
}

void PressureGroupMomentumUpdate::initializeGraph(void) {
  // create parameters
  myparams = {0};
  kernelArgs = {(void *)&fx,
                (void *)&fy,
                (void *)&fz,
                (void *)&old_momentum_invmass,
                (void *)&new_momentum_invmass,
                (void *)&group_atom_ids,
                (void *)&timestep,
                (void *)&numGroups}; // an array of void pointers
  myparams.func = (void *)pressureGroupMomentumUpdateSimpleKernel;
  myparams.gridDim = dim3(1, 1, 1);
  myparams.blockDim = dim3(1, 1, 1);
  myparams.sharedMemBytes = 0;
  myparams.kernelParams = (void **)(kernelArgs.data());
  myparams.extra = NULL;
  // Add nodes
  cudaCheck(cudaGraphAddKernelNode(&mynode, graph, NULL, 0, &myparams));
}

PressureGroupMomentumUpdate::~PressureGroupMomentumUpdate() {}
