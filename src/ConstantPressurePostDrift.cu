// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include <ConstantPressurePostDrift.h>

/** Recombine center of mass coordinates and relative coordinates to get
 * absolute coordinates.*/
__global__ void constantPressurePostDriftSimpleKernel(
    // inputs
    const double4 *__restrict__ relative_xyzq, /**< The device pointer to the
                                                  array of atom positions
                                                  relative to the
                                                  center of mass of their
                                                  pressure group.*/
    const double4 *__restrict__ relative_vel_mass, /**< The device pointer to
                                                      the array of relative
                                                      velocity and mass of
                                                      every atom.*/
    const ComID_t *__restrict__ com_ids, /**< The device pointer to the array of
                                            the position of the center of mass
                                            of each pressure group, and the
                                            atoms that are in the group.*/
    const double4 *__restrict__ com_momentum_invmass, /**< The device pointer to
                                                         the array of the group
                                                         net momentums and
                                                         inverse masses.*/
    // outputs
    double4 *__restrict__ absolute_xyzq, /**< The device pointer to the array of
                                            absolute atom positions.*/
    double4 *__restrict__ absolute_vel_mass, /**< The device pointer to the
                                                array of absolute velocity and
                                                mass of every atom.*/
    // parameters
    const int
        *sorted_atomids, /**< Array of all atom ids sorted by pressure group.*/
    int numGroups /**< number of groups.*/) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid == 0) {
    for (int i = 0; i < numGroups; i++) {
      const int *groupids = sorted_atomids + com_ids[i].ids.idarray;
      int numatoms = com_ids[i].ids.numAtoms;
      double comx = com_ids[i].x;
      double comy = com_ids[i].y;
      double comz = com_ids[i].z;
      double invmass = com_momentum_invmass[i].w;
      double nvx = com_momentum_invmass[i].x * invmass;
      double nvy = com_momentum_invmass[i].y * invmass;
      double nvz = com_momentum_invmass[i].z * invmass;
      for (int j = 0; j < numatoms; j++) {
        int atomid = groupids[j];
        absolute_vel_mass[atomid].x = relative_vel_mass[atomid].x + nvx;
        absolute_vel_mass[atomid].y = relative_vel_mass[atomid].y + nvy;
        absolute_vel_mass[atomid].z = relative_vel_mass[atomid].z + nvz;
        absolute_xyzq[atomid].x = relative_xyzq[atomid].x + comx;
        absolute_xyzq[atomid].y = relative_xyzq[atomid].y + comy;
        absolute_xyzq[atomid].z = relative_xyzq[atomid].z + comz;
      }
    }
  }
}

ConstantPressurePostDrift::ConstantPressurePostDrift(
    // inputs
    const double4 *__restrict__ relative_xyzq, /**< The device pointer to the
                                                  array of atom positions
                                                  relative to
                                                  the center of mass of their
                                                  pressure group.*/
    const double4 *__restrict__ relative_vel_mass, /**< The device pointer to
                                                      the array of relative
                                                      velocity and
                                                      mass of every atom.*/
    const ComID_t *__restrict__ com_ids, /**< The device pointer to the array of
                                            the position of the center of mass
                                            of each pressure group, and the
                                            atoms that are in the group.*/
    const double4 *__restrict__ com_momentum_invmass, /**< The device pointer to
                                                         the array of the group
                                                         net momentums and
                                                         inverse masses.*/
    // outputs
    double4 *__restrict__ absolute_xyzq, /**< The device pointer to the array of
                                            absolute atom positions.*/
    double4 *__restrict__ absolute_vel_mass, /**< The device pointer to the
                                                array of absolute velocity and
                                                mass of every atom.*/
    // parameters
    const int
        *sorted_atomids, /**< Array of all atom ids sorted by pressure group.*/
    int numGroups /**< number of groups.*/)
    : relative_xyzq(relative_xyzq), relative_vel_mass(relative_vel_mass),
      com_ids(com_ids), com_momentum_invmass(com_momentum_invmass),
      absolute_xyzq(absolute_xyzq), absolute_vel_mass(absolute_vel_mass),
      sorted_atomids(sorted_atomids), numGroups(numGroups) {
  initializeGraph();
}

void ConstantPressurePostDrift::initializeGraph(void) {
  // create parameters
  myparams = {0};
  kernelArgs = {
      (void *)&relative_xyzq,  (void *)&relative_vel_mass,
      (void *)&com_ids,        (void *)&com_momentum_invmass,
      (void *)&absolute_xyzq,  (void *)&absolute_vel_mass,
      (void *)&sorted_atomids, (void *)&numGroups}; // an array of void pointers
  myparams.func = (void *)constantPressurePostDriftSimpleKernel;
  myparams.gridDim = dim3(1, 1, 1);
  myparams.blockDim = dim3(1, 1, 1);
  myparams.sharedMemBytes = 0;
  myparams.kernelParams = (void **)(kernelArgs.data());
  myparams.extra = NULL;
  // Add nodes
  cudaCheck(cudaGraphAddKernelNode(&mynode, graph, NULL, 0, &myparams));
}
