// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include <ConstantPressurePrepareDrift.h>
/** calculate the net momentums, center of masses(COM), relative coordinates to
 * the COMs, internal momentums, and kinetic energies from the COM motion, and
 * kinetic energies from the internal motions.*/
__global__ void constantPressurePrepareDriftSimpleKernel(
    // inputs
    const double4 *__restrict__ absolute_vel_mass, /**< The device pointer to
                                                      the array of absolute
                                                      velocity and mass of
                                                      every atom.*/
    const double4 *__restrict__ absolute_xyzq, /**< The device pointer to the
                                                  array of absolute atom
                                                  positions.*/
    // outputs
    double4 *__restrict__ relative_xyzq, /**< The device pointer to the array of
                                            atom positions relative to the
                                            center of mass of their pressure
                                            group.*/
    double4 *__restrict__ relative_vel_mass, /**< The device pointer to the
                                                array of relative velocity and
                                                mass of every atom.*/
    ComID_t *__restrict__ com_ids, /**< The device pointer to the array of the
                                      position of the center of mass of each
                                      pressure group, and the atoms that are in
                                      the group.*/
    double4 *__restrict__ com_momentum_invmass, /**< The device pointer to the
                                                   array of the group net
                                                   momentums and inverse
                                                   masses.*/
    double3 *__restrict__ com_kinetic_energy,  /**< The xx,yy, and zz components
                                                  of the groups center of mass
                                                  kinetic  energy.*/
    double *__restrict__ total_kinetic_energy, /**< The total kinetic energy.*/
    // parameters
    const int
        *sorted_atomids, /**< Array of all atom ids sorted by pressure group.*/
    int numGroups /**< number of groups.*/) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid == 0) {
    double totke = 0.0;
    double comkexx = 0.0;
    double comkeyy = 0.0;
    double comkezz = 0.0;
    for (int i = 0; i < numGroups; i++) {
      const int *groupids = sorted_atomids + com_ids[i].ids.idarray;
      int numatoms = com_ids[i].ids.numAtoms;
      double comx = 0.0;
      double comy = 0.0;
      double comz = 0.0;
      double npx = 0.0;
      double npy = 0.0;
      double npz = 0.0;
      for (int j = 0; j < numatoms; j++) {
        int atomid = groupids[j];
        double mass = absolute_vel_mass[atomid].w;
        double vx = absolute_vel_mass[atomid].x;
        double vy = absolute_vel_mass[atomid].y;
        double vz = absolute_vel_mass[atomid].z;
        double x = absolute_xyzq[atomid].x;
        double y = absolute_xyzq[atomid].y;
        double z = absolute_xyzq[atomid].z;
        comx += x * mass;
        comy += y * mass;
        comz += z * mass;
        npx += vx * mass;
        npy += vy * mass;
        npz += vz * mass;
        totke += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
      }
      double invmass = com_momentum_invmass[i].w;
      comx = invmass * comx;
      comy = invmass * comy;
      comz = invmass * comz;
      double nvx = invmass * npx;
      double nvy = invmass * npy;
      double nvz = invmass * npz;
      com_ids[i].x = comx;
      com_ids[i].y = comy;
      com_ids[i].z = comz;
      com_momentum_invmass[i].x = npx;
      com_momentum_invmass[i].y = npy;
      com_momentum_invmass[i].z = npz;
      comkexx += 0.5 * invmass * (npx * npx);
      comkeyy += 0.5 * invmass * (npy * npy);
      comkezz += 0.5 * invmass * (npz * npz);
      for (int j = 0; j < numatoms; j++) {
        int atomid = groupids[j];
        relative_vel_mass[atomid].x = absolute_vel_mass[atomid].x - nvx;
        relative_vel_mass[atomid].y = absolute_vel_mass[atomid].y - nvy;
        relative_vel_mass[atomid].z = absolute_vel_mass[atomid].z - nvz;
        relative_xyzq[atomid].x = absolute_xyzq[atomid].x - comx;
        relative_xyzq[atomid].y = absolute_xyzq[atomid].y - comy;
        relative_xyzq[atomid].z = absolute_xyzq[atomid].z - comz;
      }
    }
    *total_kinetic_energy = totke;
    com_kinetic_energy->x = comkexx;
    com_kinetic_energy->y = comkeyy;
    com_kinetic_energy->z = comkezz;
  }
}

ConstantPressurePrepareDrift::ConstantPressurePrepareDrift(
    // inputs
    const double4 *__restrict__ absolute_vel_mass, /**< The device pointer to
                                                      the array of absolute
                                                      velocity and
                                                      mass of every atom.*/
    const double4 *__restrict__ absolute_xyzq, /**< The device pointer to the
                                                  array of absolute atom
                                                  positions.*/
    // outputs
    double4 *__restrict__ relative_xyzq, /**< The device pointer to the array of
                                            atom positions relative to the
                                            center of mass of their pressure
                                            group.*/
    double4 *__restrict__ relative_vel_mass, /**< The device pointer to the
                                                array of relative velocity and
                                                mass of every atom.*/
    ComID_t *__restrict__ com_ids, /**< The device pointer to the array of the
                                      position of the center of mass of each
                                      pressure group, and the atoms that are in
                                      the group.*/
    double4 *__restrict__ com_momentum_invmass, /**< The device pointer to the
                                                   array of the group net
                                                   momentums and inverse
                                                   masses.*/
    double3 *__restrict__ com_kinetic_energy,  /**< The xx,yy, and zz components
                                                  of the groups center of mass
                                                  kinetic  energy.*/
    double *__restrict__ total_kinetic_energy, /**< The total kinetic energy,
                                                  used for the thermostat
                                                  piston.*/
    // parameters
    const int
        *sorted_atomids, /**< Array of all atom ids sorted by pressure group.*/
    int numAtoms,        /**< number of atoms.*/
    int numGroups /**< number of groups.*/)
    : absolute_vel_mass(absolute_vel_mass), absolute_xyzq(absolute_xyzq),
      relative_xyzq(relative_xyzq), relative_vel_mass(relative_vel_mass),
      com_ids(com_ids), com_momentum_invmass(com_momentum_invmass),
      com_kinetic_energy(com_kinetic_energy),
      total_kinetic_energy(total_kinetic_energy), numAtoms(numAtoms),
      numGroups(numGroups), sorted_atomids(sorted_atomids) {
  initializeGraph();
}

void ConstantPressurePrepareDrift::initializeGraph(void) {
  // create parameters
  myparams = {0};
  kernelArgs = {(void *)&absolute_vel_mass,
                (void *)&absolute_xyzq,
                (void *)&relative_xyzq,
                (void *)&relative_vel_mass,
                (void *)&com_ids,
                (void *)&com_momentum_invmass,
                (void *)&com_kinetic_energy,
                (void *)&total_kinetic_energy,
                (void *)&sorted_atomids,
                (void *)&numGroups}; // an array of void pointers
  myparams.func = (void *)constantPressurePrepareDriftSimpleKernel;
  myparams.gridDim = dim3(1, 1, 1);
  myparams.blockDim = dim3(1, 1, 1);
  myparams.sharedMemBytes = 0;
  myparams.kernelParams = (void **)(kernelArgs.data());
  myparams.extra = NULL;
  // Add nodes
  cudaCheck(cudaGraphAddKernelNode(&mynode, graph, NULL, 0, &myparams));
}

ConstantPressurePrepareDrift::~ConstantPressurePrepareDrift() {}
