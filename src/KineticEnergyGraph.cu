// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Nathan, Zimmerberg, Samarjeet Prasad
//
// ENDLICENSE

#include <KineticEnergyGraph.h>
/** calculate the total kinetic energy*/
__global__ void kineticEnergySimpleKernel(
    // inputs
    const double4 *__restrict__ vel_mass, /**< The device pointer to the array
                                             of velocity and mass of every
                                             atom.*/
    // outputs
    double *__restrict__ total_kinetic_energy, /**< The total kinetic energy.*/
    // parameters
    int numAtoms /**< number of atoms.*/) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid == 0) {
    double ke = 0.0;
    for (int i = 0; i < numAtoms; i++) {
      double mass = vel_mass[i].w;
      double x = vel_mass[i].x;
      double y = vel_mass[i].y;
      double z = vel_mass[i].z;
      ke += 0.5 * mass * (x * x + y * y + z * z);
    }
    *total_kinetic_energy = ke;
  }
}

KineticEnergyGraph::KineticEnergyGraph(
    // inputs
    const double4 *__restrict__ vel_mass,
    double *__restrict__ total_kinetic_energy,
    // parameters
    int numAtoms)
    : vel_mass(vel_mass), total_kinetic_energy(total_kinetic_energy),
      numAtoms(numAtoms) {
  initializeGraph();
}

void KineticEnergyGraph::initializeGraph(void) {
  // create parameters
  myparams = {0};
  kernelArgs = {(void *)&vel_mass, (void *)&total_kinetic_energy,
                (void *)&numAtoms};
  myparams.func = (void *)kineticEnergySimpleKernel;
  myparams.gridDim = dim3(1, 1, 1);
  myparams.blockDim = dim3(1, 1, 1);
  myparams.sharedMemBytes = 0;
  myparams.kernelParams = (void **)(kernelArgs.data());
  myparams.extra = NULL;
  // Add nodes
  cudaCheck(cudaGraphAddKernelNode(&mynode, graph, NULL, 0, &myparams));
}

KineticEnergyGraph::~KineticEnergyGraph() {}
