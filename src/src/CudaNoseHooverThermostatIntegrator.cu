// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaNoseHooverThermostatIntegrator.h"
#include "gpu_utils.h"
#include <chrono>
#include <iostream>

CudaNoseHooverThermostatIntegrator::CudaNoseHooverThermostatIntegrator(
    ts_t timeStep)
    : CudaIntegrator(timeStep) {
  // std::cout << "Setting up a velocity-verlet integrator.\n";
  chainLength = 5;
}

void CudaNoseHooverThermostatIntegrator::initialize() {
  chainPositions.allocate(chainLength);
  chainVelocities.allocate(chainLength);
}

// change this to save delta

// static __global__ void firstHalfKickAndDrift(const int numAtoms,
//                                              const int stride,
//                                              const ts_t timeStep, float4
//                                              *xyzq, double4 *velMass, const
//                                              double *__restrict__ force) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < numAtoms) {
//     float fx = force[index];
//     float fy = force[index + stride];
//     float fz = force[index + 2 * stride];
//
//     velMass[index].x -= 0.5 * timeStep * fx * velMass[index].w;
//     velMass[index].y -= 0.5 * timeStep * fy * velMass[index].w;
//     velMass[index].z -= 0.5 * timeStep * fz * velMass[index].w;
//
//     xyzq[index].x += timeStep * velMass[index].x;
//     xyzq[index].y += timeStep * velMass[index].y;
//     xyzq[index].z += timeStep * velMass[index].z;
//   }
// }
//
// static __global__ void secondHalfKick(const int numAtoms, const int stride,
//                                       const ts_t timeStep, float4 *xyzq,
//                                       double4 *velMass, const double *force)
//                                       {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < numAtoms) {
//
//     float fx = force[index];
//     float fy = force[index + stride];
//     float fz = force[index + 2 * stride];
//
//     velMass[index].x -= 0.5 * timeStep * fx * velMass[index].w;
//     velMass[index].y -= 0.5 * timeStep * fy * velMass[index].w;
//     velMass[index].z -= 0.5 * timeStep * fz * velMass[index].w;
//   }
// }
//
//__global__ void ke(const double4 *velMass, int numAtoms, double *d_ke) {
//   extern __shared__ double sdata[];
//
//   unsigned int id = threadIdx.x;
//   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
//
//   if (tid < numAtoms) {
//     sdata[id] =
//         (velMass[tid].x * velMass[tid].x + velMass[tid].y * velMass[tid].y +
//          velMass[tid].z * velMass[tid].z) /
//         velMass[tid].w;
//   } else {
//     sdata[id] = 0.0f;
//   }
//   __syncthreads();
//
//   for (int s = 1; s < blockDim.x; s *= 2) {
//     if (id % (s * 2) == 0) {
//       sdata[id] += sdata[id + s];
//     }
//     __syncthreads();
//
//     if (id == 0) {
//       d_ke[blockIdx.x] = (double)sdata[0];
//     }
//   }
// }
//
/*
 Velocity-Verlet drift step propagator
*/
static __global__ void u1Propagator(float deltaT, int numAtoms,
                                    const double4 *__restrict__ velMass,
                                    double4 *__restrict__ xyzq) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    xyzq[index].x += deltaT * velMass[index].x;
    xyzq[index].y += deltaT * velMass[index].y;
    xyzq[index].z += deltaT * velMass[index].z;
  }
}

/*
Velocty-Verlet kick step propagator
*/
static __global__ void u2Propagator(float deltaT, int numAtoms, int stride,
                                    double4 *__restrict__ velMass,
                                    const double *__restrict__ force,
                                    const double4 *__restrict__ xyzq) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {

    const double fx = force[index];
    const double fy = force[index + stride];
    const double fz = force[index + 2 * stride];

    velMass[index].x -= deltaT * fx * velMass[index].w;
    velMass[index].y -= deltaT * fy * velMass[index].w;
    velMass[index].z -= deltaT * fz * velMass[index].w;
  }
}

///*
// Thermostat propagator
// chain positions
//*/
//  __global__ void
// u3Propagator(float deltaT, int numAtoms, double4 *__restrict__ velMass,
//              int chainLength, double *__restrict__ chainPositions,
//              const double *__restrict__ chainVelocities) {
//   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
//
//   if (index < numAtoms) {
//     velMass[index].x *= exp(-deltaT * chainVelocities[0]);
//     velMass[index].y *= exp(-deltaT * chainVelocities[0]);
//     velMass[index].z *= exp(-deltaT * chainVelocities[0]);
//   }
//
//   if (index == 0) {
//     for (int i = 0; i < chainLength; i++) {
//       chainPositions[i] += deltaT * chainVelocities[i];
//     }
//   }
// }
//
///*
// Thermostat propagator
// chain velocities are updated
//*/
//__global__ void u434Propagator(float deltaT, int numAtoms, double kT,
//                                       int ndegf, double4 *__restrict__
//                                       velMass, int chainLength, double
//                                       *__restrict__ chainPositions, double
//                                       *__restrict__ chainVelocities, const
//                                       double *__restrict__ chainMasses) {
//
//   // Calculate p * p / m
//   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < numAtoms) {
//   }
//
//   double kineticEnergy = 0.0; // TODO : calculate it
//   int M = chainLength - 1;
//   double G =
//       ((chainVelocities[M - 1] * chainVelocities[M - 1]) * chainMasses[M - 1]
//       -
//        kT) /
//       chainMasses[M];
//
//   chainVelocities[M] += deltaT * G;
//
//   for (int m = M - 1; m >= 0; m--) {
//     G = ((chainVelocities[m - 1] * chainVelocities[m - 1]) *
//              chainMasses[m - 1] -
//          kT) /
//         chainMasses[m];
//     double scale = exp(-deltaT * chainVelocities[m]);
//     chainVelocities[m] = scale * (scale * chainVelocities[m] + deltaT * G);
//   }
//
//   // put u3 here
//
//   G = (2 * kineticEnergy - ndegf * kT) / chainMasses[0];
//   for (int m = 0; m < M - 1; m++) {
//     double scale = exp(-deltaT * chainVelocities[m + 1]);
//     chainVelocities[m] = scale * (scale * chainVelocities[m] + deltaT * G);
//     G = (chainVelocities[m] * chainVelocities[m] * chainMasses[m] - kT) /
//         chainMasses[m + 1];
//   }
//   chainVelocities[M] += deltaT * G;
// }

static __global__ void
updateSPKernel(int numAtoms, float4 *__restrict__ xyzq,
               const double4 *__restrict__ coordsCharge) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    xyzq[index].x = (float)coordsCharge[index].x;
    xyzq[index].y = (float)coordsCharge[index].y;
    xyzq[index].z = (float)coordsCharge[index].z;
  }
}
void CudaNoseHooverThermostatIntegrator::propagateOneStep() {

  std::cout << "The Nose-Hoover integrator is not implemented yet." << std::endl;
  int numAtoms = context->getNumAtoms();
  int stride = context->getForceStride();

  // double kbt = kBoltz * bathTemperature;

  auto coords = context->getCoordinatesCharges().getDeviceArray().data();
  auto velMass = context->getVelocityMass().getDeviceArray().data();
  auto force = context->getForces();

  /*
  u4 u3 u4
  */

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  u2Propagator<<<numBlocks, numThreads>>>(timeStep / 2.0, numAtoms, stride,
                                          velMass, force->xyz(), coords);

  cudaCheck(cudaDeviceSynchronize()); // TODO : remove these

  u1Propagator<<<numBlocks, numThreads>>>(timeStep, numAtoms, velMass, coords);
  cudaCheck(cudaDeviceSynchronize()); // TODO : remove these

  auto xyzq = context->getXYZQ()->getDeviceXYZQ();
  updateSPKernel<<<numBlocks, numThreads, 0, *integratorStream>>>(numAtoms,
                                                                  xyzq, coords);
  cudaCheck(cudaDeviceSynchronize()); // TODO : remove these
  if (stepsSinceNeighborListUpdate % 20 == 0) {
    context->resetNeighborList();
  }

  context->calculateForces();
  force = context->getForces();

  u2Propagator<<<numBlocks, numThreads>>>(timeStep / 2.0, numAtoms, stride,
                                          velMass, force->xyz(), coords);
  cudaCheck(cudaDeviceSynchronize()); // TODO : remove these

}
