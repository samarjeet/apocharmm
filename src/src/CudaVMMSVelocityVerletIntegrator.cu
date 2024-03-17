// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaVMMSVelocityVerletIntegrator.h"
#include "gpu_utils.h"
#include <chrono>
#include <iostream>

CudaVMMSVelocityVerletIntegrator::CudaVMMSVelocityVerletIntegrator(
    ts_t timeStep)
    : CudaIntegrator(timeStep) {
  // std::cout << "Setting up a velocity-verlet integrator.\n";
}

void CudaVMMSVelocityVerletIntegrator::initialize() {}

void CudaVMMSVelocityVerletIntegrator::setSimulationContexts(
    std::vector<CharmmContext> ctxs) {
  contexts = ctxs;
  std::cout << "Contexts set in VMMSIntegrator" << std::endl;
}
/*
__global__ void firstHalfKickAndDrift(const int numAtoms, const int stride,
                                      const ts_t timeStep, float4 *xyzq,
                                      double4 *velMass,
                                      const long long int *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    float fx = ((float)force[index]) * INV_FORCE_SCALE;
    float fy = ((float)force[index + stride]) * INV_FORCE_SCALE;
    float fz = ((float)force[index + 2 * stride]) * INV_FORCE_SCALE;

    velMass[index].x -= 0.5 * timeStep * fx * velMass[index].w;
    velMass[index].y -= 0.5 * timeStep * fy * velMass[index].w;
    velMass[index].z -= 0.5 * timeStep * fz * velMass[index].w;

    xyzq[index].x += timeStep * velMass[index].x;
    xyzq[index].y += timeStep * velMass[index].y;
    xyzq[index].z += timeStep * velMass[index].z;
  }
}

__global__ void secondHalfKick(const int numAtoms, const int stride,
                               const ts_t timeStep, float4 *xyzq,
                               double4 *velMass, const long long int *force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {

    float fx = force[index] * INV_FORCE_SCALE;
    float fy = force[index + stride] * INV_FORCE_SCALE;
    float fz = force[index + 2 * stride] * INV_FORCE_SCALE;

    velMass[index].x -= 0.5 * timeStep * fx * velMass[index].w;
    velMass[index].y -= 0.5 * timeStep * fy * velMass[index].w;
    velMass[index].z -= 0.5 * timeStep * fz * velMass[index].w;
  }
}

// change this to save delta

__global__ void firstHalfKickAndDrift(const int numAtoms, const int stride,
                                      const ts_t timeStep, float4 *xyzq,
                                      double4 *velMass,
                                      const float *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    float fx = force[index];
    float fy = force[index + stride];
    float fz = force[index + 2 * stride];

    velMass[index].x -= 0.5 * timeStep * fx * velMass[index].w;
    velMass[index].y -= 0.5 * timeStep * fy * velMass[index].w;
    velMass[index].z -= 0.5 * timeStep * fz * velMass[index].w;

    xyzq[index].x += timeStep * velMass[index].x;
    xyzq[index].y += timeStep * velMass[index].y;
    xyzq[index].z += timeStep * velMass[index].z;
  }
}

__global__ void secondHalfKick(const int numAtoms, const int stride,
                               const ts_t timeStep, float4 *xyzq,
                               double4 *velMass, const float *force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {

    float fx = force[index];
    float fy = force[index + stride];
    float fz = force[index + 2 * stride];

    velMass[index].x -= 0.5 * timeStep * fx * velMass[index].w;
    velMass[index].y -= 0.5 * timeStep * fy * velMass[index].w;
    velMass[index].z -= 0.5 * timeStep * fz * velMass[index].w;
  }
}
*/

void CudaVMMSVelocityVerletIntegrator::setSoluteAtoms(std::vector<int> atoms) {
  soluteAtoms.allocate(atoms.size());
  soluteAtoms.set(atoms);
  std::cout << "Solute atoms set\n";
}

__global__ void combineKernel() {}
void CudaVMMSVelocityVerletIntegrator::combineForces() {

  combinedForce->clear();
  int numAtoms = contexts[0].getNumAtoms();
  int numThreads = 512;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  for (int i = 0; i < contexts.size(); ++i) {
    weights[i] = 0.1;
    combineKernel<<<numBlocks, numThreads>>>();
  }
}
void CudaVMMSVelocityVerletIntegrator::propagateOneStep() {
  combineForces();
  /*
    auto force = contexts[0]->getForces();
    for (int i=1; i < conexts.size(); ++i){

    }
    auto xyzq = context->getXYZQ()->getDeviceXYZQ();

    auto velMass = context->getVelocityMass().getDeviceArray().data();
    // context->calculateForces(false, true, true);
    auto force = context->getForces();
    // std::cout << "vv - got forces\n";

    int numAtoms = context->getNumAtoms();
    int stride = context->getForceStride();
    // std::cout << "numAtoms = " << numAtoms << "\n";
    int numThreads = 128;
    int numBlocks = (numAtoms - 1) / numThreads + 1;

    // TEST

    // centerOfAllAtoms<<<numBlocks, numThreads>>>();
    // cudaDeviceSynchronize();
    // !TEST

    // Calculate v_(n+1/2)
    // Calculate r_(n+1) - before constriants
    // std::chrono::steady_clock::time_point start =
    // std::chrono::steady_clock::now();
    firstHalfKickAndDrift<<<numBlocks, numThreads>>>(numAtoms, stride, timeStep,
                                                     xyzq, velMass,
    force->xyz()); cudaDeviceSynchronize();
    // std::chrono::steady_clock::time_point end =
    // std::chrono::steady_clock::now(); std::chrono::steady_clock::duration
    // duration = end - start; std::cout << numAtoms << " 1st kick time : " <<
    // std::chrono::duration_cast<std::chrono::microseconds>(duration).count()
    <<
    // " ms\n";

    // TODO : handle constraints
    // context->handleHolonomicConstraints();

    // TODO : when do we update the neighborlist ??

    // calculate force
    // TODO : this should not be here
    if (stepsSinceNeighborListUpdate % 20 == 0) {
      // if (stepsSinceNeighborListUpdate % 1 == 0) {
      context->imageCentering();
      context->resetNeighborList();
    }
    // auto pe = context->calculatePotentialEnergy(true);
    context->calculateForces();
    force = context->getForces();

    // calculate v_(n+1)
    secondHalfKick<<<numBlocks, numThreads>>>(numAtoms, stride, timeStep, xyzq,
                                              velMass, force->xyz());
    cudaDeviceSynchronize();

   */
}
