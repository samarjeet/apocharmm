// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "CudaLeapFrogIntegrator.h"
#include <iostream>

CudaLeapFrogIntegrator::CudaLeapFrogIntegrator(double timeStep)
    : CudaIntegrator(timeStep) {
  stepsSinceLastReport = 0;
}

void CudaLeapFrogIntegrator::initialize() {}

void CudaLeapFrogIntegrator::setCharmmContext(
    std::shared_ptr<CharmmContext> ctx) {
  CudaIntegrator::setCharmmContext(ctx);
}

__global__ static void leapFrogKernel(const int numAtoms, const int stride,
                                      const ts_t timeStep,
                                      float4 *__restrict__ xyzq,
                                      double4 *__restrict__ velMass,
                                      const double *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    float fx = force[index];
    float fy = force[index + stride];
    float fz = force[index + 2 * stride];

    velMass[index].x -= timeStep * fx * velMass[index].w;
    velMass[index].y -= timeStep * fy * velMass[index].w;
    velMass[index].z -= timeStep * fz * velMass[index].w;

    xyzq[index].x += timeStep * velMass[index].x;
    xyzq[index].y += timeStep * velMass[index].y;
    xyzq[index].z += timeStep * velMass[index].z;
  }
}

void CudaLeapFrogIntegrator::propagateOneStep() {

  auto xyzq = context->getXYZQ()->getDeviceXYZQ();
  auto velMass = context->getVelocityMass().getDeviceArray().data();

  if (stepsSinceNeighborListUpdate % 20 == 0) {
    context->resetNeighborList();
  }

  context->calculateForces();
  auto force = context->getForces();

  context->calculateKineticEnergy();
  auto ke = context->getKineticEnergy();
  auto peContainer = context->getPotentialEnergy();
  peContainer.transferFromDevice();
  auto pe = peContainer[0];

  int numAtoms = context->getNumAtoms();
  int stride = context->getForceStride();

  int numThreads = 1024;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  gpu_range_start("leapFrog");
  leapFrogKernel<<<numBlocks, numThreads>>>(numAtoms, stride, timeStep, xyzq,
                                            velMass, force->xyz());
  cudaCheck(cudaDeviceSynchronize());

  gpu_range_stop();

  stepsSinceLastReport++;
  if (debugPrintFrequency > 0 &&
      stepsSinceLastReport % debugPrintFrequency == 0) {
    stepsSinceLastReport = 0;
  }
}

std::map<std::string, std::string>
CudaLeapFrogIntegrator::getIntegratorDescriptors() {
  std::map<std::string, std::string> descriptors;
  descriptors["integratorType"] = "LeapFrog";
  descriptors["timeStep"] = std::to_string(timeStep);
  return descriptors;
}
