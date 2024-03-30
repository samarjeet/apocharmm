// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaVelocityVerletIntegrator.h"
#include "gpu_utils.h"
#include <chrono>
#include <iostream>
#include <map>

CudaVelocityVerletIntegrator::CudaVelocityVerletIntegrator(ts_t timeStep)
    : CudaIntegrator(timeStep) {
  stepsSinceLastReport = 0;
}

void CudaVelocityVerletIntegrator::setCharmmContext(
    std::shared_ptr<CharmmContext> ctx) {

  CudaIntegrator::setCharmmContext(ctx);
}

void CudaVelocityVerletIntegrator::initialize() {}

static __global__ void firstHalfKickAndDrift(const int numAtoms,
                                             const int stride,
                                             const ts_t timeStep,
                                             // float4 *__restrict__ xyzq,
                                             double4 *__restrict__ coordsCharge,
                                             double4 *__restrict__ velMass,
                                             const double *__restrict__ force) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    double fx = (double)force[index];
    double fy = (double)force[index + stride];
    double fz = (double)force[index + 2 * stride];

    velMass[index].x -= 0.5 * timeStep * fx * velMass[index].w;
    velMass[index].y -= 0.5 * timeStep * fy * velMass[index].w;
    velMass[index].z -= 0.5 * timeStep * fz * velMass[index].w;

    coordsCharge[index].x += timeStep * velMass[index].x;
    coordsCharge[index].y += timeStep * velMass[index].y;
    coordsCharge[index].z += timeStep * velMass[index].z;
  }
}

static __global__ void secondHalfKick(const int numAtoms, const int stride,
                                      const ts_t timeStep,
                                      float4 *__restrict__ xyzq,
                                      double4 *__restrict__ coordsCharge,
                                      double4 *__restrict__ velMass,
                                      const double *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {

    double fx = (double)force[index];
    double fy = (double)force[index + stride];
    double fz = (double)force[index + 2 * stride];

    velMass[index].x -= 0.5 * timeStep * fx * velMass[index].w;
    velMass[index].y -= 0.5 * timeStep * fy * velMass[index].w;
    velMass[index].z -= 0.5 * timeStep * fz * velMass[index].w;

    xyzq[index].x = (float)coordsCharge[index].x;
    xyzq[index].y = (float)coordsCharge[index].y;
    xyzq[index].z = (float)coordsCharge[index].z;
  }
}

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
void CudaVelocityVerletIntegrator::propagateOneStep() {

  auto xyzq = context->getXYZQ()->getDeviceXYZQ();

  auto coordsCharge = context->getCoordinatesCharges().getDeviceArray().data();
  auto velMass = context->getVelocityMass().getDeviceArray().data();

  auto force = context->getForces();

  int numAtoms = context->getNumAtoms();
  int stride = context->getForceStride();

  copy_DtoD_sync<double4>(coordsCharge, coordsRef.getDeviceArray().data(),
                          numAtoms);

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  // Calculate v_(n+1/2)
  // Calculate r_(n+1) - before constriants

  firstHalfKickAndDrift<<<numBlocks, numThreads, 0, *integratorStream>>>(
      numAtoms, stride, timeStep, // xyzq,
      coordsCharge, velMass, force->xyz());

  cudaCheck(cudaStreamSynchronize(*integratorStream));

  if (usingHolonomicConstraints) {
    holonomicConstraint->handleHolonomicConstraints(
        coordsRef.getDeviceArray().data());
  }

  updateSPKernel<<<numBlocks, numThreads, 0, *integratorStream>>>(
      numAtoms, xyzq, coordsCharge);

  cudaCheck(cudaStreamSynchronize(*integratorStream));

  // TODO : this should not be here
  if (stepsSinceNeighborListUpdate % 20 == 0) {
    context->resetNeighborList();
  }

  context->calculateForces();
  force = context->getForces();

  // calculate v_(n+1)
  secondHalfKick<<<numBlocks, numThreads, 0, *integratorStream>>>(
      numAtoms, stride, timeStep, xyzq, coordsCharge, velMass, force->xyz());
  cudaCheck(cudaStreamSynchronize(*integratorStream));

  auto mycccc = context->getCoordinatesCharges(),
       myccvm = context->getVelocityMass();
  mycccc.transferFromDevice();
  myccvm.transferFromDevice();

  stepsSinceLastReport++;
  if (debugPrintFrequency > 0 &&
      (stepsSinceLastReport + 1) % debugPrintFrequency == 0) {
    auto peContainer = context->getPotentialEnergy();
    peContainer.transferFromDevice();
    auto pe = peContainer[0];
    auto ke = context->getKineticEnergy();

    //    std::cout << "[VVER]Potential energy : " << pe << "\n";
    //    std::cout << "[VVER]Kinetic energy : " << ke << "\n";
    //    std::cout << "[VVER]Total Energy : " << ke + pe << "\n";
    //    std::cout << "[VVER]Temp : " << context->computeTemperature()
    //              << "\n-----------\n";

    std::cout << "[VVER] pos0x/vel0x: " << mycccc[0].x << " " << myccvm[0].x
              << "\n";
    stepsSinceLastReport = 0;
  }
}

std::map<std::string, std::string>
CudaVelocityVerletIntegrator::getIntegratorDescriptors() {
  std::map<std::string, std::string> ret;
  ret["type"] = "VelocityVerlet";
  ret["timestep"] = std::to_string(timeStep);
  return ret;
}
