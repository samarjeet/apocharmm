// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CudaVelocityVerletIntegrator.h"
#include "gpu_utils.h"
#include <chrono>
#include <iostream>
#include <map>

CudaVelocityVerletIntegrator::CudaVelocityVerletIntegrator(
    const double timeStep)
    : CudaIntegrator(timeStep) {
  m_StepsSinceLastReport = 0;
  m_IntegratorTypeName = "VelocityVerlet";
}

void CudaVelocityVerletIntegrator::initialize(void) { return; }

static __global__ void firstHalfKickAndDrift(const int numAtoms,
                                             const int stride,
                                             const double timeStep,
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
                                      const double timeStep,
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

void CudaVelocityVerletIntegrator::propagateOneStep(void) {
  auto xyzq = m_Context->getXYZQ()->getDeviceXYZQ();
  auto coordsCharge = m_Context->getCoordinatesCharges().getDeviceData();
  auto velMass = m_Context->getVelocityMass().getDeviceData();
  auto force = m_Context->getForces();

  int numAtoms = m_Context->getNumAtoms();
  int stride = m_Context->getForceStride();

  copy_DtoD_sync<double4>(coordsCharge, m_CoordsRef.getDeviceData(), numAtoms);

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  // Calculate v_(n+1/2)
  // Calculate r_(n+1) - before constriants

  firstHalfKickAndDrift<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      numAtoms, stride, m_TimeStep, // xyzq,
      coordsCharge, velMass, force->xyz());

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  if (m_UsingHolonomicConstraints) {
    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());
  }

  updateSPKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      numAtoms, xyzq, coordsCharge);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  // TODO : this should not be here
  if (m_StepsSinceNeighborListUpdate % 20 == 0) {
    m_Context->resetNeighborList();
  }

  m_Context->calculateForces();
  force = m_Context->getForces();

  // calculate v_(n+1)
  secondHalfKick<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      numAtoms, stride, m_TimeStep, xyzq, coordsCharge, velMass, force->xyz());
  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  auto mycccc = m_Context->getCoordinatesCharges();
  auto myccvm = m_Context->getVelocityMass();
  mycccc.transferFromDevice();
  myccvm.transferFromDevice();

  m_StepsSinceLastReport++;
  if (m_DebugPrintFrequency > 0 &&
      (m_StepsSinceLastReport + 1) % m_DebugPrintFrequency == 0) {
    auto peContainer = m_Context->getPotentialEnergy();
    peContainer.transferFromDevice();
    auto pe = peContainer[0];
    auto ke = m_Context->getKineticEnergy();

    //    std::cout << "[VVER]Potential energy : " << pe << "\n";
    //    std::cout << "[VVER]Kinetic energy : " << ke << "\n";
    //    std::cout << "[VVER]Total Energy : " << ke + pe << "\n";
    //    std::cout << "[VVER]Temp : " << m_Context->computeTemperature()
    //              << "\n-----------\n";

    std::cout << "[VVER] pos0x/vel0x: " << mycccc[0].x << " " << myccvm[0].x
              << "\n";
    m_StepsSinceLastReport = 0;
  }

  return;
}

std::map<std::string, std::string>
CudaVelocityVerletIntegrator::getIntegratorDescriptors(void) {
  std::map<std::string, std::string> ret;
  ret["type"] = "VelocityVerlet";
  ret["timestep"] = std::to_string(m_TimeStep);
  return ret;
}
