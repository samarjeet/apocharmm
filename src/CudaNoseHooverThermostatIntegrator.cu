// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "Constants.h"
#include "CudaNoseHooverThermostatIntegrator.h"
#include "gpu_utils.h"
#include <chrono>
#include <iostream>

CudaNoseHooverThermostatIntegrator::CudaNoseHooverThermostatIntegrator(
    const double timeStep)
    : CudaIntegrator(timeStep) {
  m_UsingHolonomicConstraints = true;

  m_ReferenceTemperature = 300.0;

  // Allocate memory for Nose-Hoover variables
  m_NoseHooverPistonMass.resize(1);
  m_NoseHooverPistonVelocity.resize(1);
  m_NoseHooverPistonVelocityPrevious.resize(1);
  m_NoseHooverPistonForce.resize(1);
  m_NoseHooverPistonForcePrevious.resize(1);

  // Set Nose-Hoover variables to default values
  m_NoseHooverPistonMass.setToValue(-9999.9999);
  m_NoseHooverPistonVelocity.setToValue(0.0);
  m_NoseHooverPistonVelocityPrevious.setToValue(0.0);
  m_NoseHooverPistonForce.setToValue(0.0);
  m_NoseHooverPistonForcePrevious.setToValue(0.0);

  m_MaxPredictorCorrectorIterations = 3;

  m_KineticEnergy.resize(1);
  m_KineticEnergy.setToValue(0.0);

  m_AverageTemperature.resize(1);
  m_AverageTemperature.setToValue(0.0);

  m_UseOldTemperature = false;
  m_AverageOldTemperature.resize(1);
  m_AverageOldTemperature.setToValue(0.0);

  m_AverageWindowSize = 0;
}

void CudaNoseHooverThermostatIntegrator::setReferenceTemperature(
    const double referenceTemperature) {
  m_ReferenceTemperature = referenceTemperature;
  return;
}

void CudaNoseHooverThermostatIntegrator::setNoseHooverPistonMass(
    const double noseHooverPistonMass) {
  m_NoseHooverPistonMass.setToValue(noseHooverPistonMass);
  return;
}

void CudaNoseHooverThermostatIntegrator::setNoseHooverPistonVelocity(
    const double noseHooverPistonVelocity) {
  m_NoseHooverPistonVelocity.setToValue(noseHooverPistonVelocity);
  return;
}

void CudaNoseHooverThermostatIntegrator::setNoseHooverPistonVelocityPrevious(
    const double noseHooverPistonVelocityPrevious) {
  m_NoseHooverPistonVelocityPrevious.setToValue(
      noseHooverPistonVelocityPrevious);
  return;
}

void CudaNoseHooverThermostatIntegrator::setNoseHooverPistonForce(
    const double noseHooverPistonForce) {
  m_NoseHooverPistonForce.setToValue(noseHooverPistonForce);
  return;
}

void CudaNoseHooverThermostatIntegrator::setNoseHooverPistonForcePrevious(
    const double noseHooverPistonForcePrevious) {
  m_NoseHooverPistonForcePrevious.setToValue(noseHooverPistonForcePrevious);
  return;
}

void CudaNoseHooverThermostatIntegrator::setMaxPredictorCorrectorIterations(
    const int maxPredictorCorrectorIterations) {
  m_MaxPredictorCorrectorIterations = maxPredictorCorrectorIterations;
  return;
}

void CudaNoseHooverThermostatIntegrator::useOldTemperature(
    const bool useOldTemperature) {
  m_UseOldTemperature = useOldTemperature;
  return;
}

void CudaNoseHooverThermostatIntegrator::resetAverageTemperature(void) {
  m_AverageTemperature.setToValue(0.0);
  m_AverageOldTemperature.setToValue(0.0);
  m_AverageWindowSize = 0;
  return;
}

double CudaNoseHooverThermostatIntegrator::getReferenceTemperature(void) const {
  return m_ReferenceTemperature;
}

const CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getNoseHooverPistonMass(void) const {
  return m_NoseHooverPistonMass;
}

const CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getNoseHooverPistonVelocity(void) const {
  return m_NoseHooverPistonVelocity;
}

const CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getNoseHooverPistonVelocityPrevious(
    void) const {
  return m_NoseHooverPistonVelocityPrevious;
}

const CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getNoseHooverPistonForce(void) const {
  return m_NoseHooverPistonForce;
}

const CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getNoseHooverPistonForcePrevious(
    void) const {
  return m_NoseHooverPistonForcePrevious;
}

int CudaNoseHooverThermostatIntegrator::getMaxPredictorCorrectorIterations(
    void) const {
  return m_MaxPredictorCorrectorIterations;
}

const CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getKineticEnergy(void) const {
  return m_KineticEnergy;
}

const CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getAverageTemperature(void) const {
  return m_AverageTemperature;
}

const CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getAverageOldTemperature(void) const {
  return m_AverageOldTemperature;
}

CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getNoseHooverPistonMass(void) {
  return m_NoseHooverPistonMass;
}

CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getNoseHooverPistonVelocity(void) {
  return m_NoseHooverPistonVelocity;
}

CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getNoseHooverPistonVelocityPrevious(void) {
  return m_NoseHooverPistonVelocityPrevious;
}

CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getNoseHooverPistonForce(void) {
  return m_NoseHooverPistonForce;
}

CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getNoseHooverPistonForcePrevious(void) {
  return m_NoseHooverPistonForcePrevious;
}

CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getKineticEnergy(void) {
  return m_KineticEnergy;
}

CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getAverageTemperature(void) {
  return m_AverageTemperature;
}

CudaContainer<double> &
CudaNoseHooverThermostatIntegrator::getAverageOldTemperature(void) {
  return m_AverageOldTemperature;
}

double CudaNoseHooverThermostatIntegrator::getInstantaneousTemperature(void) {
  const double ndegf = static_cast<double>(m_Context->getDegreesOfFreedom());
  m_KineticEnergy.transferToHost();
  return (m_KineticEnergy[0] / (0.5 * ndegf * charmm::constants::kBoltz));
}

/** @brief Updates the single-prercision coordinates container (xyzq)
 */
__global__ static void UpdateSinglePrecisionCoordinatesKernel(
    float4 *__restrict__ xyzq, const double4 *__restrict__ coordsCharges,
    const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    xyzq[i].x = static_cast<float>(coordsCharges[i].x);
    xyzq[i].y = static_cast<float>(coordsCharges[i].y);
    xyzq[i].z = static_cast<float>(coordsCharges[i].z);
  }

  return;
}

__global__ static void
InitializationKernel(double4 *__restrict__ coordsDelta,
                     double4 *__restrict__ coordsDeltaPrevious,
                     const double4 *__restrict__ velMass, const int numAtoms,
                     const double *__restrict__ forces, const int forceStride,
                     const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    const double fx = forces[0 * forceStride + i];
    const double fy = forces[1 * forceStride + i];
    const double fz = forces[2 * forceStride + i];
    const double fact = 0.5 * timeStep2 * velMass[i].w;

    coordsDelta[i].x = timeStep * velMass[i].x - fact * fx;
    coordsDelta[i].y = timeStep * velMass[i].y - fact * fy;
    coordsDelta[i].z = timeStep * velMass[i].z - fact * fz;

    coordsDeltaPrevious[i].x = timeStep * velMass[i].x + fact * fx;
    coordsDeltaPrevious[i].y = timeStep * velMass[i].y + fact * fy;
    coordsDeltaPrevious[i].z = timeStep * velMass[i].z + fact * fz;
  }

  return;
}

__global__ static void
BackStepInitializationKernel1(double4 *__restrict__ coordsCharges,
                              const double4 *__restrict__ coordsDeltaPrevious,
                              const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    coordsCharges[i].x -= coordsDeltaPrevious[i].x;
    coordsCharges[i].y -= coordsDeltaPrevious[i].y;
    coordsCharges[i].z -= coordsDeltaPrevious[i].z;
  }

  return;
}

__global__ static void
BackStepInitializationKernel2(double4 *__restrict__ coordsCharges,
                              double4 *__restrict__ coordsDeltaPrevious,
                              const double4 *__restrict__ coordsRef,
                              const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    coordsDeltaPrevious[i].x = coordsRef[i].x - coordsCharges[i].x;
    coordsDeltaPrevious[i].y = coordsRef[i].y - coordsCharges[i].y;
    coordsDeltaPrevious[i].z = coordsRef[i].z - coordsCharges[i].z;

    coordsCharges[i].x = coordsRef[i].x;
    coordsCharges[i].y = coordsRef[i].y;
    coordsCharges[i].z = coordsRef[i].z;
  }

  return;
}

void CudaNoseHooverThermostatIntegrator::initialize(void) {
  const int numAtoms = m_Context->getNumAtoms();

  m_CoordsDeltaPredicted.resize(numAtoms);

  if (m_NoseHooverPistonMass[0] == -9999.9999)
    m_NoseHooverPistonMass.setToValue(this->computeNoseHooverPistonMass());

  double4 *coordsCharges = m_Context->getCoordinatesCharges().getDeviceData();
  float4 *xyzq = m_Context->getXYZQ()->getDeviceXYZQ();

  if (m_UsingHolonomicConstraints) {
    copy_DtoD_async<double4>(coordsCharges, m_CoordsRef.getDeviceData(),
                             numAtoms, *m_IntegratorStream);

    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());

    const int numThreads = 256;
    const int numBlocks = numAtoms / numThreads + 1;
    UpdateSinglePrecisionCoordinatesKernel<<<numBlocks, numThreads, 0,
                                             *m_IntegratorStream>>>(
        xyzq, coordsCharges, numAtoms);

    copy_DtoD_async<double4>(coordsCharges, m_CoordsRef.getDeviceData(),
                             numAtoms, *m_IntegratorStream);

    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  }

  m_Context->calculateForces();

  double4 *velMass = m_Context->getVelocityMass().getDeviceData();
  double *forces = m_Context->getForces()->xyz();
  int forceStride = m_Context->getForceStride();

  {
    const int numThreads = 256;
    const int numBlocks = numAtoms / numThreads + 1;
    InitializationKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
        m_CoordsDelta.getDeviceData(), m_CoordsDeltaPrevious.getDeviceData(),
        velMass, numAtoms, forces, forceStride, m_TimeStep);
  }

  if (m_UsingHolonomicConstraints) {
    const int numThreads = 256;
    const int numBlocks = numAtoms / numThreads + 1;
    BackStepInitializationKernel1<<<numBlocks, numThreads, 0,
                                    *m_IntegratorStream>>>(
        coordsCharges, m_CoordsDeltaPrevious.getDeviceData(), numAtoms);

    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());

    BackStepInitializationKernel2<<<numBlocks, numThreads, 0,
                                    *m_IntegratorStream>>>(
        coordsCharges, m_CoordsDeltaPrevious.getDeviceData(),
        m_CoordsRef.getDeviceData(), numAtoms);
  }

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  return;
}

__global__ static void InvertDeltaAsymmetricKernel(
    double4 *__restrict__ coordsDeltaPrevious, const float4 *__restrict__ xyzq,
    const int2 *__restrict__ groups, const int numGroups, const float boxDimX) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numGroups; i += stride) {
    const int2 group = groups[i];

    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    for (int j = group.x; j <= group.y; j++) {
      gx += xyzq[j].x;
      gy += xyzq[j].y;
      gz += xyzq[j].z;
    }
    gx /= static_cast<float>(group.y - group.x + 1);
    gy /= static_cast<float>(group.y - group.x + 1);
    gz /= static_cast<float>(group.y - group.x + 1);

    if ((gx > 0.5 * boxDimX) || (gx < -0.5 * boxDimX)) {
      for (int j = group.x; j <= group.y; j++) {
        coordsDeltaPrevious[j].y *= -1.0;
        coordsDeltaPrevious[j].z *= -1.0;
      }
    }
  }

  return;
}

__global__ static void
HalfStepVelocityKernel(double4 *__restrict__ coordsCharges,
                       double4 *__restrict__ coordsDelta,
                       const double4 *__restrict__ coordsDeltaPrevious,
                       const double4 *__restrict__ velMass, const int numAtoms,
                       const double *__restrict__ forces, const int forceStride,
                       const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    const double fx = forces[0 * forceStride + i];
    const double fy = forces[1 * forceStride + i];
    const double fz = forces[2 * forceStride + i];
    const double fact = timeStep2 * velMass[i].w;

    coordsDelta[i].x = coordsDeltaPrevious[i].x - fact * fx;
    coordsDelta[i].y = coordsDeltaPrevious[i].y - fact * fy;
    coordsDelta[i].z = coordsDeltaPrevious[i].z - fact * fz;

    coordsCharges[i].x += coordsDelta[i].x;
    coordsCharges[i].y += coordsDelta[i].y;
    coordsCharges[i].z += coordsDelta[i].z;
  }

  return;
}

__global__ static void UpdateCoordsDeltaAfterHolonomicConstraintKernel(
    double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ coordsCharges,
    const double4 *__restrict__ coordsRef, const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    coordsDelta[i].x = coordsCharges[i].x - coordsRef[i].x;
    coordsDelta[i].y = coordsCharges[i].y - coordsRef[i].y;
    coordsDelta[i].z = coordsCharges[i].z - coordsRef[i].z;
  }

  return;
}

__global__ static void
ComputeOldKineticEnergyKernel(double *__restrict__ kineticEnergy,
                              const double4 *__restrict__ velMass,
                              const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  double ke = 0.0;
  for (int i = idx; i < numAtoms; i += stride) {
    ke += 0.5 *
          ((velMass[i].x * velMass[i].x) + (velMass[i].y * velMass[i].y) +
           (velMass[i].z * velMass[i].z)) /
          velMass[i].w;
  }

  ke = BlockReduceSum<double>(ke);

  if (threadIdx.x == 0)
    atomicAdd(kineticEnergy, ke);

  return;
}

__global__ static void
ComputeKineticEnergyKernel(double *__restrict__ kineticEnergy,
                           const double4 *__restrict__ velMass,
                           const double4 *__restrict__ coordsDelta,
                           const double4 *__restrict__ coordsDeltaPrevious,
                           const int numAtoms, const double timeStep) {
  constexpr double oneThird = 1.0 / 3.0;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double invTimeStep = 1.0 / timeStep;

  double ke = 0.0;
  for (int i = idx; i < numAtoms; i += stride) {
    const double4 u1 =
        make_double4(coordsDeltaPrevious[i].x * invTimeStep,
                     coordsDeltaPrevious[i].y * invTimeStep,
                     coordsDeltaPrevious[i].z * invTimeStep, 0.0);
    const double4 v = velMass[i];
    const double4 u2 = make_double4(coordsDelta[i].x * invTimeStep,
                                    coordsDelta[i].y * invTimeStep,
                                    coordsDelta[i].z * invTimeStep, 0.0);

    const double oldHalfStepKineticEnergy =
        0.5 * ((u1.x * u1.x) + (u1.y * u1.y) + (u1.z * u1.z)) / v.w;
    const double onStepKineticEnergy =
        0.5 * ((v.x * v.x) + (v.y * v.y) + (v.z * v.z)) / v.w;
    const double newHalfStepKineticEnergy =
        0.5 * ((u2.x * u2.x) + (u2.y * u2.y) + (u2.z * u2.z)) / v.w;

    ke += oneThird * (oldHalfStepKineticEnergy + onStepKineticEnergy +
                      newHalfStepKineticEnergy);
  }

  ke = BlockReduceSum<double>(ke);

  if (threadIdx.x == 0)
    atomicAdd(kineticEnergy, ke);

  return;
}

__global__ static void UpdateNoseHooverPistonKernel(
    double *__restrict__ noseHooverPistonForce,
    double *__restrict__ noseHooverPistonForcePrevious,
    double *__restrict__ noseHooverPistonVelocity,
    const double *__restrict__ noseHooverPistonVelocityPrevious,
    const double *__restrict__ noseHooverPistonMass,
    const double *__restrict__ kineticEnergy,
    const double referenceKineticEnergy, const double timeStep) {
  if (threadIdx.x == 0) {
    // Actually store the change in velocity (not the force)
    noseHooverPistonForce[0] = 2.0 * timeStep *
                               (kineticEnergy[0] - referenceKineticEnergy) /
                               noseHooverPistonMass[0];

    if (noseHooverPistonForcePrevious[0] == 0.0)
      noseHooverPistonForcePrevious[0] = noseHooverPistonForce[0];

    noseHooverPistonVelocity[0] =
        noseHooverPistonVelocityPrevious[0] +
        0.5 * (noseHooverPistonForce[0] + noseHooverPistonForcePrevious[0]);
  }

  return;
}

__global__ static void PredictorCorrectorKernel(
    double4 *__restrict__ coordsCharges, double4 *__restrict__ velMass,
    double4 *__restrict__ coordsDeltaPredicted,
    const double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ coordsDeltaPrevious,
    const double4 *__restrict__ coordsRef, const int numAtoms,
    const double *__restrict__ noseHooverPistonVelocity,
    const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    const double fact = 0.5 / timeStep;
    const double onStepVelocityX =
        fact * (coordsDeltaPrevious[i].x + coordsDeltaPredicted[i].x);
    const double onStepVelocityY =
        fact * (coordsDeltaPrevious[i].y + coordsDeltaPredicted[i].y);
    const double onStepVelocityZ =
        fact * (coordsDeltaPrevious[i].z + coordsDeltaPredicted[i].z);
    const double nhFact = timeStep * timeStep * noseHooverPistonVelocity[0];

    coordsDeltaPredicted[i].x = coordsDelta[i].x - nhFact * onStepVelocityX;
    coordsDeltaPredicted[i].y = coordsDelta[i].y - nhFact * onStepVelocityY;
    coordsDeltaPredicted[i].z = coordsDelta[i].z - nhFact * onStepVelocityZ;

    velMass[i].x = onStepVelocityX;
    velMass[i].y = onStepVelocityY;
    velMass[i].z = onStepVelocityZ;

    coordsCharges[i].x = coordsRef[i].x + coordsDeltaPredicted[i].x;
    coordsCharges[i].y = coordsRef[i].y + coordsDeltaPredicted[i].y;
    coordsCharges[i].z = coordsRef[i].z + coordsDeltaPredicted[i].z;
  }

  return;
}

__global__ static void
OnStepVelocityKernel(double4 *__restrict__ velMass,
                     const double4 *__restrict__ coordsDelta,
                     const double4 *__restrict__ coordsDeltaPrevious,
                     const int numAtoms, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double fact = 0.5 / timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    velMass[i].x = fact * (coordsDelta[i].x + coordsDeltaPrevious[i].x);
    velMass[i].y = fact * (coordsDelta[i].y + coordsDeltaPrevious[i].y);
    velMass[i].z = fact * (coordsDelta[i].z + coordsDeltaPrevious[i].z);
  }

  return;
}

__global__ static void
UpdateAverageTemperatureKernel(double *__restrict__ averageTemperature,
                               const double *__restrict__ kineticEnergy,
                               const int numDegreesOfFreedom,
                               const double kBoltz, const int step) {
  if (threadIdx.x == 0) {
    const double s = static_cast<double>(step + 1);
    const double ndegf = static_cast<double>(numDegreesOfFreedom);
    const double temperature = kineticEnergy[0] / (0.5 * ndegf * kBoltz);
    const double delta0 = temperature - averageTemperature[0];
    averageTemperature[0] += delta0 / s;
    // const double delta1 = temperature - averageTemperature[0];
    // varianceTemperature[0] += delta0 * delta1;
  }
  return;
}

void CudaNoseHooverThermostatIntegrator::propagateOneStep(void) {
  const int numDegreesOfFreedom = m_Context->getDegreesOfFreedom();
  const double referenceKineticEnergy =
      0.5 * static_cast<double>(numDegreesOfFreedom) *
      charmm::constants::kBoltz * m_ReferenceTemperature;

  const int numAtoms = m_Context->getNumAtoms();
  double4 *coordsCharges = m_Context->getCoordinatesCharges().getDeviceData();
  float4 *xyzq = m_Context->getXYZQ()->getDeviceXYZQ();
  double4 *velMass = m_Context->getVelocityMass().getDeviceData();
  const int forceStride = m_Context->getForceStride();
  double *forces = m_Context->getForces()->xyz();

  constexpr int numThreads = 256;
  const int numBlocks = numAtoms / numThreads + 1;

  if (m_StepsSinceNeighborListUpdate % m_NonbondedListUpdateFrequency == 0) {
    if (m_Context->getForceManager()->getPeriodicBoundaryCondition() ==
        PBC::P21) {
      // Find a better place for this
      int numGroups =
          m_Context->getForceManager()->getPSF()->getGroups().size();
      int2 *groups =
          m_Context->getForceManager()->getPSF()->getGroups().getDeviceData();
      float boxDimX = static_cast<float>(m_Context->getBoxDimensions()[0]);

      const int numThreads = 256;
      const int numBlocks = numGroups / numThreads + 1;
      InvertDeltaAsymmetricKernel<<<numBlocks, numThreads, 0,
                                    *m_IntegratorStream>>>(
          m_CoordsDeltaPrevious.getDeviceData(), xyzq, groups, numGroups,
          boxDimX);
      cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
    }
    m_Context->resetNeighborList();
  }

  if (m_CurrentPropagatedStep % m_RemoveCenterOfMassFrequency == 0)
    this->removeCenterOfMassMotion();

  copy_DtoD_async<double4>(coordsCharges, m_CoordsRef.getDeviceData(), numAtoms,
                           *m_IntegratorStream);

  if ((m_DebugPrintFrequency > 0) &&
      (m_CurrentPropagatedStep % m_DebugPrintFrequency == 0)) {
    m_Context->calculateForces(false, true, false);
  } else {
    m_Context->calculateForces(false, false, false);
  }

  copy_DtoD_async<double>(m_NoseHooverPistonVelocity.getDeviceData(),
                          m_NoseHooverPistonVelocityPrevious.getDeviceData(), 1,
                          *m_IntegratorStream);
  copy_DtoD_async<double>(m_NoseHooverPistonForce.getDeviceData(),
                          m_NoseHooverPistonForcePrevious.getDeviceData(), 1,
                          *m_IntegratorStream);

  HalfStepVelocityKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      coordsCharges, m_CoordsDelta.getDeviceData(),
      m_CoordsDeltaPrevious.getDeviceData(), velMass, numAtoms, forces,
      forceStride, m_TimeStep);

  if (m_UsingHolonomicConstraints) {
    // JEG250506: Need sync here, otherwise race condition? I don't get it
    // either...
    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());

    UpdateCoordsDeltaAfterHolonomicConstraintKernel<<<numBlocks, numThreads, 0,
                                                      *m_IntegratorStream>>>(
        m_CoordsDelta.getDeviceData(), coordsCharges,
        m_CoordsRef.getDeviceData(), numAtoms);
  }

  copy_DtoD_async<double4>(m_CoordsDelta.getDeviceData(),
                           m_CoordsDeltaPredicted.getDeviceData(), numAtoms,
                           *m_IntegratorStream);

  for (int iter = 0; iter < m_MaxPredictorCorrectorIterations; iter++) {
    cudaCheck(
        cudaMemsetAsync(static_cast<void *>(m_KineticEnergy.getDeviceData()), 0,
                        sizeof(double), *m_IntegratorStream));

    if (m_UseOldTemperature) {
      ComputeOldKineticEnergyKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
          m_KineticEnergy.getDeviceData(), velMass, numAtoms);
    } else {
      ComputeKineticEnergyKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
          m_KineticEnergy.getDeviceData(), velMass,
          m_CoordsDelta.getDeviceData(), m_CoordsDeltaPrevious.getDeviceData(),
          numAtoms, m_TimeStep);
    }

    UpdateNoseHooverPistonKernel<<<1, 32, 0, *m_IntegratorStream>>>(
        m_NoseHooverPistonForce.getDeviceData(),
        m_NoseHooverPistonForcePrevious.getDeviceData(),
        m_NoseHooverPistonVelocity.getDeviceData(),
        m_NoseHooverPistonVelocityPrevious.getDeviceData(),
        m_NoseHooverPistonMass.getDeviceData(), m_KineticEnergy.getDeviceData(),
        referenceKineticEnergy, m_TimeStep);

    PredictorCorrectorKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
        coordsCharges, velMass, m_CoordsDeltaPredicted.getDeviceData(),
        m_CoordsDelta.getDeviceData(), m_CoordsDeltaPrevious.getDeviceData(),
        m_CoordsRef.getDeviceData(), numAtoms,
        m_NoseHooverPistonVelocity.getDeviceData(), m_TimeStep);
  }

  if (m_UsingHolonomicConstraints) {
    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());

    UpdateCoordsDeltaAfterHolonomicConstraintKernel<<<numBlocks, numThreads, 0,
                                                      *m_IntegratorStream>>>(
        m_CoordsDeltaPredicted.getDeviceData(), coordsCharges,
        m_CoordsRef.getDeviceData(), numAtoms);
  }

  copy_DtoD_async<double4>(m_CoordsDeltaPredicted.getDeviceData(),
                           m_CoordsDelta.getDeviceData(), numAtoms,
                           *m_IntegratorStream);

  OnStepVelocityKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      velMass, m_CoordsDelta.getDeviceData(),
      m_CoordsDeltaPrevious.getDeviceData(), numAtoms, m_TimeStep);

  cudaCheck(
      cudaMemsetAsync(static_cast<void *>(m_KineticEnergy.getDeviceData()), 0,
                      sizeof(double), *m_IntegratorStream));

  if (m_UseOldTemperature) {
    ComputeOldKineticEnergyKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
        m_KineticEnergy.getDeviceData(), velMass, numAtoms);
  } else {
    ComputeKineticEnergyKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
        m_KineticEnergy.getDeviceData(), velMass, m_CoordsDelta.getDeviceData(),
        m_CoordsDeltaPrevious.getDeviceData(), numAtoms, m_TimeStep);
  }

  UpdateNoseHooverPistonKernel<<<1, 32, 0, *m_IntegratorStream>>>(
      m_NoseHooverPistonForce.getDeviceData(),
      m_NoseHooverPistonForcePrevious.getDeviceData(),
      m_NoseHooverPistonVelocity.getDeviceData(),
      m_NoseHooverPistonVelocityPrevious.getDeviceData(),
      m_NoseHooverPistonMass.getDeviceData(), m_KineticEnergy.getDeviceData(),
      referenceKineticEnergy, m_TimeStep);

  if (m_UseOldTemperature) {
    UpdateAverageTemperatureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
        m_AverageOldTemperature.getDeviceData(),
        m_KineticEnergy.getDeviceData(), numDegreesOfFreedom,
        charmm::constants::kBoltz, m_AverageWindowSize);

    cudaCheck(
        cudaMemsetAsync(static_cast<void *>(m_KineticEnergy.getDeviceData()), 0,
                        sizeof(double), *m_IntegratorStream));

    ComputeKineticEnergyKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
        m_KineticEnergy.getDeviceData(), velMass, m_CoordsDelta.getDeviceData(),
        m_CoordsDeltaPrevious.getDeviceData(), numAtoms, m_TimeStep);

    UpdateAverageTemperatureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
        m_AverageTemperature.getDeviceData(), m_KineticEnergy.getDeviceData(),
        numDegreesOfFreedom, charmm::constants::kBoltz, m_AverageWindowSize);
  } else {
    UpdateAverageTemperatureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
        m_AverageTemperature.getDeviceData(), m_KineticEnergy.getDeviceData(),
        numDegreesOfFreedom, charmm::constants::kBoltz, m_AverageWindowSize);

    cudaCheck(
        cudaMemsetAsync(static_cast<void *>(m_KineticEnergy.getDeviceData()), 0,
                        sizeof(double), *m_IntegratorStream));

    ComputeOldKineticEnergyKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
        m_KineticEnergy.getDeviceData(), velMass, numAtoms);

    UpdateAverageTemperatureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
        m_AverageOldTemperature.getDeviceData(),
        m_KineticEnergy.getDeviceData(), numDegreesOfFreedom,
        charmm::constants::kBoltz, m_AverageWindowSize);
  }

  m_AverageWindowSize++;

  UpdateSinglePrecisionCoordinatesKernel<<<numBlocks, numThreads, 0,
                                           *m_IntegratorStream>>>(
      xyzq, coordsCharges, numAtoms);

  copy_DtoD_async<double4>(m_CoordsDelta.getDeviceData(),
                           m_CoordsDeltaPrevious.getDeviceData(), numAtoms,
                           *m_IntegratorStream);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  return;
}

double CudaNoseHooverThermostatIntegrator::computeNoseHooverPistonMass(void) {
  if (m_Context == nullptr)
    return -9999.9999;

  CudaContainer<double4> velMass = m_Context->getVelocityMass();
  velMass.transferToHost();

  double totalMass = 0.0;
  for (std::size_t i = 0; i < velMass.size(); i++)
    totalMass += (1.0 / velMass[i].w);

  return (0.2 * totalMass);
}

void CudaNoseHooverThermostatIntegrator::removeCenterOfMassMotion(void) {
  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  auto pbc = m_Context->getForceManager()->getPeriodicBoundaryCondition();
  int numAtoms = m_Context->getNumAtoms();
  CudaContainer<double4> velMass = m_Context->getVelocityMass();

  velMass.transferToHost();
  m_CoordsDeltaPrevious.transferToHost();

  double3 com = make_double3(0.0, 0.0, 0.0);
  double totalMass = 0.0;
  for (int i = 0; i < numAtoms; i++) {
    double mass = 1.0 / velMass[i].w;

    com.x += m_CoordsDeltaPrevious[i].x * mass;
    if (pbc == PBC::P21) {
      com.y -= m_CoordsDeltaPrevious[i].y * mass;
      com.z -= m_CoordsDeltaPrevious[i].z * mass;
    } else {
      com.y += m_CoordsDeltaPrevious[i].y * mass;
      com.z += m_CoordsDeltaPrevious[i].z * mass;
    }

    totalMass += mass;
  }
  com.x /= totalMass;
  com.y /= totalMass;
  com.z /= totalMass;

  for (int i = 0; i < numAtoms; i++) {
    m_CoordsDeltaPrevious[i].x -= com.x;
    if (pbc == PBC::P21) {
      m_CoordsDeltaPrevious[i].y += com.y;
      m_CoordsDeltaPrevious[i].z += com.z;
    } else {
      m_CoordsDeltaPrevious[i].y -= com.y;
      m_CoordsDeltaPrevious[i].z -= com.z;
    }
  }
  m_CoordsDeltaPrevious.transferToDevice();

  return;
}
