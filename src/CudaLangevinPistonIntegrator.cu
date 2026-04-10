// BEGINLICENSE
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "Constants.h"
#include "CudaLangevinPistonIntegrator.h"
#include "gpu_utils.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

/**
 * @brief Uses the BBK algorithm to update crystal dimensions for pressure
 * control
 */
CudaLangevinPistonIntegrator::CudaLangevinPistonIntegrator(
    const double timeStep)
    : CudaIntegrator(timeStep) {
  m_UsingHolonomicConstraints = true;

  m_UsingNoseHooverThermostat = true;

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

  // Allocate pressure variables
  m_HolonomicConstraintVirial.resize(9);
  m_KineticPressureTensor.resize(9);
  m_PressureTensor.resize(9);
  m_PressureScalar.resize(1);
  m_ReferencePressureTensor.resize(9);
  m_DeltaPressureTensor.resize(9);
  m_DeltaKineticPressureTensor.resize(9);
  m_StaticDeltaPressureTensor.resize(9);

  // Set pressure variables to default values
  m_CrystalType = CRYSTAL::NONE;
  m_LangevinPistonDegreesOfFreedom = -1;
  m_HolonomicConstraintVirial.setToValue(0.0);
  m_KineticPressureTensor.setToValue(0.0);
  m_PressureTensor.setToValue(0.0);
  m_PressureScalar.setToValue(0.0);
  m_ReferencePressureTensor.set({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
  m_DeltaPressureTensor.setToValue(0.0);
  m_DeltaKineticPressureTensor.setToValue(0.0);
  m_StaticDeltaPressureTensor.setToValue(0.0);

  m_Pgamma = 0.0;

  m_OnStepCrystalFactor.resize(3);
  m_HalfStepCrystalFactor.resize(3);

  m_OnStepCrystalFactor.setToValue(0.0);
  m_HalfStepCrystalFactor.setToValue(0.0);

  std::random_device rd{};
  m_Seed = rd();
  m_RngSequencePos = 0;
  m_RngStates = nullptr;

  m_ConstantSurfaceTensionFlag = false;
  m_SurfaceTension.resize(1);

  m_MaxPredictorCorrectorIterations = 3;

  // 0 -> New "JUNG" T
  // 1 -> Old T
  m_AverageWindowSize = 0;
  m_KineticEnergy.resize(2);
  m_AverageTemperature.resize(2);
  m_AveragePressureTensor.resize(9);
  m_AveragePressureScalar.resize(1);

  m_KineticEnergy.setToValue(0.0);
  m_AverageTemperature.setToValue(0.0);
  m_AveragePressureTensor.setToValue(0.0);
  m_AveragePressureScalar.setToValue(0.0);

  m_UsingOldTemperature = false;
}

CudaLangevinPistonIntegrator::~CudaLangevinPistonIntegrator(void) {
  this->dealloc();
}

void CudaLangevinPistonIntegrator::useNoseHooverThermostat(
    const bool usingNoseHooverThermostat) {
  m_UsingNoseHooverThermostat = usingNoseHooverThermostat;
  return;
}

void CudaLangevinPistonIntegrator::setReferenceTemperature(
    const double referenceTemperature) {
  m_ReferenceTemperature = referenceTemperature;

  // If the temperature changes, the friction variables need to be updated
  double oldGamma = m_Pgamma;
  this->setLangevinPistonFriction(oldGamma);

  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonMass(
    const double noseHooverPistonMass) {
  m_NoseHooverPistonMass.setToValue(noseHooverPistonMass);
  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonVelocity(
    const double noseHooverPistonVelocity) {
  m_NoseHooverPistonVelocity.setToValue(noseHooverPistonVelocity);
  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonVelocityPrevious(
    const double noseHooverPistonVelocityPrevious) {
  m_NoseHooverPistonVelocityPrevious.setToValue(
      noseHooverPistonVelocityPrevious);
  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonForce(
    const double noseHooverPistonForce) {
  m_NoseHooverPistonForce.setToValue(noseHooverPistonForce);
  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonForcePrevious(
    const double noseHooverPistonForcePrevious) {
  m_NoseHooverPistonForcePrevious.setToValue(noseHooverPistonForcePrevious);
  return;
}

void CudaLangevinPistonIntegrator::setMaxPredictorCorrectorIterations(
    const int maxPredictorCorrectorIterations) {
  m_MaxPredictorCorrectorIterations = maxPredictorCorrectorIterations;
  return;
}

void CudaLangevinPistonIntegrator::useOldTemperature(
    const bool usingOldTemperature) {
  m_UsingOldTemperature = usingOldTemperature;
  return;
}

void CudaLangevinPistonIntegrator::setPressure(
    const std::vector<double> &referencePressureTensor) {
  if (referencePressureTensor.size() != 9) {
    throw std::invalid_argument("CudaLangevinPistonIntegrator::setPressure: "
                                "reference pressure must be of length 9 (XX, "
                                "XY, XZ, YX, YY, YZ, ZX, ZY, ZZ)");
  }
  m_ReferencePressureTensor = referencePressureTensor;
  return;
}

void CudaLangevinPistonIntegrator::setConstantSurfaceTension(
    const bool constantSurfaceTensionFlag) {
  m_ConstantSurfaceTensionFlag = constantSurfaceTensionFlag;
  return;
}

void CudaLangevinPistonIntegrator::setCrystalType(const CRYSTAL crystalType) {
  m_CrystalType = crystalType;

  switch (crystalType) {
  case CRYSTAL::CUBIC:
    m_LangevinPistonDegreesOfFreedom = 1;
    break;
  case CRYSTAL::TETRAGONAL:
    m_LangevinPistonDegreesOfFreedom = 2;
    break;
  case CRYSTAL::ORTHORHOMBIC:
    m_LangevinPistonDegreesOfFreedom = 3;
    break;
  default:
    throw std::invalid_argument(
        "CudaLangevinPiston::setCrystalType: crystalType must be either "
        "CRYSTAL::CUBIC, CRYSTAL::TETRAGONAL, or CRYSTAL::ORTHORHOMBIC");
  }

  this->allocateLangevinPistonVariables();

  // Set default values for Langevin piston mass
  double m = this->computeLangevinPistonMass();
  std::vector<double> mass(m_LangevinPistonDegreesOfFreedom, m);
  this->setLangevinPistonMass(mass);

  // Set default values for Langevin piston friction
  this->setLangevinPistonFriction(0.0);

  return;
}

void CudaLangevinPistonIntegrator::setLangevinPistonMass(
    const std::vector<double> &mass) {
  if (m_LangevinPistonDegreesOfFreedom == -1) {
    throw std::runtime_error(
        "CudaLangevinPistonIntegrator::setLangevinPistonMass: CrystalType must "
        "be set before Langevin piston mass");
  }
  if (static_cast<int>(mass.size()) != m_LangevinPistonDegreesOfFreedom) {
    throw std::invalid_argument(
        "CudaLangevinPistonIntegratoor::setLangevinPistonMass: Length of input "
        "vector must be equal to the degrees of freedom of the Langevin "
        "piston (" +
        std::to_string(m_LangevinPistonDegreesOfFreedom) + ")");
  }

  for (int i = 0; i < m_LangevinPistonDegreesOfFreedom; i++) {
    m_LangevinPistonMass[i] = mass[i];
    if (mass[i] == 0.0)
      m_LangevinPistonInverseMass[i] = 0.0;
    else
      m_LangevinPistonInverseMass[i] = 1.0 / mass[i];
  }
  m_LangevinPistonMass.transferToDevice();
  m_LangevinPistonInverseMass.transferToDevice();

  // If the Langevin piston mass changes, the friction variables need to be
  // updated
  double oldGamma = m_Pgamma;
  this->setLangevinPistonFriction(oldGamma);

  return;
}

void CudaLangevinPistonIntegrator::setLangevinPistonFrictionSeed(
    const std::uint64_t seed) {
  m_Seed = seed;
  this->initializeRng();
  return;
}

void CudaLangevinPistonIntegrator::setRngSequencePos(
    const unsigned long long int rngSequencePos) {
  m_RngSequencePos = rngSequencePos;
  this->initializeRng();
  return;
}

void CudaLangevinPistonIntegrator::setLangevinPistonFriction(
    const double pgamma) {
  if (m_LangevinPistonDegreesOfFreedom == -1) {
    throw std::runtime_error(
        "CudaLangevinPistonIntegrator::setPressurePistonFriction: CrystalType "
        "must be set before Langevin piston friction");
  }

  const double kbt = charmm::constants::kBoltz * m_ReferenceTemperature;
  const double pgam = m_Timfac * m_TimeStep * pgamma;

  m_Pgamma = pgamma;
  m_Palpha = (1.0 - 0.5 * pgam) / (1.0 + 0.5 * pgam);
  m_Pbfact = m_TimeStep * m_TimeStep / (1.0 + 0.5 * pgam);
  for (int i = 0; i < m_LangevinPistonDegreesOfFreedom; i++) {
    m_Prfwd[i] = std::sqrt(2.0 * m_LangevinPistonInverseMass[i] * pgam * kbt) /
                 m_TimeStep;
  }
  m_Prfwd.transferToDevice();

  return;
}

void CudaLangevinPistonIntegrator::resetAverages(void) {
  m_AverageWindowSize = 0;
  m_AverageTemperature.setToValue(0.0);
  m_AveragePressureTensor.setToValue(0.0);
  m_AveragePressureScalar.setToValue(0.0);
  return;
}

double CudaLangevinPistonIntegrator::getReferenceTemperature(void) const {
  return m_ReferenceTemperature;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonMass(void) const {
  return m_NoseHooverPistonMass;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonVelocity(void) const {
  return m_NoseHooverPistonVelocity;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonVelocityPrevious(void) const {
  return m_NoseHooverPistonVelocityPrevious;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonForce(void) const {
  return m_NoseHooverPistonForce;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonForcePrevious(void) const {
  return m_NoseHooverPistonForcePrevious;
}

int CudaLangevinPistonIntegrator::getMaxPredictorCorrectorIterations(
    void) const {
  return m_MaxPredictorCorrectorIterations;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getKineticEnergy(void) const {
  return m_KineticEnergy;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getAverageTemperature(void) const {
  return m_AverageTemperature;
}

CRYSTAL CudaLangevinPistonIntegrator::getCrystalType(void) const {
  return m_CrystalType;
}

std::uint64_t
CudaLangevinPistonIntegrator::getLangevinPistonFrictionSeed(void) const {
  return m_Seed;
}

unsigned long long int
CudaLangevinPistonIntegrator::getRngSequencePos(void) const {
  return m_RngSequencePos;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getReferencePressureTensor(void) const {
  return m_ReferencePressureTensor;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonMass(void) const {
  return m_LangevinPistonMass;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonOnStepPosition(void) const {
  return m_LangevinPistonOnStepPosition;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonHalfStepPosition(void) const {
  return m_LangevinPistonHalfStepPosition;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonOnStepVelocity(void) const {
  return m_LangevinPistonOnStepVelocity;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonHalfStepVelocity(void) const {
  return m_LangevinPistonHalfStepVelocity;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPosition(void) const {
  return m_LangevinPistonDeltaPosition;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPositionPrevious(
    void) const {
  return m_LangevinPistonDeltaPositionPrevious;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPositionPredicted(
    void) const {
  return m_LangevinPistonDeltaPositionPredicted;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPressure(void) const {
  return m_LangevinPistonDeltaPressure;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getInstantaneousPressureTensor(void) const {
  return m_PressureTensor;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getInstantaneousPressureScalar(void) const {
  return m_PressureScalar;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getAveragePressureTensor(void) const {
  return m_AveragePressureTensor;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getAveragePressureScalar(void) const {
  return m_AveragePressureScalar;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonMass(void) {
  return m_NoseHooverPistonMass;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonVelocity(void) {
  return m_NoseHooverPistonVelocity;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonVelocityPrevious(void) {
  return m_NoseHooverPistonVelocityPrevious;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonForce(void) {
  return m_NoseHooverPistonForce;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonForcePrevious(void) {
  return m_NoseHooverPistonForcePrevious;
}

CudaContainer<double> &CudaLangevinPistonIntegrator::getKineticEnergy(void) {
  return m_KineticEnergy;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getAverageTemperature(void) {
  return m_AverageTemperature;
}

double CudaLangevinPistonIntegrator::getInstantaneousTemperature(void) {
  const double ndegf = static_cast<double>(m_Context->getDegreesOfFreedom());
  m_KineticEnergy.transferToHost();
  if (m_UsingOldTemperature)
    return (m_KineticEnergy[1] / (0.5 * ndegf * charmm::constants::kBoltz));
  return (m_KineticEnergy[0] / (0.5 * ndegf * charmm::constants::kBoltz));
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getReferencePressureTensor(void) {
  return m_ReferencePressureTensor;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonMass(void) {
  return m_LangevinPistonMass;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonOnStepPosition(void) {
  return m_LangevinPistonOnStepPosition;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonHalfStepPosition(void) {
  return m_LangevinPistonHalfStepPosition;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonOnStepVelocity(void) {
  return m_LangevinPistonOnStepVelocity;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonHalfStepVelocity(void) {
  return m_LangevinPistonHalfStepVelocity;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPosition(void) {
  return m_LangevinPistonDeltaPosition;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPositionPrevious(void) {
  return m_LangevinPistonDeltaPositionPrevious;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPositionPredicted(void) {
  return m_LangevinPistonDeltaPositionPredicted;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getInstantaneousPressureTensor(void) {
  return m_PressureTensor;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getInstantaneousPressureScalar(void) {
  return m_PressureScalar;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPressure(void) {
  return m_LangevinPistonDeltaPressure;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getAveragePressureTensor(void) {
  return m_AveragePressureTensor;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getAveragePressureScalar(void) {
  return m_AveragePressureScalar;
}

double CudaLangevinPistonIntegrator::getInstantaneousSurfaceTension(void) {
  m_PressureTensor.transferToHost();
  return (0.5 * m_Context->getBoxDimensions()[2] *
          (m_PressureTensor[8] -
           0.5 * (m_PressureTensor[0] + m_PressureTensor[4])) /
          charmm::constants::surfaceTensionFactor);
  return -9999.9999;
}

/** @brief Updates the single-precision coordinates container (xyzq)
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

void CudaLangevinPistonIntegrator::initialize(void) {
  const int numAtoms = m_Context->getNumAtoms();
  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

  this->initializeRng();

  m_CoordsDeltaPredicted.resize(numAtoms);

  m_HolonomicConstraintForces.resize(numAtoms);

  if (m_NoseHooverPistonMass[0] == -9999.9999)
    m_NoseHooverPistonMass.setToValue(this->computeNoseHooverPistonMass());

  // JEG260112: setCrystalType initializes all of the default values for the
  // Langevin piston, so we shouldn't need to do any other initialization
  if (m_LangevinPistonDegreesOfFreedom == -1) {
    throw std::runtime_error(
        "CudaLangevinPistonIntegrator::initialize: CrystalType must be set "
        "before initialization (try setting the crystal type before setting "
        "the Charmm Context)");
  }

  if (m_LangevinPistonMass[0] == -9999.9999) {
    double m = this->computeLangevinPistonMass();
    std::vector<double> mass(m_LangevinPistonDegreesOfFreedom, m);
    this->setLangevinPistonMass(mass);
  }

  double4 *coordsCharges = m_Context->getCoordinatesCharges().getDeviceData();
  float4 *xyzq = m_Context->getXYZQ()->getDeviceXYZQ();

  if (m_UsingHolonomicConstraints) {
    copy_DtoD_async<double4>(coordsCharges, m_CoordsRef.getDeviceData(),
                             numAtoms, *m_IntegratorStream);

    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());

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
  const int forceStride = m_Context->getForceStride();

  InitializationKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      m_CoordsDelta.getDeviceData(), m_CoordsDeltaPrevious.getDeviceData(),
      velMass, numAtoms, forces, forceStride, m_TimeStep);

  if (m_UsingHolonomicConstraints) {
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

void CudaLangevinPistonIntegrator::initializeFromRestartFile(
    const std::string &rstFileName) {
  // Ensure that the CharmmContext has been set before we try to initialize
  if (m_Context == nullptr) {
    throw std::runtime_error(
        "CharmmContext must be set before initializing from a restart file");
  }

  std::ifstream fin(rstFileName);
  if (!fin.is_open())
    throw std::runtime_error("Could not open file \"" + rstFileName + "\"");

  std::string line = "";

  // Get crystal type and check if we need to use the RNG state
  std::string crystalString = "NONE";
  bool isApoRstFile = false;
  line.clear();
  std::getline(fin, line);
  crystalString = line.substr(18, 4);
  if (line.length() >= 33) // Check for APO flag
    isApoRstFile = (line.substr(30, 3) == "APO");

  if (crystalString == "CUBI")
    this->setCrystalType(CRYSTAL::CUBIC);
  else if (crystalString == "TETR")
    this->setCrystalType(CRYSTAL::TETRAGONAL);
  else if (crystalString == "ORTH")
    this->setCrystalType(CRYSTAL::ORTHORHOMBIC);

  bool foundSection = false;

  // Find CRYSTAL PARAMETERS section
  while (!fin.eof()) {
    line.clear();
    std::getline(fin, line);
    if (line == " !CRYSTAL PARAMETERS") {
      foundSection = true;
      break;
    }
  }
  if (!foundSection)
    throw std::runtime_error("Could not find !CRYSTAL PARAMETERS section");

  // Parse CRYSTAL PARAMETERS section
  std::vector<double> XTLABC(6, 0.0);
  std::vector<double> HDOT(6, 0.0);
  // double PNH = 0.0,
  double PNHV = 0.0, PNHF = 0.0;
  // std::vector<double> UC1A(6, 0.0), UC2A(6, 0.0), UC1B(6, 0.0), UC2B(6, 0.0);
  // double GRAD1A = 0.0, GRAD1B = 0.0, GRAD2A = 0.0, GRAD2B = 0.0;

  line.clear();
  std::getline(fin, line);
  XTLABC[0] = apo::fortSciStrToCDouble(line.substr(0, 22));
  XTLABC[1] = apo::fortSciStrToCDouble(line.substr(22, 22));
  XTLABC[2] = apo::fortSciStrToCDouble(line.substr(44, 22));
  line.clear();
  std::getline(fin, line);
  XTLABC[3] = apo::fortSciStrToCDouble(line.substr(0, 22));
  XTLABC[4] = apo::fortSciStrToCDouble(line.substr(22, 22));
  XTLABC[5] = apo::fortSciStrToCDouble(line.substr(44, 22));

  line.clear();
  std::getline(fin, line);
  HDOT[0] = apo::fortSciStrToCDouble(line.substr(0, 22));
  HDOT[1] = apo::fortSciStrToCDouble(line.substr(22, 22));
  HDOT[2] = apo::fortSciStrToCDouble(line.substr(44, 22));
  line.clear();
  std::getline(fin, line);
  HDOT[3] = apo::fortSciStrToCDouble(line.substr(0, 22));
  HDOT[4] = apo::fortSciStrToCDouble(line.substr(22, 22));
  HDOT[5] = apo::fortSciStrToCDouble(line.substr(44, 22));

  line.clear();
  std::getline(fin, line);
  // PNH = apo::fortSciStrToCDouble(line.substr(0, 22)); // Not needed for LP
  PNHV = apo::fortSciStrToCDouble(line.substr(22, 22));
  PNHF = apo::fortSciStrToCDouble(line.substr(44, 22));

  // Not needed for Langevin-Piston
  line.clear();
  std::getline(fin, line);
  // UC1A[0] = apo::fortSciStrToCDouble(line.substr(0, 22));
  // UC1A[1] = apo::fortSciStrToCDouble(line.substr(22, 22));
  // UC1A[2] = apo::fortSciStrToCDouble(line.substr(44, 22));
  line.clear();
  std::getline(fin, line);
  // UC1A[3] = apo::fortSciStrToCDouble(line.substr(0, 22));
  // UC1A[4] = apo::fortSciStrToCDouble(line.substr(22, 22));
  // UC1A[5] = apo::fortSciStrToCDouble(line.substr(44, 22));

  // Not needed for Langevin-Piston
  line.clear();
  std::getline(fin, line);
  // UC2A[0] = apo::fortSciStrToCDouble(line.substr(0, 22));
  // UC2A[1] = apo::fortSciStrToCDouble(line.substr(22, 22));
  // UC2A[2] = apo::fortSciStrToCDouble(line.substr(44, 22));
  line.clear();
  std::getline(fin, line);
  // UC2A[3] = apo::fortSciStrToCDouble(line.substr(0, 22));
  // UC2A[4] = apo::fortSciStrToCDouble(line.substr(22, 22));
  // UC2A[5] = apo::fortSciStrToCDouble(line.substr(44, 22));

  // // Not needed for Langevin-Piston
  line.clear();
  std::getline(fin, line);
  // UC1B[0] = apo::fortSciStrToCDouble(line.substr(0, 22));
  // UC1B[1] = apo::fortSciStrToCDouble(line.substr(22, 22));
  // UC1B[2] = apo::fortSciStrToCDouble(line.substr(44, 22));
  line.clear();
  std::getline(fin, line);
  // UC1B[3] = apo::fortSciStrToCDouble(line.substr(0, 22));
  // UC1B[4] = apo::fortSciStrToCDouble(line.substr(22, 22));
  // UC1B[5] = apo::fortSciStrToCDouble(line.substr(44, 22));

  // Not needed for Langevin-Piston
  line.clear();
  std::getline(fin, line);
  // UC2B[0] = apo::fortSciStrToCDouble(line.substr(0, 22));
  // UC2B[1] = apo::fortSciStrToCDouble(line.substr(22, 22));
  // UC2B[2] = apo::fortSciStrToCDouble(line.substr(44, 22));
  line.clear();
  std::getline(fin, line);
  // UC2B[3] = apo::fortSciStrToCDouble(line.substr(0, 22));
  // UC2B[4] = apo::fortSciStrToCDouble(line.substr(22, 22));
  // UC2B[5] = apo::fortSciStrToCDouble(line.substr(44, 22));

  // Not needed for Langevin-Piston
  line.clear();
  std::getline(fin, line);
  // GRAD1A = apo::fortSciStrToCDouble(line.substr(0, 22));
  // GRAD1B = apo::fortSciStrToCDouble(line.substr(22, 22));
  // GRAD2A = apo::fortSciStrToCDouble(line.substr(44, 22));
  line.clear();
  std::getline(fin, line);
  // GRAD2B = apo::fortSciStrToCDouble(line.substr(0, 22));

  m_Context->setBoxDimensions({XTLABC[0], XTLABC[2], XTLABC[5]});

  switch (m_CrystalType) {
  case CRYSTAL::CUBIC:
    m_LangevinPistonDeltaPosition[0] = HDOT[0];
    break;

  case CRYSTAL::TETRAGONAL:
    m_LangevinPistonDeltaPosition[0] = HDOT[0];
    m_LangevinPistonDeltaPosition[1] = HDOT[1];
    break;

  case CRYSTAL::ORTHORHOMBIC:
    m_LangevinPistonDeltaPosition[0] = HDOT[0];
    m_LangevinPistonDeltaPosition[1] = HDOT[1];
    m_LangevinPistonDeltaPosition[2] = HDOT[2];
    break;

  default:
    throw std::invalid_argument(
        "Unable to use HDOT data for unknown CRYSTAL type");
    break;
  }
  m_LangevinPistonDeltaPosition.transferToDevice();

  m_NoseHooverPistonVelocity[0] = PNHV;
  m_NoseHooverPistonVelocity.transferToDevice();

  m_NoseHooverPistonForce[0] = PNHF;
  m_NoseHooverPistonForce.transferToDevice();

  // Find integer section
  foundSection = false;
  while (!fin.eof()) {
    line.clear();
    std::getline(fin, line);
    if (line == " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL") {
      foundSection = true;
      break;
    }
  }
  if (!foundSection) {
    throw std::runtime_error(
        "Could not find !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL "
        "section");
  }

  int NATOM = 0;
  unsigned long long int NPRIV = 0;
  int NSTEP = 0;
  // int NSAVC = 0; // Not needed for Langevin-Piston
  // int NSAVV = 0;  // Not needed for Langevin-Piston
  // int JHSTRT = 0; // Not needed for Langevin-Piston
  int NDEGF = 0;
  std::uint64_t SEED = 0;
  std::string RNGSTATE = "";

  line.clear();
  std::getline(fin, line);
  NATOM = std::stoi(line.substr(0, 12));
  NPRIV = std::stoull(line.substr(12, 12));
  NSTEP = std::stoi(line.substr(24, 12));
  // NSAVC = std::stoi(line.substr(36, 12));
  // NSAVV = std::stoi(line.substr(48, 12));
  // JHSTRT = std::stoi(line.substr(60, 12));
  NDEGF = std::stoi(line.substr(72, 12));
  SEED = std::stoull(line.substr(84, 22));
  RNGSTATE = line.substr(106, std::string::npos);

  if (NATOM != m_Context->getNumAtoms()) {
    throw std::invalid_argument("NATOM mismatch in restart file \"" +
                                rstFileName + "\"");
  }
  m_TotNumSteps = NPRIV;
  m_NumSteps = NSTEP;
  if (NDEGF != m_Context->getDegreesOfFreedom()) {
    throw std::invalid_argument("NDEGF mismatch in restart file \"" +
                                rstFileName + "\"");
  }
  this->setLangevinPistonFrictionSeed(SEED);
  this->setRngSequencePos((isApoRstFile) ? std::stoull(RNGSTATE) : 0);

  // Skip ENERGIES and STATISTICS section

  // Find XOLD, YOLD, ZOLD section
  foundSection = false;
  while (!fin.eof()) {
    line.clear();
    std::getline(fin, line);
    if (line == " !XOLD, YOLD, ZOLD") {
      foundSection = true;
      break;
    }
  }
  if (!foundSection)
    throw std::runtime_error("Could not find !XOLD, YOLD, ZOLD section");

  // Parse XOLD, YOLD, ZOLD section
  for (int i = 0; i < NATOM; i++) {
    line.clear();
    std::getline(fin, line);
    m_CoordsDeltaPrevious[i].x = apo::fortSciStrToCDouble(line.substr(0, 22));
    m_CoordsDeltaPrevious[i].y = apo::fortSciStrToCDouble(line.substr(22, 22));
    m_CoordsDeltaPrevious[i].z = apo::fortSciStrToCDouble(line.substr(44, 22));
  }
  m_CoordsDeltaPrevious.transferToDevice();

  // Find VX, VY, VZ section
  foundSection = false;
  while (!fin.eof()) {
    line.clear();
    std::getline(fin, line);
    if (line == " !VX, VY, VZ") {
      foundSection = true;
      break;
    }
  }
  if (!foundSection)
    throw std::runtime_error("Could not find !VX, VY, VZ section");

  for (int i = 0; i < NATOM; i++) {
    line.clear();
    std::getline(fin, line);
    m_Context->getVelocityMass()[i].x =
        apo::fortSciStrToCDouble(line.substr(0, 22));
    m_Context->getVelocityMass()[i].y =
        apo::fortSciStrToCDouble(line.substr(22, 22));
    m_Context->getVelocityMass()[i].z =
        apo::fortSciStrToCDouble(line.substr(44, 22));
  }
  m_Context->getVelocityMass().transferToDevice();

  // Find X, Y, Z section
  foundSection = false;
  while (!fin.eof()) {
    line.clear();
    std::getline(fin, line);
    if (line == " !X, Y, Z") {
      foundSection = true;
      break;
    }
  }
  if (!foundSection)
    throw std::runtime_error("Could not find !X, Y, Z section");

  // Parse X, Y, Z section
  for (int i = 0; i < NATOM; i++) {
    line.clear();
    std::getline(fin, line);
    m_Context->getCoordinatesCharges()[i].x =
        apo::fortSciStrToCDouble(line.substr(0, 22));
    m_Context->getCoordinatesCharges()[i].y =
        apo::fortSciStrToCDouble(line.substr(22, 22));
    m_Context->getCoordinatesCharges()[i].z =
        apo::fortSciStrToCDouble(line.substr(44, 22));
  }
  m_Context->getCoordinatesCharges().transferToDevice();

  {
    double4 *dptr = m_Context->getCoordinatesCharges().getDeviceData();
    float4 *fptr = m_Context->getXYZQ()->getDeviceXYZQ();

    constexpr int numThreads = 256;
    const int numBlocks = (NATOM + numThreads - 1) / numThreads;
    UpdateSinglePrecisionCoordinatesKernel<<<numBlocks, numThreads, 0,
                                             *m_IntegratorStream>>>(fptr, dptr,
                                                                    NATOM);
    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  }

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

__global__ static void ComputeHolonomicConstraintForcesKernel(
    double4 *__restrict__ holonomicConstraintForces,
    const double4 *__restrict__ coordsCharges,
    const double4 *__restrict__ coordsRef,
    const double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ velMass, const int numAtoms,
    const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    const double fact = 1.0 / (timeStep2 * velMass[i].w);

    holonomicConstraintForces[i].x =
        fact * (coordsCharges[i].x - coordsRef[i].x - coordsDelta[i].x);
    holonomicConstraintForces[i].y =
        fact * (coordsCharges[i].y - coordsRef[i].y - coordsDelta[i].y);
    holonomicConstraintForces[i].z =
        fact * (coordsCharges[i].z - coordsRef[i].z - coordsDelta[i].z);
  }

  return;
}

__global__ static void ComputeHolonomicConstraintVirialKernel(
    double *__restrict__ holonomicConstraintVirial,
    const double4 *__restrict__ coordsRef,
    const double4 *__restrict__ holonomicConstraintForces, const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  double vir[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = idx; i < numAtoms; i += stride) {
    vir[0] += coordsRef[i].x * holonomicConstraintForces[i].x;
    vir[1] += coordsRef[i].x * holonomicConstraintForces[i].y;
    vir[2] += coordsRef[i].x * holonomicConstraintForces[i].z;
    vir[3] += coordsRef[i].y * holonomicConstraintForces[i].x;
    vir[4] += coordsRef[i].y * holonomicConstraintForces[i].y;
    vir[5] += coordsRef[i].y * holonomicConstraintForces[i].z;
    vir[6] += coordsRef[i].z * holonomicConstraintForces[i].x;
    vir[7] += coordsRef[i].z * holonomicConstraintForces[i].y;
    vir[8] += coordsRef[i].z * holonomicConstraintForces[i].z;
  }

  for (int i = 0; i < 9; i++)
    vir[i] = BlockReduceSum<double>(vir[i]);

  if (threadIdx.x == 0) {
    for (int i = 0; i < 9; i++)
      atomicAdd(holonomicConstraintVirial + i, vir[i]);
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
__global__ static void UpdateVirialWithHolonomicConstraintVirialKernel(
    double *__restrict__ virialTensor,
    const double *__restrict__ holonomicConstraintVirial) {
  if (threadIdx.x < 9)
    virialTensor[threadIdx.x] += holonomicConstraintVirial[threadIdx.x];
  return;
}

/** @brief Computes the kinetic energy contribution to the pressure tensor
 * using previous and next half step velocities. One might think the on-step
 * velocity would be a better thing to use, but it would be unsound (Brooks
 * 1987)
 */
__global__ static void ComputeAverageKineticPressureKernel(
    double *__restrict__ kineticPressureTensor,
    const double4 *__restrict__ velMass,
    const double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ coordsDeltaPrevious, const int numAtoms,
    const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;

  double prs[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = idx; i < numAtoms; i += stride) {
    const double fact = 0.5 / (timeStep2 * velMass[i].w);
    prs[0] += fact * (coordsDelta[i].x * coordsDelta[i].x +
                      coordsDeltaPrevious[i].x * coordsDeltaPrevious[i].x);
    prs[1] += fact * (coordsDelta[i].x * coordsDelta[i].y +
                      coordsDeltaPrevious[i].x * coordsDeltaPrevious[i].y);
    prs[2] += fact * (coordsDelta[i].x * coordsDelta[i].z +
                      coordsDeltaPrevious[i].x * coordsDeltaPrevious[i].z);
    prs[3] += fact * (coordsDelta[i].y * coordsDelta[i].x +
                      coordsDeltaPrevious[i].y * coordsDeltaPrevious[i].x);
    prs[4] += fact * (coordsDelta[i].y * coordsDelta[i].y +
                      coordsDeltaPrevious[i].y * coordsDeltaPrevious[i].y);
    prs[5] += fact * (coordsDelta[i].y * coordsDelta[i].z +
                      coordsDeltaPrevious[i].y * coordsDeltaPrevious[i].z);
    prs[6] += fact * (coordsDelta[i].z * coordsDelta[i].x +
                      coordsDeltaPrevious[i].z * coordsDeltaPrevious[i].x);
    prs[7] += fact * (coordsDelta[i].z * coordsDelta[i].y +
                      coordsDeltaPrevious[i].z * coordsDeltaPrevious[i].y);
    prs[8] += fact * (coordsDelta[i].z * coordsDelta[i].z +
                      coordsDeltaPrevious[i].z * coordsDeltaPrevious[i].z);
  }

  for (int i = 0; i < 9; i++)
    prs[i] = BlockReduceSum<double>(prs[i]);

  if (threadIdx.x == 0) {
    for (int i = 0; i < 9; i++)
      atomicAdd(kineticPressureTensor + i, prs[i]);
  }

  return;
}

__global__ static void UpdatePressureKernel(
    double *__restrict__ pressureTensor, double *__restrict__ pressureScalar,
    double *__restrict__ referencePressureTensor,
    double *__restrict__ deltaPressureTensor,
    const double *__restrict__ virialTensor,
    const double *__restrict__ kineticPressureTensor, const double volumeFactor,
    const double *__restrict__ surfaceTension,
    const double surfaceTensionFactor, const bool constantSurfaceTensionFlag) {
  if (threadIdx.x < 9) {
    pressureTensor[threadIdx.x] =
        volumeFactor *
        (kineticPressureTensor[threadIdx.x] + virialTensor[threadIdx.x]);

    deltaPressureTensor[threadIdx.x] =
        pressureTensor[threadIdx.x] - referencePressureTensor[threadIdx.x];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    *pressureScalar =
        (pressureTensor[0] + pressureTensor[4] + pressureTensor[8]) / 3.0;

    if (constantSurfaceTensionFlag == true) {
      referencePressureTensor[0] =
          referencePressureTensor[8] - surfaceTension[0] * surfaceTensionFactor;
      referencePressureTensor[4] = referencePressureTensor[0];
    }
  }

  return;
}

/** @brief Compute the next-half-step kinetic energy component of the pressure
 * using updated coordsDelta. This is the only component of the pressure that
 * is updated during the pred-corr (the previous-half-step does not change and
 * the virial part is considered constant)
 */
__global__ static void ComputeDeltaKineticPressureKernel(
    double *__restrict__ deltaKineticPressureTensor,
    const double4 *__restrict__ velMass,
    const double4 *__restrict__ coordsDelta, const int numAtoms,
    const double volumeFactor, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;

  double prs[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = idx; i < numAtoms; i += stride) {
    const double fact = 0.5 * volumeFactor / (timeStep2 * velMass[i].w);
    prs[0] += fact * coordsDelta[i].x * coordsDelta[i].x;
    prs[1] += fact * coordsDelta[i].x * coordsDelta[i].y;
    prs[2] += fact * coordsDelta[i].x * coordsDelta[i].z;
    prs[3] += fact * coordsDelta[i].y * coordsDelta[i].x;
    prs[4] += fact * coordsDelta[i].y * coordsDelta[i].y;
    prs[5] += fact * coordsDelta[i].y * coordsDelta[i].z;
    prs[6] += fact * coordsDelta[i].z * coordsDelta[i].x;
    prs[7] += fact * coordsDelta[i].z * coordsDelta[i].y;
    prs[8] += fact * coordsDelta[i].z * coordsDelta[i].z;
  }

  for (int i = 0; i < 9; i++)
    prs[i] = BlockReduceSum<double>(prs[i]);

  if (threadIdx.x == 0) {
    for (int i = 0; i < 9; i++)
      atomicAdd(deltaKineticPressureTensor + i, prs[i]);
  }

  return;
}

__global__ static void ComputeStaticDeltaPressureKernel(
    double *__restrict__ staticDeltaPressureTensor,
    const double *__restrict__ deltaPressureTensor,
    const double *__restrict__ deltaKineticPressureTensor) {
  if (threadIdx.x < 9) {
    staticDeltaPressureTensor[threadIdx.x] =
        deltaPressureTensor[threadIdx.x] -
        deltaKineticPressureTensor[threadIdx.x];
  }
  return;
}

__global__ static void UpdateLangevinPistonKernel(
    double *__restrict__ langevinPistonOnStepPosition,
    double *__restrict__ langevinPistonHalfStepPosition,
    double *__restrict__ langevinPistonDeltaPosition,
    double *__restrict__ langevinPistonDeltaPositionPrevious,
    double *__restrict__ langevinPistonOnStepVelocity,
    double *__restrict__ langevinPistonDeltaPressure,
    double *__restrict__ onStepCrystalFactor,
    double *__restrict__ halfStepCrystalFactor,
    double *__restrict__ boxDimensions,
    curandStatePhilox4_32_10_t *__restrict__ rngStates,
    const double *__restrict__ langevinPistonInverseMass,
    const double *__restrict__ deltaPressureTensor,
    const double *__restrict__ prfwd, const double palpha, const double pbfact,
    const double volumeFactor, const int dof, const CRYSTAL crystalType,
    const double timeStep) {
  if (threadIdx.x == 0) {
    switch (crystalType) {
    case CRYSTAL::CUBIC:
      langevinPistonDeltaPressure[0] =
          (deltaPressureTensor[0] + deltaPressureTensor[4] +
           deltaPressureTensor[8]) /
          boxDimensions[0];
      break;

    case CRYSTAL::TETRAGONAL:
      langevinPistonDeltaPressure[0] =
          (deltaPressureTensor[0] + deltaPressureTensor[4]) / boxDimensions[0];
      langevinPistonDeltaPressure[1] =
          deltaPressureTensor[8] / boxDimensions[2];
      break;

    case CRYSTAL::ORTHORHOMBIC:
      langevinPistonDeltaPressure[0] =
          deltaPressureTensor[0] / boxDimensions[0];
      langevinPistonDeltaPressure[1] =
          deltaPressureTensor[4] / boxDimensions[1];
      langevinPistonDeltaPressure[2] =
          deltaPressureTensor[8] / boxDimensions[2];
      break;

    default:
      break;
    }

    for (int i = 0; i < dof; i++) {
      curandStatePhilox4_32_10_t rngState = rngStates[i];

      langevinPistonDeltaPositionPrevious[i] = langevinPistonDeltaPosition[i];

      langevinPistonDeltaPosition[i] =
          palpha * langevinPistonDeltaPosition[i] +
          langevinPistonInverseMass[i] * langevinPistonDeltaPressure[i] *
              volumeFactor +
          pbfact * prfwd[i] * curand_normal_double(&rngState);

      langevinPistonOnStepVelocity[i] =
          0.5 *
          (langevinPistonDeltaPosition[i] +
           langevinPistonDeltaPositionPrevious[i]) /
          timeStep;

      rngStates[i] = rngState;
    }

    switch (crystalType) {
    case CRYSTAL::CUBIC:
      langevinPistonOnStepPosition[0] =
          (boxDimensions[0] + boxDimensions[1] + boxDimensions[2]) / 3.0;
      break;

    case CRYSTAL::TETRAGONAL:
      langevinPistonOnStepPosition[0] =
          (boxDimensions[0] + boxDimensions[1]) / 2.0;
      langevinPistonOnStepPosition[1] = boxDimensions[2];
      break;

    case CRYSTAL::ORTHORHOMBIC:
      langevinPistonOnStepPosition[0] = boxDimensions[0];
      langevinPistonOnStepPosition[1] = boxDimensions[1];
      langevinPistonOnStepPosition[2] = boxDimensions[2];
      break;

    default:
      break;
    }

    for (int i = 0; i < dof; i++) {
      langevinPistonOnStepPosition[i] += langevinPistonDeltaPosition[i];

      langevinPistonHalfStepPosition[i] = langevinPistonOnStepPosition[i] -
                                          0.5 * langevinPistonDeltaPosition[i];
    }

    switch (crystalType) {
    case CRYSTAL::CUBIC:
      onStepCrystalFactor[0] =
          timeStep * langevinPistonOnStepVelocity[0] / boxDimensions[0];
      onStepCrystalFactor[1] = onStepCrystalFactor[0];
      onStepCrystalFactor[2] = onStepCrystalFactor[0];

      halfStepCrystalFactor[0] =
          langevinPistonDeltaPosition[0] / langevinPistonHalfStepPosition[0];
      halfStepCrystalFactor[1] = halfStepCrystalFactor[0];
      halfStepCrystalFactor[2] = halfStepCrystalFactor[0];

      boxDimensions[0] = langevinPistonOnStepPosition[0];
      boxDimensions[1] = langevinPistonOnStepPosition[0];
      boxDimensions[2] = langevinPistonOnStepPosition[0];
      break;

    case CRYSTAL::TETRAGONAL:
      onStepCrystalFactor[0] =
          timeStep * langevinPistonOnStepVelocity[0] / boxDimensions[0];
      onStepCrystalFactor[1] = onStepCrystalFactor[0];
      onStepCrystalFactor[2] =
          timeStep * langevinPistonOnStepVelocity[1] / boxDimensions[2];

      halfStepCrystalFactor[0] =
          langevinPistonDeltaPosition[0] / langevinPistonHalfStepPosition[0];
      halfStepCrystalFactor[1] = halfStepCrystalFactor[0];
      halfStepCrystalFactor[2] =
          langevinPistonDeltaPosition[1] / langevinPistonHalfStepPosition[1];

      boxDimensions[0] = langevinPistonOnStepPosition[0];
      boxDimensions[1] = langevinPistonOnStepPosition[0];
      boxDimensions[2] = langevinPistonOnStepPosition[1];
      break;

    case CRYSTAL::ORTHORHOMBIC:
      onStepCrystalFactor[0] =
          timeStep * langevinPistonOnStepVelocity[0] / boxDimensions[0];
      onStepCrystalFactor[1] =
          timeStep * langevinPistonOnStepVelocity[1] / boxDimensions[1];
      onStepCrystalFactor[2] =
          timeStep * langevinPistonOnStepVelocity[2] / boxDimensions[2];

      halfStepCrystalFactor[0] =
          langevinPistonDeltaPosition[0] / langevinPistonHalfStepPosition[0];
      halfStepCrystalFactor[1] =
          langevinPistonDeltaPosition[1] / langevinPistonHalfStepPosition[1];
      halfStepCrystalFactor[2] =
          langevinPistonDeltaPosition[2] / langevinPistonHalfStepPosition[2];

      boxDimensions[0] = langevinPistonOnStepPosition[0];
      boxDimensions[1] = langevinPistonOnStepPosition[1];
      boxDimensions[2] = langevinPistonOnStepPosition[2];
      break;

    default:
      break;
    }
  }

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

  double ke[2] = {0.0, 0.0};
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

    ke[0] += oneThird * (oldHalfStepKineticEnergy + onStepKineticEnergy +
                         newHalfStepKineticEnergy);
    ke[1] += onStepKineticEnergy;
  }

  ke[0] = BlockReduceSum<double>(ke[0]);
  ke[1] = BlockReduceSum<double>(ke[1]);

  if (threadIdx.x == 0) {
    atomicAdd(kineticEnergy + 0, ke[0]);
    atomicAdd(kineticEnergy + 1, ke[1]);
  }

  return;
}

__global__ static void UpdateNoseHooverPistonKernel(
    double *__restrict__ noseHooverPistonForce,
    double *__restrict__ noseHooverPistonForcePrevious,
    double *__restrict__ noseHooverPistonVelocity,
    const double *__restrict__ noseHooverPistonVelocityPrevious,
    const double *__restrict__ noseHooverPistonMass,
    const double *__restrict__ kineticEnergy,
    const double referenceKineticEnergy, const bool usingOldTemperature,
    const double timeStep) {
  if (threadIdx.x == 0) {
    const int i = (usingOldTemperature) ? 1 : 0;

    // Acutally store the change in velocity (not the force)
    *noseHooverPistonForce = 2.0 * timeStep *
                             (kineticEnergy[i] - referenceKineticEnergy) /
                             *noseHooverPistonMass;

    if (*noseHooverPistonForcePrevious == 0.0)
      *noseHooverPistonForcePrevious = *noseHooverPistonForce;

    *noseHooverPistonVelocity =
        *noseHooverPistonVelocityPrevious +
        0.5 * (*noseHooverPistonForce + *noseHooverPistonForcePrevious);
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
    const bool usingNoseHooverThermostat,
    const double *__restrict__ onStepCrystalFactor,
    const double *__restrict__ halfStepCrystalFactor, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double fact = 0.5 / timeStep;

  const double lpFactX = timeStep * onStepCrystalFactor[0];
  const double lpFactY = timeStep * onStepCrystalFactor[1];
  const double lpFactZ = timeStep * onStepCrystalFactor[2];
  double nhFact = 0.0;
  if (usingNoseHooverThermostat)
    nhFact = timeStep * timeStep * noseHooverPistonVelocity[0];

  for (int i = idx; i < numAtoms; i += stride) {
    const double onStepVelocityX =
        fact * (coordsDeltaPrevious[i].x + coordsDeltaPredicted[i].x);
    const double onStepVelocityY =
        fact * (coordsDeltaPrevious[i].y + coordsDeltaPredicted[i].y);
    const double onStepVelocityZ =
        fact * (coordsDeltaPrevious[i].z + coordsDeltaPredicted[i].z);

    // v(t+ dt/2)*dt = [v(t-dt/2)*dt + f(t)*dt^2/m] - h.(t)/h(t)*v(t)*dt^2
    coordsDeltaPredicted[i].x =
        coordsDelta[i].x - (nhFact + lpFactX) * onStepVelocityX;
    coordsDeltaPredicted[i].y =
        coordsDelta[i].y - (nhFact + lpFactY) * onStepVelocityY;
    coordsDeltaPredicted[i].z =
        coordsDelta[i].z - (nhFact + lpFactZ) * onStepVelocityZ;

    velMass[i].x = onStepVelocityX;
    velMass[i].y = onStepVelocityY;
    velMass[i].z = onStepVelocityZ;

    const double halfStepPositionX =
        0.5 * (coordsRef[i].x + coordsCharges[i].x);
    const double halfStepPositionY =
        0.5 * (coordsRef[i].y + coordsCharges[i].y);
    const double halfStepPositionZ =
        0.5 * (coordsRef[i].z + coordsCharges[i].z);

    const double scaledCrystalVelocityX =
        halfStepCrystalFactor[0] * halfStepPositionX;
    const double scaledCrystalVelocityY =
        halfStepCrystalFactor[1] * halfStepPositionY;
    const double scaledCrystalVelocityZ =
        halfStepCrystalFactor[2] * halfStepPositionZ;

    coordsCharges[i].x =
        coordsRef[i].x + coordsDeltaPredicted[i].x + scaledCrystalVelocityX;
    coordsCharges[i].y =
        coordsRef[i].y + coordsDeltaPredicted[i].y + scaledCrystalVelocityY;
    coordsCharges[i].z =
        coordsRef[i].z + coordsDeltaPredicted[i].z + scaledCrystalVelocityZ;
  }

  return;
}

__global__ static void UpdateDeltaPressureKernel(
    double *__restrict__ deltaPressureTensor,
    const double *__restrict__ staticDeltaPressureTensor,
    const double *__restrict__ deltaKineticPressureTensor) {
  if (threadIdx.x < 9) {
    deltaPressureTensor[threadIdx.x] = staticDeltaPressureTensor[threadIdx.x] +
                                       deltaKineticPressureTensor[threadIdx.x];
  }
  return;
}

__global__ static void ApplyBarostatToReferenceCoordsKernel(
    double4 *__restrict__ coordsRef, const double4 *__restrict__ coordsCharges,
    const int numAtoms, const double *__restrict__ halfStepCrystalFactor) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    const double halfStepPositionX =
        0.5 * (coordsRef[i].x + coordsCharges[i].x);
    const double halfStepPositionY =
        0.5 * (coordsRef[i].y + coordsCharges[i].y);
    const double halfStepPositionZ =
        0.5 * (coordsRef[i].z + coordsCharges[i].z);

    const double scaledCrystalVelocityX =
        halfStepCrystalFactor[0] * halfStepPositionX;
    const double scaledCrystalVelocityY =
        halfStepCrystalFactor[1] * halfStepPositionY;
    const double scaledCrystalVelocityZ =
        halfStepCrystalFactor[2] * halfStepPositionZ;

    coordsRef[i].x += scaledCrystalVelocityX;
    coordsRef[i].y += scaledCrystalVelocityY;
    coordsRef[i].z += scaledCrystalVelocityZ;
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
  if (threadIdx.x < 2) {
    const double s = static_cast<double>(step + 1);
    const double ndegf = static_cast<double>(numDegreesOfFreedom);
    const double temperature =
        kineticEnergy[threadIdx.x] / (0.5 * ndegf * kBoltz);
    const double delta0 = temperature - averageTemperature[threadIdx.x];
    averageTemperature[threadIdx.x] += delta0 / s;
    // const double delta1 = temperature - averageTemperature[threadIdx.x];
    // varianceTemperature[threadIdx.x] += delta0 * delta1;
  }
  return;
}

__global__ static void
UpdateAveragePressureKernel(double *__restrict__ pressureTensor,
                            double *__restrict__ pressureScalar,
                            double *__restrict__ averagePressureTensor,
                            double *__restrict__ averagePressureScalar,
                            const double *__restrict__ virialTensor,
                            const double *__restrict__ kineticPressureTensor,
                            const double volumeFactor, const int step) {
  const double s = static_cast<double>(step + 1);

  if (threadIdx.x < 9) {
    pressureTensor[threadIdx.x] =
        volumeFactor *
        (kineticPressureTensor[threadIdx.x] + virialTensor[threadIdx.x]);
    const double delta0 =
        pressureTensor[threadIdx.x] - averagePressureTensor[threadIdx.x];
    averagePressureTensor[threadIdx.x] += delta0 / s;
    // const double delta1 =
    //     pressureTensor[threadIdx.x] - averagePressureTensor[threadIdx.x];
    // variancePressureTensor[threadIdx.x] += delta0 * delta1;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    *pressureScalar =
        (pressureTensor[0] + pressureTensor[4] + pressureTensor[8]) / 3.0;
    const double delta0 = *pressureScalar - *averagePressureScalar;
    *averagePressureScalar += delta0 / s;
    // const double delta1 = *pressureScalar - *averagePressureScalar;
    // *variancePressureScalar += delta0 * delta1;
  }

  return;
}

void CudaLangevinPistonIntegrator::propagateOneStep(void) {
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

  CudaContainer<double> boxDimensions = m_Context->getBoxDimensions();
  const double volume = m_Context->getVolume();

  if (m_StepsSinceNeighborListUpdate % m_NonbondedListUpdateFrequency == 0) {
    if (m_Context->getForceManager()->getPeriodicBoundaryCondition() ==
        PBC::P21) {
      // Find a better place for this
      int numGroups =
          m_Context->getForceManager()->getPSF()->getGroups().size();
      int2 *groups =
          m_Context->getForceManager()->getPSF()->getGroups().getDeviceData();

      constexpr int numThreads = 256;
      const int numBlocks = (numGroups + numThreads - 1) / numThreads;
      InvertDeltaAsymmetricKernel<<<numBlocks, numThreads, 0,
                                    *m_IntegratorStream>>>(
          m_CoordsDeltaPrevious.getDeviceData(), xyzq, groups, numGroups,
          static_cast<float>(boxDimensions[0]));
      cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
    }
    m_Context->resetNeighborList();
  }

  if (m_CurrentPropagatedStep % m_RemoveCenterOfMassFrequency == 0)
    this->removeCenterOfMassMotion();

  copy_DtoD_async<double4>(coordsCharges, m_CoordsRef.getDeviceData(), numAtoms,
                           *m_IntegratorStream);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  m_Context->calculateForces(false, true, true);

  copy_DtoD_async<double>(m_NoseHooverPistonVelocity.getDeviceData(),
                          m_NoseHooverPistonVelocityPrevious.getDeviceData(), 1,
                          *m_IntegratorStream);

  copy_DtoD_async<double>(m_NoseHooverPistonForce.getDeviceData(),
                          m_NoseHooverPistonForcePrevious.getDeviceData(), 1,
                          *m_IntegratorStream);

  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

  HalfStepVelocityKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      coordsCharges, m_CoordsDelta.getDeviceData(),
      m_CoordsDeltaPrevious.getDeviceData(), velMass, numAtoms, forces,
      forceStride, m_TimeStep);

  // JEG260114: This needs to be called after calculateForces becuase getVirial
  // actually updates the virial in the forceManager
  double *virialTensor = m_Context->getVirial().getDeviceData();

  if (m_UsingHolonomicConstraints) {
    // JEG260107: Need sync here, otherwise race condition? I don't get it
    // either...
    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());

    ComputeHolonomicConstraintForcesKernel<<<numBlocks, numThreads, 0,
                                             *m_IntegratorStream>>>(
        m_HolonomicConstraintForces.getDeviceData(), coordsCharges,
        m_CoordsRef.getDeviceData(), m_CoordsDelta.getDeviceData(), velMass,
        numAtoms, m_TimeStep);

    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(m_HolonomicConstraintVirial.getDeviceData()), 0,
        9 * sizeof(double), *m_IntegratorStream));

    ComputeHolonomicConstraintVirialKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
        m_HolonomicConstraintVirial.getDeviceData(),
        m_CoordsRef.getDeviceData(),
        m_HolonomicConstraintForces.getDeviceData(), numAtoms);
    // ComputeHolonomicConstraintVirialKernel<<<numBlocks, numThreads, 0,
    //                                          *m_IntegratorStream>>>(
    //     m_HolonomicConstraintVirial.getDeviceData(),
    //     m_CoordsRef.getDeviceData(),
    //     m_HolonomicConstraintForces.getDeviceData(), numAtoms);

    UpdateCoordsDeltaAfterHolonomicConstraintKernel<<<numBlocks, numThreads, 0,
                                                      *m_IntegratorStream>>>(
        m_CoordsDelta.getDeviceData(), coordsCharges,
        m_CoordsRef.getDeviceData(), numAtoms);

    UpdateVirialWithHolonomicConstraintVirialKernel<<<1, 32, 0,
                                                      *m_IntegratorStream>>>(
        virialTensor, m_HolonomicConstraintVirial.getDeviceData());
  }

  cudaCheck(cudaMemsetAsync(
      static_cast<void *>(m_KineticPressureTensor.getDeviceData()), 0,
      9 * sizeof(double), *m_IntegratorStream));

  ComputeAverageKineticPressureKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
      m_KineticPressureTensor.getDeviceData(), velMass,
      m_CoordsDelta.getDeviceData(), m_CoordsDeltaPrevious.getDeviceData(),
      numAtoms, m_TimeStep);
  // ComputeAverageKineticPressureKernel<<<numBlocks, numThreads, 0,
  //                                       *m_IntegratorStream>>>(
  //     m_KineticPressureTensor.getDeviceData(), velMass,
  //     m_CoordsDelta.getDeviceData(), m_CoordsDeltaPrevious.getDeviceData(),
  //     numAtoms, m_TimeStep);

  UpdatePressureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
      m_PressureTensor.getDeviceData(), m_PressureScalar.getDeviceData(),
      m_ReferencePressureTensor.getDeviceData(),
      m_DeltaPressureTensor.getDeviceData(), virialTensor,
      m_KineticPressureTensor.getDeviceData(),
      charmm::constants::patmos / volume, m_SurfaceTension.getDeviceData(),
      charmm::constants::surfaceTensionFactor / boxDimensions[2],
      m_ConstantSurfaceTensionFlag);

  cudaCheck(cudaMemsetAsync(
      static_cast<void *>(m_DeltaKineticPressureTensor.getDeviceData()), 0,
      9 * sizeof(double), *m_IntegratorStream));

  ComputeDeltaKineticPressureKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
      m_DeltaKineticPressureTensor.getDeviceData(), velMass,
      m_CoordsDelta.getDeviceData(), numAtoms,
      0.5 * charmm::constants::patmos / volume, m_TimeStep);
  // ComputeDeltaKineticPressureKernel<<<numBlocks, numThreads, 0,
  //                                     *m_IntegratorStream>>>(
  //     m_DeltaKineticPressureTensor.getDeviceData(), velMass,
  //     m_CoordsDelta.getDeviceData(), numAtoms,
  //     0.5 * charmm::constants::patmos / volume, m_TimeStep);

  ComputeStaticDeltaPressureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
      m_StaticDeltaPressureTensor.getDeviceData(),
      m_DeltaPressureTensor.getDeviceData(),
      m_DeltaKineticPressureTensor.getDeviceData());

  copy_DtoD_async<double4>(m_CoordsDelta.getDeviceData(),
                           m_CoordsDeltaPredicted.getDeviceData(), numAtoms,
                           *m_IntegratorStream);

  CudaContainer<double> boxDimensionsPredicted = boxDimensions;

  for (int iter = 0; iter < m_MaxPredictorCorrectorIterations; iter++) {
    copy_DtoD_async<double>(boxDimensions.getDeviceData(),
                            boxDimensionsPredicted.getDeviceData(), 3,
                            *m_IntegratorStream);

    copy_DtoD_async<double>(
        m_LangevinPistonDeltaPosition.getDeviceData(),
        m_LangevinPistonDeltaPositionPredicted.getDeviceData(),
        m_LangevinPistonDegreesOfFreedom, *m_IntegratorStream);

    UpdateLangevinPistonKernel<<<1, 32, 0, *m_IntegratorStream>>>(
        m_LangevinPistonOnStepPosition.getDeviceData(),
        m_LangevinPistonHalfStepPosition.getDeviceData(),
        m_LangevinPistonDeltaPositionPredicted.getDeviceData(),
        m_LangevinPistonDeltaPositionPrevious.getDeviceData(),
        m_LangevinPistonOnStepVelocity.getDeviceData(),
        m_LangevinPistonDeltaPressure.getDeviceData(),
        m_OnStepCrystalFactor.getDeviceData(),
        m_HalfStepCrystalFactor.getDeviceData(),
        boxDimensionsPredicted.getDeviceData(), m_RngStates,
        m_LangevinPistonInverseMass.getDeviceData(),
        m_DeltaPressureTensor.getDeviceData(), m_Prfwd.getDeviceData(),
        m_Palpha, m_Pbfact, volume * m_Pbfact * charmm::constants::atmosp,
        m_LangevinPistonDegreesOfFreedom, m_CrystalType, m_TimeStep);

    m_RngSequencePos++; // curand_normal_double iterates 1 step

    cudaCheck(
        cudaMemsetAsync(static_cast<void *>(m_KineticEnergy.getDeviceData()), 0,
                        2 * sizeof(double), *m_IntegratorStream));

    ComputeKineticEnergyKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
        m_KineticEnergy.getDeviceData(), velMass,
        m_CoordsDeltaPredicted.getDeviceData(),
        m_CoordsDeltaPrevious.getDeviceData(), numAtoms, m_TimeStep);

    UpdateNoseHooverPistonKernel<<<1, 32, 0, *m_IntegratorStream>>>(
        m_NoseHooverPistonForce.getDeviceData(),
        m_NoseHooverPistonForcePrevious.getDeviceData(),
        m_NoseHooverPistonVelocity.getDeviceData(),
        m_NoseHooverPistonVelocityPrevious.getDeviceData(),
        m_NoseHooverPistonMass.getDeviceData(), m_KineticEnergy.getDeviceData(),
        referenceKineticEnergy, m_UsingOldTemperature, m_TimeStep);

    PredictorCorrectorKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
        coordsCharges, velMass, m_CoordsDeltaPredicted.getDeviceData(),
        m_CoordsDelta.getDeviceData(), m_CoordsDeltaPrevious.getDeviceData(),
        m_CoordsRef.getDeviceData(), numAtoms,
        m_NoseHooverPistonVelocity.getDeviceData(), m_UsingNoseHooverThermostat,
        m_OnStepCrystalFactor.getDeviceData(),
        m_HalfStepCrystalFactor.getDeviceData(), m_TimeStep);

    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(m_DeltaKineticPressureTensor.getDeviceData()), 0,
        9 * sizeof(double), *m_IntegratorStream));

    ComputeDeltaKineticPressureKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
        m_DeltaKineticPressureTensor.getDeviceData(), velMass,
        m_CoordsDeltaPredicted.getDeviceData(), numAtoms,
        0.5 * charmm::constants::patmos / volume, m_TimeStep);
    // ComputeDeltaKineticPressureKernel<<<numBlocks, numThreads, 0,
    //                                     *m_IntegratorStream>>>(
    //     m_DeltaKineticPressureTensor.getDeviceData(), velMass,
    //     m_CoordsDeltaPredicted.getDeviceData(), numAtoms,
    //     0.5 * charmm::constants::patmos / volume, m_TimeStep);

    UpdateDeltaPressureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
        m_DeltaPressureTensor.getDeviceData(),
        m_StaticDeltaPressureTensor.getDeviceData(),
        m_DeltaKineticPressureTensor.getDeviceData());
  }

  copy_DtoD_async<double>(boxDimensionsPredicted.getDeviceData(),
                          boxDimensions.getDeviceData(), 3,
                          *m_IntegratorStream);

  copy_DtoD_async<double>(
      m_LangevinPistonDeltaPositionPredicted.getDeviceData(),
      m_LangevinPistonDeltaPosition.getDeviceData(),
      m_LangevinPistonDegreesOfFreedom, *m_IntegratorStream);

  if (m_UsingHolonomicConstraints) {
    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());

    ApplyBarostatToReferenceCoordsKernel<<<numBlocks, numThreads, 0,
                                           *m_IntegratorStream>>>(
        m_CoordsRef.getDeviceData(), coordsCharges, numAtoms,
        m_HalfStepCrystalFactor.getDeviceData());

    double4 *tmp = m_CoordsRef.getDeviceData();
    m_Context->getCoordinatesCharges().getDeviceArray().assignData(tmp);

    m_HolonomicConstraint->handleHolonomicConstraints(coordsCharges);

    m_CoordsRef.getDeviceArray().assignData(tmp);
    m_Context->getCoordinatesCharges().getDeviceArray().assignData(
        coordsCharges);

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
                      2 * sizeof(double), *m_IntegratorStream));

  ComputeKineticEnergyKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
      m_KineticEnergy.getDeviceData(), velMass, m_CoordsDelta.getDeviceData(),
      m_CoordsDeltaPrevious.getDeviceData(), numAtoms, m_TimeStep);

  UpdateNoseHooverPistonKernel<<<1, 32, 0, *m_IntegratorStream>>>(
      m_NoseHooverPistonForce.getDeviceData(),
      m_NoseHooverPistonForcePrevious.getDeviceData(),
      m_NoseHooverPistonVelocity.getDeviceData(),
      m_NoseHooverPistonVelocityPrevious.getDeviceData(),
      m_NoseHooverPistonMass.getDeviceData(), m_KineticEnergy.getDeviceData(),
      referenceKineticEnergy, m_UsingOldTemperature, m_TimeStep);

  UpdateAverageTemperatureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
      m_AverageTemperature.getDeviceData(), m_KineticEnergy.getDeviceData(),
      numDegreesOfFreedom, charmm::constants::kBoltz, m_AverageWindowSize);

  cudaCheck(cudaMemsetAsync(
      static_cast<void *>(m_KineticPressureTensor.getDeviceData()), 0,
      9 * sizeof(double), *m_IntegratorStream));

  ComputeAverageKineticPressureKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
      m_KineticPressureTensor.getDeviceData(), velMass,
      m_CoordsDelta.getDeviceData(), m_CoordsDeltaPrevious.getDeviceData(),
      numAtoms, m_TimeStep);
  // ComputeAverageKineticPressureKernel<<<numBlocks, numThreads, 0,
  //                                       *m_IntegratorStream>>>(
  //     m_KineticPressureTensor.getDeviceData(), velMass,
  //     m_CoordsDelta.getDeviceData(), m_CoordsDeltaPrevious.getDeviceData(),
  //     numAtoms, m_TimeStep);

  UpdateAveragePressureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
      m_PressureTensor.getDeviceData(), m_PressureScalar.getDeviceData(),
      m_AveragePressureTensor.getDeviceData(),
      m_AveragePressureScalar.getDeviceData(), virialTensor,
      m_KineticPressureTensor.getDeviceData(),
      charmm::constants::patmos / volume, m_AverageWindowSize);

  m_AverageWindowSize++;

  UpdateSinglePrecisionCoordinatesKernel<<<numBlocks, numThreads, 0,
                                           *m_IntegratorStream>>>(
      xyzq, coordsCharges, numAtoms);

  copy_DtoD_async<double4>(m_CoordsDelta.getDeviceData(),
                           m_CoordsDeltaPrevious.getDeviceData(), numAtoms,
                           *m_IntegratorStream);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  boxDimensions.transferToHost();
  m_Context->setBoxDimensions(boxDimensions.getHostArray());

  // DO HFCTE CALCULATION HERE ONLY IN DEBUG MODE

  return;
}

double CudaLangevinPistonIntegrator::computeNoseHooverPistonMass(void) {
  if (m_Context == nullptr)
    return -9999.9999;

  CudaContainer<double4> velMass = m_Context->getVelocityMass();
  velMass.transferToHost();

  double totalMass = 0.0;
  for (std::size_t i = 0; i < velMass.size(); i++)
    totalMass += (1.0 / velMass[i].w);

  return (0.2 * totalMass);
}

double CudaLangevinPistonIntegrator::computeLangevinPistonMass(void) {
  if (m_Context == nullptr)
    return -9999.9999;

  CudaContainer<double4> velMass = m_Context->getVelocityMass();
  velMass.transferToHost();

  double totalMass = 0.0;
  for (std::size_t i = 0; i < velMass.size(); i++)
    totalMass += (1.0 / velMass[i].w);

  return (0.02 * totalMass);
}

void CudaLangevinPistonIntegrator::allocateLangevinPistonVariables(void) {
  // Allocate memory for Langevin piston variables
  m_LangevinPistonMass.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonInverseMass.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonOnStepPosition.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonHalfStepPosition.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonOnStepVelocity.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonHalfStepVelocity.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonDeltaPosition.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonDeltaPositionPrevious.resize(
      m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonDeltaPositionPredicted.resize(
      m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonDeltaPressure.resize(m_LangevinPistonDegreesOfFreedom);
  m_Prfwd.resize(m_LangevinPistonDegreesOfFreedom);

  // Set pressure piston variables to default values
  m_LangevinPistonMass.setToValue(-9999.9999);
  m_LangevinPistonInverseMass.setToValue(-9999.9999);
  m_LangevinPistonOnStepPosition.setToValue(0.0);
  m_LangevinPistonHalfStepPosition.setToValue(0.0);
  m_LangevinPistonOnStepVelocity.setToValue(0.0);
  m_LangevinPistonHalfStepVelocity.setToValue(0.0);
  m_LangevinPistonDeltaPosition.setToValue(0.0);
  m_LangevinPistonDeltaPositionPrevious.setToValue(0.0);
  m_LangevinPistonDeltaPositionPredicted.setToValue(0.0);
  m_LangevinPistonDeltaPressure.setToValue(0.0);
  m_Prfwd.setToValue(0.0);

  return;
}

__global__ static void
InitializeRngKernel(curandStatePhilox4_32_10_t *__restrict__ states,
                    const int n, const unsigned long long int seed,
                    const unsigned long long int offset,
                    const unsigned long long int nskip) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < n; i += stride) {
    curand_init(seed, i, offset, states + i);
    skipahead(nskip, states + i);
  }

  return;
}

void CudaLangevinPistonIntegrator::initializeRng(void) {
  if (m_LangevinPistonDegreesOfFreedom == -1)
    return;

  this->alloc(m_LangevinPistonDegreesOfFreedom);
  constexpr int numThreads = 256;
  const int numBlocks =
      (m_LangevinPistonDegreesOfFreedom + numThreads - 1) / numThreads;
  InitializeRngKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      m_RngStates, m_LangevinPistonDegreesOfFreedom, m_Seed, 0,
      m_RngSequencePos);

  return;
}

void CudaLangevinPistonIntegrator::removeCenterOfMassMotion(void) {
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

void CudaLangevinPistonIntegrator::alloc(const int n) {
  if (m_RngStates != nullptr) // Deallocate to be safe
    this->dealloc();
  // Allocate memory for RNG
  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&m_RngStates),
                       n * sizeof(curandStatePhilox4_32_10_t)));
  return;
}

void CudaLangevinPistonIntegrator::dealloc(void) {
  if (m_RngStates != nullptr) {
    // Deallocate memory for RNG
    cudaCheck(cudaFree(static_cast<void *>(m_RngStates)));
    m_RngStates = nullptr;
  }
  return;
}
