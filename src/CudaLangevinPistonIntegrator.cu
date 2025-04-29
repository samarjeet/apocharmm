// BEGINLICENSE
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

// Hello this works

#include "Constants.h"
#include "CudaContainer.h"
#include "CudaLangevinPistonIntegrator.h"
#include "gpu_utils.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>

CudaLangevinPistonIntegrator::CudaLangevinPistonIntegrator(
    const double timeStep)
    : CudaLangevinPistonIntegrator(timeStep, CRYSTAL::ORTHORHOMBIC) {}

CudaLangevinPistonIntegrator::CudaLangevinPistonIntegrator(
    const double timeStep, const CRYSTAL crystalType)
    : CudaIntegrator(timeStep) {
  m_UsingHolonomicConstraints = true;
  m_MaxPredictorCorrectorSteps = 3;

  m_DevPHILOXStates = nullptr;

  m_StepsSinceLastReport = 0;
  m_BathTemperature = 300.0;

  m_NoseHooverFlag = true;
  m_NoseHooverPistonMass = 0.0;
  m_NoseHooverPistonPosition = 0.0;
  m_NoseHooverPistonVelocity = 0.0;
  m_NoseHooverPistonVelocityPrevious = 0.0;
  m_NoseHooverPistonForce = 0.0;
  m_NoseHooverPistonForcePrevious = 0.0;

  m_ConstantSurfaceTensionFlag = false;

  m_KineticEnergyPressureTensor.resize(6);
  m_PressureTensor.resize(6);
  if (m_UsingHolonomicConstraints) {
    m_HolonomicVirial.resize(6);
    m_HolonomicVirial.setToValue(0.0);
  }

  m_CrystalDimensions.resize(6);
  m_CrystalDimensionsPrevious.resize(6);
  m_InverseCrystalDimensions.resize(6);

  m_ReferencePressure.resize(6);
  m_ReferencePressure.set({1.0, 0.0, 1.0, 0.0, 0.0, 1.0});

  m_DeltaPressure.resize(6);
  m_DeltaPressureNonChanging.resize(6);
  m_DeltaPressureHalfStepKinetic.resize(6);

  m_Pgamma = 0.0;
  // m_Pbfact = m_TimeStep * m_TimeStep;
  m_Pvfact = 1.0 / m_TimeStep;

  m_Palpha = 1.0;

  std::random_device rd{};
  m_Seed = rd();
  m_Rng.seed(m_Seed);

  m_StepId = 0;

  switch (crystalType) {
  case CRYSTAL::ORTHORHOMBIC:
    m_PistonDegreesOfFreedom = 3;
    break;
  case CRYSTAL::TETRAGONAL:
    m_PistonDegreesOfFreedom = 2;
    break;
  case CRYSTAL::CUBIC:
    m_PistonDegreesOfFreedom = 1;
    break;
  default:
    throw std::invalid_argument(
        "Invalid crystal type. Please use CRYSTAL::ORTHORHOMBIC, "
        "CRYSTAL::TETRAGONAL or CRYSTAL::CUBIC.");
  }

  this->allocatePistonVariables();
  for (int i = 0; i < m_PistonDegreesOfFreedom; i++) {
    m_PistonMass[i] = 500.0;
    m_InversePistonMass[i] = 1.0 / m_PistonMass[i];
  }
  m_PistonMass.transferToDevice();
  m_InversePistonMass.transferToDevice();

  m_OnStepCrystalFactor.resize(m_CrystalDegreesOfFreedom);
  m_HalfStepCrystalFactor.resize(m_CrystalDegreesOfFreedom);

  m_PressureScalar.resize(1);

  // Allocations for HFCTEN calculation
  m_HalfStepKineticEnergy.resize(1);
  m_HalfStepKineticEnergy1StepPrevious.resize(1);
  m_HalfStepKineticEnergy1StepPrevious.setToValue(0.0);
  m_HalfStepKineticEnergy2StepsPrevious.resize(1);
  m_PotentialEnergyPrevious.resize(1);
  m_PotentialEnergyPrevious.setToValue(0.0);
  m_HfctenTerm.resize(1);

  m_AveragePressureScalar.resize(1);
  m_AveragePressureScalar.setToValue(0.0);
  m_AveragePressureTensor.resize(6);
  m_AveragePressureTensor.setToValue(0.0);

  m_PistonFrictionSetFlag = false;
}

CudaLangevinPistonIntegrator::~CudaLangevinPistonIntegrator() {
  // cudaCheck(cudaFree(devPHILOXStates));
}

void CudaLangevinPistonIntegrator::setPressure(
    const std::vector<double> &referencePressure) {
  m_ReferencePressure = referencePressure;
  return;
}

// void CudaLangevinPistonIntegrator::setPistonMass(double _pistonMass) {
//
//   std::cout << "Setting pistonMass using a double value."
//             << "\n Deprecated ! Please use [float] for a cubic system, "
//                "[float,float] for a tetragonal system, [float,float,float] "
//                "for an orthorombic system."
//             << std::endl;
//
//   for (int i = 0; i < pistonDegreesOfFreedom; i++) {
//     pistonMass[i] = _pistonMass;
//     inversePistonMass[i] = 1.0 / _pistonMass;
//   }
//   inversePistonMass.transferToDevice();
// }

void CudaLangevinPistonIntegrator::setPistonMass(
    const std::vector<double> &pistonMass) {
  assert(
      pistonMass.size() == m_PistonDegreesOfFreedom &&
      "size of pistonMass vector and pistonDegreesOfFreedom should be equal.");

  for (int i = 0; i < m_PistonDegreesOfFreedom; i++) {
    if (pistonMass[i] == 0.0) {
      // pistonMass[i] = std::numeric_limits<double>::max();
      m_InversePistonMass[i] = 0.0;
      m_PistonMass[i] = 0.0;
    } else {
      m_InversePistonMass[i] = 1.0 / pistonMass[i];
      m_PistonMass[i] = pistonMass[i];
    }
  }
  m_PistonMass.transferToDevice();
  m_InversePistonMass.transferToDevice();

  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonMass(
    const double nhMass) {
  m_NoseHooverPistonMass = nhMass;
  return;
}

void CudaLangevinPistonIntegrator::setCrystalType(const CRYSTAL crystalType) {
  m_CrystalType = crystalType;

  switch (m_CrystalType) {
  case CRYSTAL::ORTHORHOMBIC:
    m_PistonDegreesOfFreedom = 3;
    break;
  case CRYSTAL::TETRAGONAL:
    m_PistonDegreesOfFreedom = 2;
    break;
  case CRYSTAL::CUBIC:
    m_PistonDegreesOfFreedom = 1;
    break;
  default:
    break;
  }

  this->allocatePistonVariables();

  return;
}

void CudaLangevinPistonIntegrator::setSurfaceTension(const double st) {
  m_ConstantSurfaceTensionFlag = true;

  // Since we only have an orthorhombic box, only Z perpendicular to X-Y are
  // apt.
  m_SurfaceTension = 2 * st;

  return;
}

// void CudaLangevinPistonIntegrator::setBoxDimensions(
//     const std::vector<double> &boxDimensions) {
//   m_BoxDimensions.resize(m_CrystalDegreesOfFreedom);
//   m_BoxDimensions.setToValue(0.0);
//   for (int i = 0; i < m_CrystalDegreesOfFreedom; i++)
//     m_BoxDimensions[i] = boxDimensions[i];
//   m_BoxDimensions.transferToDevice();
//   return;
// }

void CudaLangevinPistonIntegrator::setPistonFriction(const double _friction) {
  m_Pgamma = _friction;
  double pgam = m_Timfac * m_TimeStep * m_Pgamma;
  m_Palpha = (1 - pgam * 0.5) / (1 + pgam * 0.5);
  m_Pbfact = m_TimeStep * m_TimeStep / (1 + pgam * 0.5);
  m_Pvfact = 0.5 / m_TimeStep;

  double kbt = charmm::constants::kBoltz * m_BathTemperature;
  assert(m_PistonDegreesOfFreedom != 0);
  m_Prfwd.resize(m_PistonDegreesOfFreedom);
  m_Prfwd.setToValue(0.0);
  for (int i = 0; i < m_PistonDegreesOfFreedom; i++) {
    m_Prfwd[i] =
        std::sqrt(2 * m_InversePistonMass[i] * pgam * kbt) / m_TimeStep;
  }
  m_Prfwd.transferToDevice();

  m_PistonFrictionSetFlag = true;

  return;
}

void CudaLangevinPistonIntegrator::setBathTemperature(
    const double bathTemperature) {
  m_BathTemperature = bathTemperature;
  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverFlag(
    const bool noseHooverFlag) {
  m_NoseHooverFlag = noseHooverFlag;
  return;
}

void CudaLangevinPistonIntegrator::setOnStepPistonVelocity(
    const CudaContainer<double> &onStepPistonVelocity) {
  m_OnStepPistonVelocity = onStepPistonVelocity;
  return;
}

void CudaLangevinPistonIntegrator::setOnStepPistonVelocity(
    const std::vector<double> &onStepPistonVelocity) {
  m_OnStepPistonVelocity = onStepPistonVelocity;
  return;
}

void CudaLangevinPistonIntegrator::setHalfStepPistonVelocity(
    const CudaContainer<double> &halfStepPistonVelocity) {
  m_HalfStepPistonVelocity = halfStepPistonVelocity;
  return;
}

void CudaLangevinPistonIntegrator::setHalfStepPistonVelocity(
    const std::vector<double> &halfStepPistonVelocity) {
  m_HalfStepPistonVelocity = halfStepPistonVelocity;
  return;
}

void CudaLangevinPistonIntegrator::setOnStepPistonPosition(
    const CudaContainer<double> &onStepPistonPosition) {
  m_OnStepPistonPosition = onStepPistonPosition;
  return;
}

void CudaLangevinPistonIntegrator::setOnStepPistonPosition(
    const std::vector<double> &onStepPistonPosition) {
  m_OnStepPistonPosition = onStepPistonPosition;
  return;
}

void CudaLangevinPistonIntegrator::setHalfStepPistonPosition(
    const CudaContainer<double> &halfStepPistonPosition) {
  m_HalfStepPistonPosition = halfStepPistonPosition;
  return;
}

void CudaLangevinPistonIntegrator::setHalfStepPistonPosition(
    const std::vector<double> &halfStepPistonPosition) {
  m_HalfStepPistonPosition = halfStepPistonPosition;
  return;
}

void CudaLangevinPistonIntegrator::setCoordsDeltaPrevious(
    const std::vector<std::vector<double>> &coordsDelta) {
  assert((coordsDelta.size() == m_Context->getNumAtoms(),
          "Wrong size in setCoordsDeltaPrevious"));
  std::vector<double4> cdp;
  for (std::size_t i = 0; i < coordsDelta.size(); i++) {
    cdp.emplace_back(make_double4(coordsDelta[i][0], coordsDelta[i][1],
                                  coordsDelta[i][2], 0.0));
  }
  m_CoordsDeltaPrevious = cdp;

  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonVelocity(
    const double noseHooverPistonVelocity) {
  m_NoseHooverPistonVelocity = noseHooverPistonVelocity;
  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonVelocityPrevious(
    const double noseHooverPistonVelocityPrevious) {
  m_NoseHooverPistonVelocityPrevious = noseHooverPistonVelocityPrevious;
  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonForce(
    const double noseHooverPistonForce) {
  m_NoseHooverPistonForce = noseHooverPistonForce;
  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonForcePrevious(
    const double noseHooverPistonForcePrevious) {
  m_NoseHooverPistonForcePrevious = noseHooverPistonForcePrevious;
  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonPosition(
    const double noseHooverPistonPosition) {
  m_NoseHooverPistonPosition = noseHooverPistonPosition;
  return;
}

void CudaLangevinPistonIntegrator::setMaxPredictorCorrectorSteps(
    const int maxPredictorCorrectorSteps) {
  m_MaxPredictorCorrectorSteps = maxPredictorCorrectorSteps;
  return;
}

void CudaLangevinPistonIntegrator::setSeedForPistonFriction(
    const uint64_t seed) {
  m_Seed = seed;
  m_Rng.seed(seed);
  return;
}

double CudaLangevinPistonIntegrator::getPressureScalar(void) const {
  // m_AveragePressureScalar.transferToHost();
  return m_AveragePressureScalar[0];
}

const std::vector<double> &
CudaLangevinPistonIntegrator::getPressureTensor(void) const {
  // m_AveragePressureTensor.transferToHost();
  return m_AveragePressureTensor.getHostArray();
}

std::vector<double> &CudaLangevinPistonIntegrator::getPressureTensor(void) {
  // m_AveragePressureTensor.transferToHost();
  return m_AveragePressureTensor.getHostArray();
}

double
CudaLangevinPistonIntegrator::getInstantaneousPressureScalar(void) const {
  // m_PressureScalar.transferToHost();
  return m_PressureScalar[0];
}

const std::vector<double> &
CudaLangevinPistonIntegrator::getInstantaneousPressureTensor(void) const {
  // m_PressureTensor.transferToHost();
  return m_PressureTensor.getHostArray();
}

std::vector<double> &
CudaLangevinPistonIntegrator::getInstantaneousPressureTensor(void) {
  // m_PressureTensor.transferToHost();
  return m_PressureTensor.getHostArray();
}

double CudaLangevinPistonIntegrator::getPistonMass(void) const {
  // m_PistonMass.transferToHost();
  return m_PistonMass[0];
}

CRYSTAL CudaLangevinPistonIntegrator::getCrystalType(void) const {
  return m_CrystalType;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getReferencePressure(void) const {
  return m_ReferencePressure;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getReferencePressure(void) {
  return m_ReferencePressure;
}

int CudaLangevinPistonIntegrator::getPistonDegreesOfFreedom(void) const {
  return m_PistonDegreesOfFreedom;
}

double CudaLangevinPistonIntegrator::getBathTemperature(void) const {
  return m_BathTemperature;
}

double CudaLangevinPistonIntegrator::getNoseHooverPistonMass(void) const {
  return m_NoseHooverPistonMass;
}

double CudaLangevinPistonIntegrator::getNoseHooverPistonPosition(void) const {
  return m_NoseHooverPistonPosition;
}

double CudaLangevinPistonIntegrator::getNoseHooverPistonVelocity(void) const {
  return m_NoseHooverPistonVelocity;
}

double
CudaLangevinPistonIntegrator::getNoseHooverPistonVelocityPrevious(void) const {
  return m_NoseHooverPistonVelocityPrevious;
}

double CudaLangevinPistonIntegrator::getNoseHooverPistonForce(void) const {
  return m_NoseHooverPistonForce;
}

double
CudaLangevinPistonIntegrator::getNoseHooverPistonForcePrevious(void) const {
  return m_NoseHooverPistonForcePrevious;
}

const CudaContainer<double4> &
CudaLangevinPistonIntegrator::getCoordsDelta(void) const {
  return m_CoordsDelta;
}

CudaContainer<double4> &CudaLangevinPistonIntegrator::getCoordsDelta(void) {
  return m_CoordsDelta;
}

const CudaContainer<double4> &
CudaLangevinPistonIntegrator::getCoordsDeltaPrevious(void) const {
  return m_CoordsDeltaPrevious;
}

CudaContainer<double4> &
CudaLangevinPistonIntegrator::getCoordsDeltaPrevious(void) {
  return m_CoordsDeltaPrevious;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getOnStepPistonVelocity(void) const {
  return m_OnStepPistonVelocity;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getOnStepPistonVelocity(void) {
  return m_OnStepPistonVelocity;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getHalfStepPistonVelocity(void) const {
  return m_HalfStepPistonVelocity;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getHalfStepPistonVelocity(void) {
  return m_HalfStepPistonVelocity;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getOnStepPistonPosition(void) const {
  return m_OnStepPistonPosition;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getOnStepPistonPosition(void) {
  return m_OnStepPistonPosition;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getHalfStepPistonPosition(void) const {
  return m_HalfStepPistonPosition;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getHalfStepPistonPosition(void) {
  return m_HalfStepPistonPosition;
}

bool CudaLangevinPistonIntegrator::hasPistonFrictionSet(void) const {
  return m_PistonFrictionSetFlag;
}

uint64_t CudaLangevinPistonIntegrator::getSeedForPistonFriction(void) const {
  return m_Seed;
}

/** @brief Updates the single-precision coordinates container (xyzq)
 */
__global__ static void
updateSPKernel(int numAtoms, float4 *__restrict__ xyzq,
               const double4 *__restrict__ coordsCharge) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    xyzq[index].x = (float)coordsCharge[index].x;
    xyzq[index].y = (float)coordsCharge[index].y;
    xyzq[index].z = (float)coordsCharge[index].z;
  }
}

/*
__global__ static void setup_kernel(int numAtoms,
                                    curandStatePhilox4_32_10_t *state) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // if (index < numAtoms)
  //   curand_init(1234, index, 0, &state[index]);
}
*/
__global__ static void init(double kbt, const int numAtoms, const int stride,
                            const double timeStep,
                            // double4 *__restrict__ coords,
                            double4 *__restrict__ coordsDelta,
                            double4 *__restrict__ coordsDeltaPrevious,
                            const double4 *__restrict__ velMass,
                            const double *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    double fx = -force[index];
    double fy = -force[index + stride];
    double fz = -force[index + 2 * stride];

    double fact = timeStep * timeStep * velMass[index].w * 0.5;

    coordsDeltaPrevious[index].x = velMass[index].x * timeStep - fx * fact;
    coordsDeltaPrevious[index].y = velMass[index].y * timeStep - fy * fact;
    coordsDeltaPrevious[index].z = velMass[index].z * timeStep - fz * fact;

    coordsDelta[index].x = velMass[index].x * timeStep + fx * fact;
    coordsDelta[index].y = velMass[index].y * timeStep + fy * fact;
    coordsDelta[index].z = velMass[index].z * timeStep + fz * fact;
  }
}

__global__ static void
backStepInitializationKernel(int numAtoms, double4 *__restrict__ coords,
                             double4 *__restrict__ coordsDeltaPrevious) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    coords[index].x -= coordsDeltaPrevious[index].x;
    coords[index].y -= coordsDeltaPrevious[index].y;
    coords[index].z -= coordsDeltaPrevious[index].z;
  }
}

__global__ static void
backStepInitializationKernel2(int numAtoms, double4 *__restrict__ coords,
                              double4 *__restrict__ coordsRef,
                              double4 *__restrict__ coordsDeltaPrevious) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    coordsDeltaPrevious[index].x = coordsRef[index].x - coords[index].x;
    coordsDeltaPrevious[index].y = coordsRef[index].y - coords[index].y;
    coordsDeltaPrevious[index].z = coordsRef[index].z - coords[index].z;

    coords[index].x = coordsRef[index].x;
    coords[index].y = coordsRef[index].y;
    coords[index].z = coordsRef[index].z;
  }
}

void CudaLangevinPistonIntegrator::initialize(void) {
  int numAtoms = m_Context->getNumAtoms();

  if (not this->hasPistonFrictionSet()) {
    throw std::invalid_argument(
        "Piston friction not set. Please set piston friction before "
        "using the Langevin piston integrator.");
  }

  //  Get the mass of the system and divide that by 50. (charmm-gui does that)
  m_NoseHooverPistonMass = this->computeNoseHooverPistonMass();

  // Reset Nose-Hoover piston variables (if integrator is reused from earlier,
  // e.g.)
  m_NoseHooverPistonForce = 0.0;
  m_NoseHooverPistonForcePrevious = 0.0;
  m_NoseHooverPistonVelocity = 0.0;
  m_NoseHooverPistonVelocityPrevious = 0.0;
  m_OnStepPistonVelocity.setToValue(0.0);
  m_HalfStepPistonVelocity.setToValue(0.0);
  m_OnStepPistonPosition.setToValue(0.0);
  m_HalfStepPistonPosition.setToValue(0.0);
  m_PistonDeltaPressure.setToValue(0.0);
  m_PressurePistonPositionDelta.setToValue(0.0);
  m_PressurePistonPositionDelta.setToValue(0.0);
  m_PressurePistonPositionDeltaPrevious.setToValue(0.0);
  m_PressurePistonPositionDeltaStored.setToValue(0.0);
  m_DeltaPressure.setToValue(0.0);

  m_CoordsDelta.resize(numAtoms);
  m_CoordsDeltaPrevious.resize(numAtoms);
  m_CoordsDeltaPredicted.resize(numAtoms);

  auto coordsRefDevice = m_CoordsRef.getDeviceData();
  if (m_UsingHolonomicConstraints)
    m_HolonomicConstraintForces.resize(numAtoms);

  auto boxDimensions = m_Context->getBoxDimensions();
  // setBoxDimensions(boxDimensionsOriginal);

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  auto coords = m_Context->getCoordinatesCharges().getDeviceData();

  auto xyzq = m_Context->getXYZQ()->getDeviceXYZQ();

  auto coordsDeltaDevice = m_CoordsDelta.getDeviceData();
  auto coordsDeltaPreviousDevice = m_CoordsDeltaPrevious.getDeviceData();

  auto velMass = m_Context->getVelocityMass().getDeviceData();

  if (m_UsingHolonomicConstraints) {
    copy_DtoD_async<double4>(coords, coordsRefDevice, numAtoms,
                             *m_IntegratorStream);
    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

    m_HolonomicConstraint->handleHolonomicConstraints(coordsRefDevice);
    updateSPKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
        numAtoms, xyzq, coords);
    copy_DtoD_async<double4>(coords, coordsRefDevice, numAtoms,
                             *m_IntegratorStream);
    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  }

  m_Context->calculateForces();
  auto force = m_Context->getForces();

  int stride = m_Context->getForceStride();
  double kbt = charmm::constants::kBoltz * m_BathTemperature;

  init<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      kbt, numAtoms, stride, m_TimeStep, // coords,
      coordsDeltaDevice, coordsDeltaPreviousDevice, velMass, force->xyz());
  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  if (m_UsingHolonomicConstraints) {
    backStepInitializationKernel<<<numBlocks, numThreads, 0,
                                   *m_IntegratorStream>>>(
        numAtoms, coords, coordsDeltaPreviousDevice);

    m_HolonomicConstraint->handleHolonomicConstraints(coordsRefDevice);

    backStepInitializationKernel2<<<numBlocks, numThreads, 0,
                                    *m_IntegratorStream>>>(
        numAtoms, coords, coordsRefDevice, coordsDeltaPreviousDevice);
  }
  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  return;
}

/* Integrate the forces NOT TAKING THE BAROSTAT INTO ACCOUNT, to compute
the non-barostatted half-step velocities. These will be used as initial
value for the predictor corrector.
*/
__global__ static void nonBarostatHalfStepVelocityUpdate(
    double kbt, const int numAtoms, const int stride, const double timeStep,
    double4 *__restrict__ coords, double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ coordsDeltaPrevious,
    double4 *__restrict__ velMass, const double *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {

    double fx = -force[index];
    double fy = -force[index + stride];
    double fz = -force[index + 2 * stride];

    double fact = timeStep * timeStep * velMass[index].w;

    coordsDelta[index].x = coordsDeltaPrevious[index].x + fact * fx;
    coordsDelta[index].y = coordsDeltaPrevious[index].y + fact * fy;
    coordsDelta[index].z = coordsDeltaPrevious[index].z + fact * fz;

    coords[index].x += coordsDelta[index].x;
    coords[index].y += coordsDelta[index].y;
    coords[index].z += coordsDelta[index].z;
  }
}

/** @brief Computes the kinetic energy contribution to the pressure tensor
 * using previous and next half step velocities. One might think the on-step
 * velocity would be a better thing to use, but it would be unsound (Brooks
 * 1987)
 */
__global__ static void calculateAverageKineticPressureKernel(
    int numAtoms, double timeStep,
    const double4 *__restrict__ coordsDeltaPrevious,
    const double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ velMass, double *accumulant) {

  constexpr int blockSize = 128 * 6;
  __shared__ double sdata[blockSize];

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int threadId = threadIdx.x;

  for (int i = 0; i < 6; i++) {
    if (index == 0) {
      accumulant[i] = 0.0;
    }
    sdata[threadId * 6 + i] = 0.0;
  }

  double timeStepSquared = timeStep * timeStep;
  while (index < numAtoms) {
    double factor = 0.5 / velMass[index].w / timeStepSquared;

    // expand these to\ xx, xy, xz, yy, yz, zz

    sdata[threadId * 6 + 0] +=
        factor * (coordsDelta[index].x * coordsDelta[index].x +
                  coordsDeltaPrevious[index].x * coordsDeltaPrevious[index].x);
    sdata[threadId * 6 + 1] +=
        factor * (coordsDelta[index].x * coordsDelta[index].y +
                  coordsDeltaPrevious[index].x * coordsDeltaPrevious[index].y);
    sdata[threadId * 6 + 2] +=
        factor * (coordsDelta[index].y * coordsDelta[index].y +
                  coordsDeltaPrevious[index].y * coordsDeltaPrevious[index].y);
    sdata[threadId * 6 + 3] +=
        factor * (coordsDelta[index].x * coordsDelta[index].z +
                  coordsDeltaPrevious[index].x * coordsDeltaPrevious[index].z);
    sdata[threadId * 6 + 4] +=
        factor * (coordsDelta[index].y * coordsDelta[index].z +
                  coordsDeltaPrevious[index].y * coordsDeltaPrevious[index].z);
    sdata[threadId * 6 + 5] +=
        factor * (coordsDelta[index].z * coordsDelta[index].z +
                  coordsDeltaPrevious[index].z * coordsDeltaPrevious[index].z);

    index += gridDim.x * blockDim.x;
  }

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (threadId < s) {
      for (int i = 0; i < 6; i++) {
        sdata[threadId * 6 + i] += sdata[(threadId + s) * 6 + i];
      }
    }
  }

  if (threadId == 0) {
    for (int i = 0; i < 6; i++) {
      atomicAdd(accumulant + i, sdata[i]);
    }
  }
}

/** @brief Compute the next-half-step kinetic energy component of the pressure
 * using updated coordsDelta. This is the only component of the pressure that
 * is updated during the pred-corr (the previous-half-step does not change and
 * the virial part is considered constant)
 */
__global__ static void calculateHalfStepKineticPressureKernel(
    int numAtoms, double vcell, const double4 *__restrict__ coordsDelta,
    const double4 *velMass, double *accumulant) {

  constexpr int blockSize = 128 * 6;
  __shared__ double sdata[blockSize];

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int threadId = threadIdx.x;

  for (int i = 0; i < 6; i++) {
    if (index == 0) {
      accumulant[i] = 0.0;
    }
    sdata[threadId * 6 + i] = 0.0;
  }

  while (index < numAtoms) {
    double factor = vcell / velMass[index].w;

    sdata[threadId * 6 + 0] +=
        factor * coordsDelta[index].x * coordsDelta[index].x;
    sdata[threadId * 6 + 1] +=
        factor * coordsDelta[index].x * coordsDelta[index].y;
    sdata[threadId * 6 + 2] +=
        factor * coordsDelta[index].y * coordsDelta[index].y;
    sdata[threadId * 6 + 3] +=
        factor * coordsDelta[index].x * coordsDelta[index].z;
    sdata[threadId * 6 + 4] +=
        factor * coordsDelta[index].y * coordsDelta[index].z;
    sdata[threadId * 6 + 5] +=
        factor * coordsDelta[index].z * coordsDelta[index].z;
    index += gridDim.x * blockDim.x;
  }

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (threadId < s) {
      for (int i = 0; i < 6; i++) {
        sdata[threadId * 6 + i] += sdata[(threadId + s) * 6 + i];
      }
    }
  }

  if (threadId == 0) {
    for (int i = 0; i < 6; i++) {
      atomicAdd(accumulant + i, sdata[i]);
    }
  }
}

// It's an independent function because we have to test it soon as a cuda
// kernel
/** @brief Projects deltaPressure tensor onto piston dofs
 */
void projectDeltaPressureToPistonDof(
    CRYSTAL crystalType, std::vector<double> boxDimensions,
    CudaContainer<double> deltaPressure,
    CudaContainer<double> &pistonDeltaPressure) {

  switch (crystalType) {
  case CRYSTAL::ORTHORHOMBIC:
    pistonDeltaPressure[0] = deltaPressure[0] / boxDimensions[0];
    pistonDeltaPressure[1] = deltaPressure[2] / boxDimensions[1];
    pistonDeltaPressure[2] = deltaPressure[5] / boxDimensions[2];
    break;

  case CRYSTAL::TETRAGONAL:
    pistonDeltaPressure[0] =
        (deltaPressure[0] + deltaPressure[2]) / boxDimensions[0];
    pistonDeltaPressure[1] = deltaPressure[5] / boxDimensions[2];
    break;

  case CRYSTAL::CUBIC:
    pistonDeltaPressure[0] =
        (deltaPressure[0] + deltaPressure[2] + deltaPressure[5]) /
        boxDimensions[0];
    break;
  default:
    break;
  }
}

void projectCrystalDimensionsToPistonPosition(
    CRYSTAL crystalType, std::vector<double> boxDimensions,
    CudaContainer<double> &pistonPosition) {

  switch (crystalType) {
  case CRYSTAL::ORTHORHOMBIC:
    pistonPosition[0] = boxDimensions[0];
    pistonPosition[1] = boxDimensions[1];
    pistonPosition[2] = boxDimensions[2];
    break;
  case CRYSTAL::TETRAGONAL:
    pistonPosition[0] = (boxDimensions[0] + boxDimensions[1]) / 2;
    pistonPosition[1] = boxDimensions[2];
    break;
  case CRYSTAL::CUBIC:
    pistonPosition[0] =
        (boxDimensions[0] + boxDimensions[1] + boxDimensions[2]) / 3;

    break;
  default:
    break;
  }
}

/** boxDimenions is r_n at this time. It will be used to update
 * onStepCrytalFactor and then updated to the value stored in
 * onStepPistonPosition.
 *
 * Both Factors computed (onStepCrystalFactor and halfStepCrystalFactors)
 * are dimensionless quantities.
 *
 * @todo name it sthg like "computeCrystalFactors" ? Also, this updates
 * boxDimensions
 */
void projectPistonQuantitiesToCrystalQuantities(
    CRYSTAL crystalType, double timeStep,
    CudaContainer<double> onStepPistonPosition,
    CudaContainer<double> halfStepPistonPosition,
    CudaContainer<double> onStepPistonVelocity,
    CudaContainer<double> pressurePistonPositionDelta,
    CudaContainer<double> &onStepCrystalFactor,
    CudaContainer<double> &halfStepCrystalFactor,
    std::vector<double> &boxDimensions) {

  switch (crystalType) {
  case CRYSTAL::ORTHORHOMBIC:
    onStepCrystalFactor[0] =
        timeStep * onStepPistonVelocity[0] / boxDimensions[0];
    onStepCrystalFactor[1] =
        timeStep * onStepPistonVelocity[1] / boxDimensions[1];
    onStepCrystalFactor[2] =
        timeStep * onStepPistonVelocity[2] / boxDimensions[2];

    halfStepCrystalFactor[0] =
        pressurePistonPositionDelta[0] / halfStepPistonPosition[0];
    halfStepCrystalFactor[1] =
        pressurePistonPositionDelta[1] / halfStepPistonPosition[1];
    halfStepCrystalFactor[2] =
        pressurePistonPositionDelta[2] / halfStepPistonPosition[2];

    boxDimensions[0] = onStepPistonPosition[0];
    boxDimensions[1] = onStepPistonPosition[1];
    boxDimensions[2] = onStepPistonPosition[2];
    break;

  case CRYSTAL::TETRAGONAL:
    onStepCrystalFactor[0] =
        timeStep * onStepPistonVelocity[0] / boxDimensions[0];
    onStepCrystalFactor[1] = onStepCrystalFactor[0];
    onStepCrystalFactor[2] =
        timeStep * onStepPistonVelocity[1] / boxDimensions[2];

    halfStepCrystalFactor[0] =
        pressurePistonPositionDelta[0] / halfStepPistonPosition[0];
    halfStepCrystalFactor[1] = halfStepCrystalFactor[0];
    halfStepCrystalFactor[2] =
        pressurePistonPositionDelta[1] / halfStepPistonPosition[1];

    boxDimensions[0] = onStepPistonPosition[0];
    boxDimensions[1] = onStepPistonPosition[0];
    boxDimensions[2] = onStepPistonPosition[1];
    break;

  case CRYSTAL::CUBIC:
    onStepCrystalFactor[0] =
        timeStep * onStepPistonVelocity[0] / boxDimensions[0];
    onStepCrystalFactor[1] = onStepCrystalFactor[0];
    onStepCrystalFactor[2] = onStepCrystalFactor[0];

    halfStepCrystalFactor[0] =
        pressurePistonPositionDelta[0] / halfStepPistonPosition[0];
    halfStepCrystalFactor[1] = halfStepCrystalFactor[0];
    halfStepCrystalFactor[2] = halfStepCrystalFactor[0];

    boxDimensions[0] = onStepPistonPosition[0];
    boxDimensions[1] = onStepPistonPosition[0];
    boxDimensions[2] = onStepPistonPosition[0];
    break;
  default:
    break;
  }
}

/** @brief One iteration of the predictor-corrector for the crystal dof
 * velocity, the predicted coordinate change and the predicted coordinates
 */
__global__ void predictorCorrectorKernel(
    bool noseHooverFlag, int numAtoms, double timeStep,
    double noseHooverPistonVelocity, double4 *coordsRef, double4 *velMass,
    double4 *coords, double4 *coordsDeltaPrevious, double4 *coordsDelta,
    double4 *coordsDeltaPredicted, double *onStepCrystalFactor,
    double *halfStepCrystalFactor) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index;
  if (i < numAtoms) {
    double onStepVelocityX =
        (coordsDeltaPrevious[i].x + coordsDeltaPredicted[i].x) / (2 * timeStep);
    double onStepVelocityY =
        (coordsDeltaPrevious[i].y + coordsDeltaPredicted[i].y) / (2 * timeStep);
    double onStepVelocityZ =
        (coordsDeltaPrevious[i].z + coordsDeltaPredicted[i].z) / (2 * timeStep);

    // v(t+ dt/2)*dt = [v(t-dt/2)*dt + f(t)*dt^2/m] - h.(t)/h(t)*v(t)*dt^2
    coordsDeltaPredicted[i].x =
        coordsDelta[i].x - onStepCrystalFactor[0] * onStepVelocityX * timeStep;
    coordsDeltaPredicted[i].y =
        coordsDelta[i].y - onStepCrystalFactor[1] * onStepVelocityY * timeStep;
    coordsDeltaPredicted[i].z =
        coordsDelta[i].z - onStepCrystalFactor[2] * onStepVelocityZ * timeStep;

    if (noseHooverFlag) {
      coordsDeltaPredicted[i].x -=
          timeStep * timeStep * onStepVelocityX * noseHooverPistonVelocity;
      coordsDeltaPredicted[i].y -=
          timeStep * timeStep * onStepVelocityY * noseHooverPistonVelocity;
      coordsDeltaPredicted[i].z -=
          timeStep * timeStep * onStepVelocityZ * noseHooverPistonVelocity;
    }

    velMass[i].x = onStepVelocityX;
    velMass[i].y = onStepVelocityY;
    velMass[i].z = onStepVelocityX;

    double halfStepPositionX = (coordsRef[i].x + coords[i].x) / 2.0;
    double halfStepPositionY = (coordsRef[i].y + coords[i].y) / 2.0;
    double halfStepPositionZ = (coordsRef[i].z + coords[i].z) / 2.0;

    double scaledCrystalVelocityX =
        halfStepCrystalFactor[0] * halfStepPositionX;
    double scaledCrystalVelocityY =
        halfStepCrystalFactor[1] * halfStepPositionY;
    double scaledCrystalVelocityZ =
        halfStepCrystalFactor[2] * halfStepPositionZ;

    coords[i].x =
        coordsRef[i].x + coordsDeltaPredicted[i].x + scaledCrystalVelocityX;
    coords[i].y =
        coordsRef[i].y + coordsDeltaPredicted[i].y + scaledCrystalVelocityY;
    coords[i].z =
        coordsRef[i].z + coordsDeltaPredicted[i].z + scaledCrystalVelocityZ;
  }
}

__global__ static void prepareCoordsRefForHolonomicConstraintsKernel(
    int numAtoms, double4 *__restrict__ coordsRef,
    const double4 *__restrict__ coords, double *halfStepCrystalFactor) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index;
  if (i < numAtoms) {
    double halfStepPositionX = (coordsRef[i].x + coords[i].x) / 2.0;
    double halfStepPositionY = (coordsRef[i].y + coords[i].y) / 2.0;
    double halfStepPositionZ = (coordsRef[i].z + coords[i].z) / 2.0;

    double scaledCrystalVelocityX =
        halfStepCrystalFactor[0] * halfStepPositionX;
    double scaledCrystalVelocityY =
        halfStepCrystalFactor[1] * halfStepPositionY;
    double scaledCrystalVelocityZ =
        halfStepCrystalFactor[2] * halfStepPositionZ;

    coordsRef[i].x += scaledCrystalVelocityX;
    coordsRef[i].y += scaledCrystalVelocityY;
    coordsRef[i].z += scaledCrystalVelocityZ;
  }
}

__global__ static void updateCoordsDeltaPredictedKernel(
    int numAtoms, double4 *__restrict__ coordsDeltaPredicted,
    const double4 *__restrict__ coords, const double4 *__restrict__ coordsRef) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index;
  if (i < numAtoms) {
    coordsDeltaPredicted[i].x = coords[i].x - coordsRef[i].x;
    coordsDeltaPredicted[i].y = coords[i].y - coordsRef[i].y;
    coordsDeltaPredicted[i].z = coords[i].z - coordsRef[i].z;
  }
}

/** @brief Given coordsDelta of previous and next half steps, returns the
 * on-step velocity
 */
__global__ static void
onStepVelocityCalculation(const int numAtoms, const double timeStep,
                          double4 *__restrict__ coordsDelta,
                          double4 *__restrict__ coordsDeltaPrevious,
                          double4 *__restrict__ velMass) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {

    double fact = 0.5 / timeStep;

    velMass[index].x =
        (coordsDelta[index].x + coordsDeltaPrevious[index].x) * fact;
    velMass[index].y =
        (coordsDelta[index].y + coordsDeltaPrevious[index].y) * fact;
    velMass[index].z =
        (coordsDelta[index].z + coordsDeltaPrevious[index].z) * fact;
  }
}

__global__ static void updateCoordsDeltaAfterConstraint(
    int numAtoms, const double4 *__restrict__ coordsRef,
    const double4 *__restrict__ coords, double4 *__restrict__ coordsDelta) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < numAtoms) {
    coordsDelta[index].x = coords[index].x - coordsRef[index].x;
    coordsDelta[index].y = coords[index].y - coordsRef[index].y;
    coordsDelta[index].z = coords[index].z - coordsRef[index].z;
  }
}

__global__ static void computeHolonomicConstraintForces(
    int numAtoms, double timeStep, const double4 *__restrict__ velMass,
    const double4 *__restrict__ coordsRef, const double4 *__restrict__ coords,
    const double4 *__restrict__ coordsDelta,
    double4 *__restrict__ holonomicConstraintForces) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  double timeStepSquared = timeStep * timeStep;

  if (index < numAtoms) {
    double factor = 1.0 / (velMass[index].w * timeStepSquared);

    double3 delta = make_double3(
        (coords[index].x - coordsRef[index].x - coordsDelta[index].x) * factor,
        (coords[index].y - coordsRef[index].y - coordsDelta[index].y) * factor,
        (coords[index].z - coordsRef[index].z - coordsDelta[index].z) * factor);

    holonomicConstraintForces[index].x = delta.x;
    holonomicConstraintForces[index].y = delta.y;
    holonomicConstraintForces[index].z = delta.z;
  }
}

__global__ static void computeHolonomicConstraintVirial(
    int numAtoms, const double4 *__restrict__ coordsRef,
    const double4 *__restrict__ holonomicConstraintForces,
    double *__restrict__ accumulant) {
  constexpr int blockSize = 128 * 6;
  __shared__ double sdata[blockSize];

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int threadId = threadIdx.x;

  for (int i = 0; i < 6; i++) {
    if (index == 0) {
      accumulant[i] = 0.0;
    }
    sdata[threadId * 6 + i] = 0.0;
  }

  while (index < numAtoms) {

    sdata[threadId * 6 + 0] +=
        coordsRef[index].x * holonomicConstraintForces[index].x;
    sdata[threadId * 6 + 1] +=
        coordsRef[index].x * holonomicConstraintForces[index].y;
    sdata[threadId * 6 + 2] +=
        coordsRef[index].y * holonomicConstraintForces[index].y;
    sdata[threadId * 6 + 3] +=
        coordsRef[index].x * holonomicConstraintForces[index].z;
    sdata[threadId * 6 + 4] +=
        coordsRef[index].y * holonomicConstraintForces[index].z;
    sdata[threadId * 6 + 5] +=
        coordsRef[index].z * holonomicConstraintForces[index].z;
    index += gridDim.x * blockDim.x;
  }

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (threadId < s) {
      for (int i = 0; i < 6; i++) {
        sdata[threadId * 6 + i] += sdata[(threadId + s) * 6 + i];
      }
    }
  }

  if (threadId == 0) {
    for (int i = 0; i < 6; i++) {
      atomicAdd(accumulant + i, sdata[i]);
    }
  }
}

// TODO : do this in the imageCentering kernel in CharmmContext itself
__global__ static void invertDeltaAsymmetric(int numGroups,
                                             const int2 *__restrict__ groups,
                                             float boxx, const float4 *xyzq,
                                             int stride,
                                             double4 *coordsDeltaPrevious) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index < numGroups) {
    int2 group = groups[index];

    float gx = 0.0;
    float gy = 0.0;
    float gz = 0.0;

    for (int i = group.x; i <= group.y; ++i) {
      gx += xyzq[i].x;
      gy += xyzq[i].y;
      gz += xyzq[i].z;
    }

    gx /= (group.y - group.x + 1);
    gy /= (group.y - group.x + 1);
    gz /= (group.y - group.x + 1);

    if (gx > boxx / 2.0 || gx < -boxx / 2.0) {
      for (int i = group.x; i <= group.y; ++i) {
        coordsDeltaPrevious[i].y *= -1.0;
        coordsDeltaPrevious[i].z *= -1.0;
      }
    }
  }
}

/*
 - (update nblist if needed)
 - (copy coords in coordsRef)
 - compute forces, energies, virial
 - Integrate without barostat
 - Prepare pressure terms :
   * Compute kinetic pressure using halfsteps velocities
   * Compute pressureTensor from virial + kinetic
   * Compute deltaPressure
   * compute the next-half-step term of the kinetic component of the
 pressure
   * compute the non-changing part of deltaPressure (deltaPressure -
 next-half-step kinetic term)
 - (store pressurepistonPositionDelta, store boxdimensions)
 - START PREDCORR LOOP
    * (reset pressurePistonPositionDelta and boxDimensions to stored values)
    * Project deltaPressure onto piston dofs
    * update pressurePistonPositionDelta (hdot)

*/

void CudaLangevinPistonIntegrator::propagateOneStep(void) {
  auto coords = m_Context->getCoordinatesCharges().getDeviceData();
  auto xyzq = m_Context->getXYZQ()->getDeviceXYZQ();
  auto coordsDeltaDevice = m_CoordsDelta.getDeviceData();
  auto coordsDeltaPreviousDevice = m_CoordsDeltaPrevious.getDeviceData();
  auto coordsRefDevice = m_CoordsRef.getDeviceData();
  auto velMass = m_Context->getVelocityMass().getDeviceData();

  int numDegreesOfFreedom = m_Context->getDegreesOfFreedom();

  double referenceKineticEnergy =
      0.5 * numDegreesOfFreedom * charmm::constants::kBoltz * m_BathTemperature;

  std::vector<double> boxDimensions = m_Context->getBoxDimensions();

  if (m_DebugPrintFrequency > 0 && m_StepId % m_DebugPrintFrequency == 0) {
    std::cout << "\n Step id : " << m_StepId << "\n---\n";
    for (int i = 0; i < m_CrystalDegreesOfFreedom; ++i) {
      std::cout << "boxDimensions[" << i << "] = " << boxDimensions[i]
                << std::endl;
    }
  }

  auto volume = m_Context->getVolume();
  int numAtoms = m_Context->getNumAtoms();
  int stride = m_Context->getForceStride();
  double kbt = charmm::constants::kBoltz * m_BathTemperature;
  // const double gamma = m_TimeStep * m_Timfac * m_Friction;

  std::normal_distribution<double> dist(0, 1);

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;
  int numBlocksReduction = 64;

  if (m_StepsSinceNeighborListUpdate % m_NonbondedListUpdateFrequency == 0) {
    if (m_Context->getForceManager()->getPeriodicBoundaryCondition() ==
        PBC::P21) {
      auto groups = m_Context->getForceManager()->getPSF()->getGroups();

      // find a better place for this
      int numGroups = groups.size();
      int numThreads = 128;
      int numBlocks = (numGroups - 1) / numThreads + 1;

      auto boxDimensions = m_Context->getBoxDimensions();
      float3 box = {(float)boxDimensions[0], (float)boxDimensions[1],
                    (float)boxDimensions[2]};

      invertDeltaAsymmetric<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          numGroups, groups.getDeviceData(), box.x, xyzq, stride,
          coordsDeltaPreviousDevice);
      cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
    }
    m_Context->resetNeighborList();
  }

  if (m_StepId % m_RemoveCenterOfMassFrequency == 0)
    this->removeCenterOfMassMotion();

  copy_DtoD_async<double4>(coords, m_CoordsRef.getDeviceData(), numAtoms,
                           *m_IntegratorStream);

  // {
  //   auto tmp = m_Context->getCoordinatesCharges();
  //   tmp.transferFromDevice();
  //   std::cout << std::scientific << std::setprecision(8);
  //   std::cout << "coords[0] = {" << tmp[0].x << ", " << tmp[0].y << ", "
  //             << tmp[0].z << "}" << std::endl;
  // }

  m_Context->calculateForces(false, true, true);
  auto force = m_Context->getForces();

  // cudaCheck(cudaDeviceSynchronize()); // Returning from here eliminates error
  // return;                             // Returning from here eliminates error

  // if (m_StepId % m_RemoveCenterOfMassFrequency == 0) {
  //  this->removeCenterOfMassAverageNetForce();
  //}

  m_NoseHooverPistonVelocityPrevious = m_NoseHooverPistonVelocity;
  m_NoseHooverPistonForcePrevious = m_NoseHooverPistonForce;

  // {
  //   auto tmp = m_Context->getVelocityMass();
  //   tmp.transferFromDevice();
  //   std::cout << std::scientific << std::setprecision(8);
  //   std::cout << "velMass[0] = {" << tmp[0].x << ", " << tmp[0].y << ", "
  //             << tmp[0].z << "}" << std::endl;
  // }

  nonBarostatHalfStepVelocityUpdate<<<numBlocks, numThreads, 0,
                                      *m_IntegratorStream>>>(
      kbt, numAtoms, stride, m_TimeStep, coords, coordsDeltaDevice,
      coordsDeltaPreviousDevice, velMass, force->xyz());
  // cudaCheck(cudaDeviceSynchronize()); // Returning from here eliminates error
  // return;                             // Returning from here eliminates error

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  // {
  //   int idx = 1;
  //   auto tmp = m_Context->getVelocityMass();
  //   tmp.transferFromDevice();
  //   std::cout << std::scientific << std::setprecision(8);
  //   std::cout << idx << ": velMass = {" << tmp[idx].x << ", " << tmp[idx].y
  //             << ", " << tmp[idx].z << ", " << tmp[idx].w << "}" <<
  //             std::endl;
  // }
  auto virialTensor = m_Context->getVirial();
  virialTensor.transferFromDevice();

  // std::cout << std::scientific << std::setprecision(8);
  // for (int i = 0; i < 9; i++)
  //   std::cout << i << ": virialTensor = " << virialTensor[i] << std::endl;

  // TODO :  Use profiler to determine where we do this computation
  if (m_UsingHolonomicConstraints) {
    m_HolonomicVirial.setToValue(0.0);

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

    m_HolonomicConstraint->handleHolonomicConstraints(coordsRefDevice);
    cudaCheck(cudaDeviceSynchronize());

    computeHolonomicConstraintForces<<<numBlocks, numThreads, 0,
                                       *m_IntegratorStream>>>(
        numAtoms, m_TimeStep, velMass, coordsRefDevice, coords,
        coordsDeltaDevice, m_HolonomicConstraintForces.getDeviceData());

    computeHolonomicConstraintVirial<<<numBlocksReduction, numThreads, 0,
                                       *m_IntegratorStream>>>(
        numAtoms, coordsRefDevice, m_HolonomicConstraintForces.getDeviceData(),
        m_HolonomicVirial.getDeviceData());

    updateCoordsDeltaAfterConstraint<<<numBlocks, numThreads, 0,
                                       *m_IntegratorStream>>>(
        numAtoms, coordsRefDevice, coords, coordsDeltaDevice);

    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
    m_HolonomicVirial.transferFromDevice();

    virialTensor[0] += m_HolonomicVirial[0];
    virialTensor[1] += m_HolonomicVirial[1];
    virialTensor[2] += m_HolonomicVirial[3];
    virialTensor[3] += m_HolonomicVirial[1];
    virialTensor[4] += m_HolonomicVirial[2];
    virialTensor[5] += m_HolonomicVirial[4];
    virialTensor[6] += m_HolonomicVirial[3];
    virialTensor[7] += m_HolonomicVirial[4];
    virialTensor[8] += m_HolonomicVirial[5];

    virialTensor.transferToDevice();
  }

  m_KineticEnergyPressureTensor.setToValue(0.0);
  calculateAverageKineticPressureKernel<<<numBlocksReduction, numThreads, 0,
                                          *m_IntegratorStream>>>(
      numAtoms, m_TimeStep, coordsDeltaPreviousDevice, coordsDeltaDevice,
      velMass, m_KineticEnergyPressureTensor.getDeviceData());
  // cudaCheck(cudaDeviceSynchronize()); // Returning from here eliminates error
  // return;                             // Returning from here eliminates error

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  m_KineticEnergyPressureTensor.transferFromDevice();

  // // kineticEnergyPressureTensor differs
  // std::cout << std::scientific << std::setprecision(8);
  // for (int i = 0; i < 6; i++)
  //   std::cout << i << ": m_KineticEnergyPressureTensor = "
  //             << m_KineticEnergyPressureTensor[i] << std::endl;

  // for (int i = 0; i < 6; i++)
  //   std::cout << i << ": pressureTensor = " << m_PressureTensor[i] <<
  //   std::endl;

  double volumeFactor = charmm::constants::patmos / volume;
  m_PressureTensor[0] =
      (virialTensor[0] + m_KineticEnergyPressureTensor[0]) * volumeFactor;
  m_PressureTensor[1] =
      (virialTensor[3] + m_KineticEnergyPressureTensor[1]) * volumeFactor;
  m_PressureTensor[2] =
      (virialTensor[4] + m_KineticEnergyPressureTensor[2]) * volumeFactor;
  m_PressureTensor[3] =
      (virialTensor[6] + m_KineticEnergyPressureTensor[3]) * volumeFactor;
  m_PressureTensor[4] =
      (virialTensor[7] + m_KineticEnergyPressureTensor[4]) * volumeFactor;
  m_PressureTensor[5] =
      (virialTensor[8] + m_KineticEnergyPressureTensor[5]) * volumeFactor;

  m_PressureScalar[0] =
      (m_PressureTensor[0] + m_PressureTensor[2] + m_PressureTensor[5]) / 3.0;
  m_PressureScalar.transferToDevice();

  if (m_ConstantSurfaceTensionFlag) {
    m_ReferencePressure[0] =
        (m_ReferencePressure[5] - m_SurfaceTension *
                                      charmm::constants::surfaceTensionFactor /
                                      boxDimensions[2]);
    m_ReferencePressure[2] = m_ReferencePressure[0];
  }

  // for (int i = 0; i < 6; i++)
  //   std::cout << i << ": pressureTensor = " << m_PressureTensor[i] <<
  //   std::endl;

  // std::cout << "BEFORE UPDATE" << std::endl;
  // for (int i = 0; i < 6; i++)
  //   std::cout << i << ": deltaPressure = " << m_DeltaPressure[i] <<
  //   std::endl;
  m_DeltaPressure[0] = m_PressureTensor[0] - m_ReferencePressure[0];
  m_DeltaPressure[1] = m_PressureTensor[1] - m_ReferencePressure[1];
  m_DeltaPressure[2] = m_PressureTensor[2] - m_ReferencePressure[2];
  m_DeltaPressure[3] = m_PressureTensor[3] - m_ReferencePressure[3];
  m_DeltaPressure[4] = m_PressureTensor[4] - m_ReferencePressure[4];
  m_DeltaPressure[5] = m_PressureTensor[5] - m_ReferencePressure[5];
  m_DeltaPressure.transferToDevice();
  // std::cout << "AFTER UPDATE" << std::endl;
  // for (int i = 0; i < 6; i++)
  //   std::cout << i << ": deltaPressure = " << m_DeltaPressure[i] <<
  //   std::endl;

  auto vcell =
      0.25 * charmm::constants::patmos / volume / (m_TimeStep * m_TimeStep);
  m_DeltaPressureHalfStepKinetic.setToValue(0.0);
  calculateHalfStepKineticPressureKernel<<<numBlocksReduction, numThreads, 0,
                                           *m_IntegratorStream>>>(
      numAtoms, vcell, coordsDeltaDevice, velMass,
      m_DeltaPressureHalfStepKinetic.getDeviceData());
  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  // cudaCheck(cudaDeviceSynchronize()); // Returning from here eliminates error
  // return;                             // Returning from here eliminates error

  m_DeltaPressureHalfStepKinetic.transferFromDevice();

  for (int i = 0; i < 6; i++) {
    m_DeltaPressureNonChanging[i] =
        m_DeltaPressure[i] - m_DeltaPressureHalfStepKinetic[i];
  }

  for (int i = 0; i < m_PistonDegreesOfFreedom; i++)
    m_PressurePistonPositionDeltaStored[i] = m_PressurePistonPositionDelta[i];

  copy_DtoD_async<double4>(coordsDeltaDevice,
                           m_CoordsDeltaPredicted.getDeviceData(), numAtoms,
                           *m_IntegratorStream);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  // cudaCheck(cudaDeviceSynchronize()); // Returning from here eliminates error
  // return;                             // Returning from here eliminates error

  std::vector<double> boxDimensionsStored = boxDimensions;
  // for (int i = 0; i < crystalDegreesOfFreedom; i++) {
  //   boxDimensionsStored.push_back(boxDimensions[i]);
  // }

  double surfaceTensionInstantaneous =
      0.5 * boxDimensions[2] *
      (m_PressureTensor[5] -
       0.5 * (m_PressureTensor[0] + m_PressureTensor[2])) /
      charmm::constants::surfaceTensionFactor;

  // predictor corrector loop
  ///////////////////////////////

  for (int iter = 0; iter < m_MaxPredictorCorrectorSteps; iter++) {
    boxDimensions = boxDimensionsStored;
    // for (int i = 0; i < m_CrystalDegreesOfFreedom; i++)
    //   boxDimensions[i] = boxDimensionsStored[i];

    for (int i = 0; i < m_PistonDegreesOfFreedom; i++)
      m_PressurePistonPositionDelta[i] = m_PressurePistonPositionDeltaStored[i];

    // TODO : move to the device
    // for (int i = 0; i < 6; i++)
    //   std::cout << i << ": deltaPressure = " << m_DeltaPressure[i] <<
    //   std::endl;
    projectDeltaPressureToPistonDof(m_CrystalType, boxDimensions,
                                    m_DeltaPressure, m_PistonDeltaPressure);

    double fact = volume * m_Pbfact * charmm::constants::atmosp;

    for (int i = 0; i < m_PistonDegreesOfFreedom; i++) {
      m_PressurePistonPositionDeltaPrevious[i] =
          m_PressurePistonPositionDelta[i];
      // double randVal = dist(rng);
      // double rdum = pbfact * prfwd[i] * randVal;
      // std::cout << "=============================================" <<
      // std::endl; std::cout << "                    iter, i = " << iter << ",
      // " << i
      //           << std::endl;
      // std::cout << "                     palpha = " << palpha << std::endl;
      // std::cout << "pressurePistonPositionDelta = "
      //           << pressurePistonPositionDelta[i] << std::endl;
      // std::cout << "          inversePistonMass = " << inversePistonMass[i]
      //           << std::endl;
      // std::cout << "        pistonDeltaPressure = " << pistonDeltaPressure[i]
      //           << std::endl;
      // std::cout << "                       fact = " << fact << std::endl;
      // // std::cout << "palpha * pressurePistonPositionDelta = "
      // //           << palpha * pressurePistonPositionDelta[i] << std::endl;
      // // std::cout << "inversePistonMass * pistonDeltaPressure * fact = "
      // //           << inversePistonMass[i] * pistonDeltaPressure[i] * fact
      // //           << std::endl;
      // std::cout << "                    randVal = " << randVal << std::endl;
      // std::cout << "                     pbfact = " << pbfact << std::endl;
      // std::cout << "                      prfwd = " << prfwd[i] << std::endl;
      // std::cout << "                       rdum = " << rdum << std::endl;
      // pressurePistonPositionDelta[i] =
      //     palpha * pressurePistonPositionDelta[i] +
      // inversePistonMass[i] * pistonDeltaPressure[i] * fact + rdum;

      m_PressurePistonPositionDelta[i] =
          m_Palpha * m_PressurePistonPositionDelta[i] +
          m_InversePistonMass[i] * m_PistonDeltaPressure[i] * fact +
          m_Pbfact * m_Prfwd[i] * dist(m_Rng);

      // std::cout << std::scientific << std::setprecision(8);
      // std::cout << "BEFORE: pressurePistonPositionDelta[" << i
      //           << "] = " << pressurePistonPositionDelta[i] << std::endl;
      // double randVal = dist(rng);
      // pressurePistonPositionDelta[i] =
      //     palpha * pressurePistonPositionDelta[i] +
      //     inversePistonMass[i] * pistonDeltaPressure[i] * fact +
      //     pbfact * prfwd[i] * randVal;
      // std::cout << "palpha = " << palpha << std::endl;
      // std::cout << "inversePistonMass[" << i << "] = " <<
      // inversePistonMass[i]
      //           << std::endl;
      // std::cout << "pistonDeltaPressure[" << i
      //           << "] = " << pistonDeltaPressure[i] << std::endl;
      // std::cout << "fact = " << fact << std::endl;
      // std::cout << "pbfact = " << pbfact << std::endl;
      // std::cout << "prfwd[" << i << "] = " << prfwd[i] << std::endl;
      // std::cout << "randVal = " << randVal << std::endl;
      // std::cout << "AFTER: pressurePistonPositionDelta[" << i
      //           << "] = " << pressurePistonPositionDelta[i] << std::endl;

      m_OnStepPistonVelocity[i] = (m_PressurePistonPositionDelta[i] +
                                   m_PressurePistonPositionDeltaPrevious[i]) /
                                  (2.0 * m_TimeStep);

      // std::cout << "onStepPistonVelocity[" << i
      //           << "] = " << onStepPistonVelocity[i] << std::endl;
    }
    projectCrystalDimensionsToPistonPosition(m_CrystalType, boxDimensions,
                                             m_OnStepPistonPosition);
    // cudaCheck(cudaDeviceSynchronize()); // Returning from here eliminates
    // error return;                             // Returning from here
    // eliminates error

    // Commenting out this loop makes it work
    for (int i = 0; i < m_PistonDegreesOfFreedom; i++) {
      m_OnStepPistonPosition[i] += m_PressurePistonPositionDelta[i];
      m_HalfStepPistonPosition[i] =
          m_OnStepPistonPosition[i] - m_PressurePistonPositionDelta[i] / 2.0;
    }

    // Calculate nose hoover thermal piston velocity and position
    if (m_NoseHooverFlag) {
      // TODO : change this to on step kinetic energy
      /*double onStepKineticEnergy =
          (deltaPressureHalfStepKinetic[0] + deltaPressureHalfStepKinetic[2] +
           deltaPressureHalfStepKinetic[5]) /
          (0.5 * charmm::constants::patmos / volume);
      */
      double onStepKineticEnergy =
          m_Context->computeTemperature() *
          (0.5 * numDegreesOfFreedom * charmm::constants::kBoltz);
      m_NoseHooverPistonForce = 2.0 * m_TimeStep *
                                (onStepKineticEnergy - referenceKineticEnergy) /
                                m_NoseHooverPistonMass;
      if (m_NoseHooverPistonForcePrevious == 0.0)
        m_NoseHooverPistonForcePrevious = m_NoseHooverPistonForce;

      m_NoseHooverPistonVelocity =
          m_NoseHooverPistonVelocityPrevious +
          (m_NoseHooverPistonForce + m_NoseHooverPistonForcePrevious) / 2.0;
      // std::cout << "onStepKineticEnergy: " << onStepKineticEnergy <<
      // std::endl; std::cout << "referenceKineticEnergy: " <<
      // referenceKineticEnergy
      //          << std::endl;
      // std::cout << "pistonNHmass: " << noseHooverPistonMass <<
      // std::endl; std::cout << "noseHooverPistonForce: " <<
      // noseHooverPistonForce
      //          << std::endl;
    }

    //    std::cout << "noseHooverPistonVelocity: " << noseHooverPistonVelocity
    //              << std::endl;
    //    std::cout << "noseHooverPistonForcePrevious: "
    //              << noseHooverPistonForcePrevious << std::endl;
    //    std::cout << "noseHooverPistonVelocityPrevious: "
    //              << noseHooverPistonVelocityPrevious << std::endl;

    // std::cout << "BEFORE UPDATE:\n";
    // std::cout << "boxDimensions = {" << boxDimensions[0] << ", "
    //           << boxDimensions[1] << ", " << boxDimensions[2] << "}"
    //           << std::endl;
    // std::cout << "onStepCrystalFactor = {" << onStepCrystalFactor[0] << ", "
    //           << onStepCrystalFactor[1] << ", " << onStepCrystalFactor[2] <<
    //           "}"
    //           << std::endl;
    // std::cout << "halfStepCrystalFactor = {" << halfStepCrystalFactor[0] <<
    // ", "
    //           << halfStepCrystalFactor[1] << ", " << halfStepCrystalFactor[2]
    //           << "}" << std::endl;
    projectPistonQuantitiesToCrystalQuantities(
        m_CrystalType, m_TimeStep, m_OnStepPistonPosition,
        m_HalfStepPistonPosition, m_OnStepPistonVelocity,
        m_PressurePistonPositionDelta, m_OnStepCrystalFactor,
        m_HalfStepCrystalFactor, boxDimensions);
    // std::cout << "AFTER UPDATE:\n";
    // std::cout << "boxDimensions = {" << boxDimensions[0] << ", "
    //           << boxDimensions[1] << ", " << boxDimensions[2] << "}"
    //           << std::endl;
    // std::cout << "onStepCrystalFactor = {" << onStepCrystalFactor[0] << ", "
    //           << onStepCrystalFactor[1] << ", " << onStepCrystalFactor[2] <<
    //           "}"
    //           << std::endl;
    // std::cout << "halfStepCrystalFactor = {" << halfStepCrystalFactor[0] <<
    // ", "
    //           << halfStepCrystalFactor[1] << ", " << halfStepCrystalFactor[2]
    //           << "}" << std::endl;
    // cudaCheck(cudaDeviceSynchronize()); // Returning from here eliminates
    // error return;                             // Returning from here
    // eliminates error

    m_OnStepCrystalFactor.transferToDevice();
    m_HalfStepCrystalFactor.transferToDevice();

    //        std::cout << " -- before predCorrKernel (iter " << iter << ")" <<
    //        std::endl; coordsDeltaPredicted.transferFromDevice(); std::cout <<
    //        "coordsDeltaPredicted: "
    //                  << coordsDeltaPredicted.getHostArray().data()[0].x <<
    //                  std::endl;
    //        auto velmassprint = context->getVelocityMass();
    //        velmassprint.transferFromDevice();
    //        std::cout << "velMass: " <<
    //        velmassprint.getHostArray().data()[0].x
    //                  << std::endl;
    //        coordsDeltaPrevious.transferFromDevice();
    //                  << coordsDeltaPrevious.getHostArray().data()[0].x <<
    //                  std::endl;
    //        coordsRef.transferFromDevice();
    //        std::cout << "coordsRef: " << coordsRef.getHostArray().data()[0].x
    //                  << std::endl;
    //        std::cout << "onStepCrystalFactor: " << onStepCrystalFactor[0] <<
    //        std::endl; std::cout << "halfStepCrystalFactor: " <<
    //        halfStepCrystalFactor[0]
    //                  << std::endl;
    //        coordsDelta.transferFromDevice();
    //        std::cout << "coordsDelta: " <<
    //        coordsDelta.getHostArray().data()[0].x
    //                  << std::endl;
    //        std::cout << "timestep: " << timeStep << std::endl;
    //        std::cout << "noseHooverPistonVelocity: " <<
    //        noseHooverPistonVelocity
    //                  << std::endl;

    predictorCorrectorKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
        m_NoseHooverFlag, numAtoms, m_TimeStep, m_NoseHooverPistonVelocity,
        m_CoordsRef.getDeviceData(), velMass, coords, coordsDeltaPreviousDevice,
        coordsDeltaDevice, m_CoordsDeltaPredicted.getDeviceData(),
        m_OnStepCrystalFactor.getDeviceData(),
        m_HalfStepCrystalFactor.getDeviceData());
    // cudaCheck(cudaDeviceSynchronize()); // Returning from here eliminates
    // error return;                             // Returning from here
    // eliminates error

    m_DeltaPressureHalfStepKinetic.setToValue(0.0);

    //    std::cout << " -- before hafstepkinpress" << std::endl;
    //    coordsDeltaPredicted.transferFromDevice();
    //    std::cout << "coordsDeltaPredicted: "
    //              << coordsDeltaPredicted.getHostArray().data()[0].x <<
    //              std::endl;
    //    velmassprint = context->getVelocityMass();
    //    velmassprint.transferFromDevice();
    //    std::cout << "velMass: " << velmassprint.getHostArray().data()[0].x
    //              << std::endl;
    //
    calculateHalfStepKineticPressureKernel<<<numBlocks, numThreads, 0,
                                             *m_IntegratorStream>>>(
        numAtoms, vcell, m_CoordsDeltaPredicted.getDeviceData(), velMass,
        m_DeltaPressureHalfStepKinetic.getDeviceData());
    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
    // cudaCheck(cudaDeviceSynchronize()); // Returning from here eliminates
    // error return;                             // Returning from here
    // eliminates error

    // xx, yy terms have been checked on CPU side and correspond

    m_DeltaPressureHalfStepKinetic.transferFromDevice();

    for (int i = 0; i < 6; i++) {
      m_DeltaPressure[i] =
          m_DeltaPressureNonChanging[i] + m_DeltaPressureHalfStepKinetic[i];
    }

    // if (iter == 1) {
    //   cudaCheck(cudaDeviceSynchronize()); // Error persists returning from
    //   here return;                             // Error persists returning
    //   from here
    // }
    cudaCheck(cudaDeviceSynchronize());
  }
  //
  // end predcorr loop
  /////////////////////

  // cudaCheck(cudaDeviceSynchronize()); // Error persists returning from here
  // return;                             // Error persists returning from here

  //  std::cout << "after predcor" << std::endl;
  //  std::cout << "deltaPressureNonChanging: " << deltaPressureNonChanging[0]
  //            << " deltaPressureHalfStepKinetic: "
  //            << deltaPressureHalfStepKinetic[0] << std::endl;
  //  std::cout << "deltaPressure[0] = " << deltaPressure[0] << std::endl;
  //  coordsPrinter = context->getCoordinatesCharges();
  //  coordsPrinter.transferFromDevice();
  //  std::cout << "coords:" << coordsPrinter.getHostArray().data()[0].x
  //            << std::endl;
  //  coordsPrinter = context->getVelocityMass();
  //  coordsPrinter.transferFromDevice();
  //  std::cout << "velMass: " << coordsPrinter.getHostArray().data()[0].x
  //            << std::endl;
  //
  //  xyzqPrinter = context->getXYZQ()->get_xyz();
  //  std::cout << " xyzq: " << xyzqPrinter[0] << std::endl;
  //  coordsDelta.transferFromDevice();
  //  std::cout << "coordsDelta: " << coordsDelta.getHostArray().data()[0].x
  //            << std::endl;
  //  coordsDeltaPrevious.transferFromDevice();
  //  std::cout << "coordsDeltaPrevious: "
  //            << coordsDeltaPrevious.getHostArray().data()[0].x << std::endl;
  //  coordsRef.transferFromDevice();
  //  std::cout << "coordsRef: " << coordsRef.getHostArray().data()[0].x
  //            << std::endl;
  //  std::cout << "numdof: " << numDegreesOfFreedom << std::endl;
  //  std::cout << "numAtoms: " << numAtoms << std::endl;
  //  std::cout << "stride: " << stride << std::endl;
  //  std::cout << "kbt: " << kbt << std::endl;
  //  std::cout << "refke: " << referenceKineticEnergy << std::endl;
  //  std::cout << "volume : " << volume << std::endl;
  //
  if (m_UsingHolonomicConstraints) {
    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());

    prepareCoordsRefForHolonomicConstraintsKernel<<<numBlocks, numThreads, 0,
                                                    *m_IntegratorStream>>>(
        numAtoms, m_CoordsRef.getDeviceData(), coords,
        m_HalfStepCrystalFactor.getDeviceData());

    // Swapping coordsRef and coords as the holonomic constraint will update
    // the coords by calling it from the CharmmContext

    double4 *temporary = m_CoordsRef.getDeviceData();
    double4 *temporary2 = coords;
    coordsRefDevice = coords;
    coords = temporary;
    m_Context->getCoordinatesCharges().getDeviceArray().assignData(temporary);

    m_HolonomicConstraint->handleHolonomicConstraints(coordsRefDevice);

    m_CoordsRef.getDeviceArray().assignData(temporary);
    m_Context->getCoordinatesCharges().getDeviceArray().assignData(temporary2);

    coords = m_Context->getCoordinatesCharges().getDeviceData();
    coordsRefDevice = m_CoordsRef.getDeviceData();

    updateCoordsDeltaPredictedKernel<<<numBlocks, numThreads, 0,
                                       *m_IntegratorStream>>>(
        numAtoms, m_CoordsDeltaPredicted.getDeviceData(), coords,
        m_CoordsRef.getDeviceData());
  }

  copy_DtoD_async<double4>(m_CoordsDeltaPredicted.getDeviceData(),
                           coordsDeltaDevice, numAtoms, *m_IntegratorStream);

  onStepVelocityCalculation<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      numAtoms, m_TimeStep, coordsDeltaDevice, coordsDeltaPreviousDevice,
      velMass);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  // cudaCheck(cudaDeviceSynchronize()); // Error persists returning from here
  // return;                             // Error persists returning from here

  // Calculate nose hoover thermal piston velocity and position
  if (m_NoseHooverFlag) {

    // TODO : change this to on step kinetic energy
    /*double onStepKineticEnergy =
        (deltaPressureHalfStepKinetic[0] + deltaPressureHalfStepKinetic[2] +
         deltaPressureHalfStepKinetic[5]) /
        (0.5 * charmm::constants::patmos / volume);
    */
    double onStepKineticEnergy =
        m_Context->computeTemperature() *
        (0.5 * numDegreesOfFreedom * charmm::constants::kBoltz);
    m_NoseHooverPistonForce = 2.0 * m_TimeStep *
                              (onStepKineticEnergy - referenceKineticEnergy) /
                              m_NoseHooverPistonMass;

    m_NoseHooverPistonVelocity =
        m_NoseHooverPistonVelocityPrevious +
        (m_NoseHooverPistonForce + m_NoseHooverPistonForcePrevious) / 2.0;
  }

  // start : MovePressureCalculationToEndTesting
  m_KineticEnergyPressureTensor.setToValue(0.0);
  calculateAverageKineticPressureKernel<<<numBlocksReduction, numThreads, 0,
                                          *m_IntegratorStream>>>(
      numAtoms, m_TimeStep, coordsDeltaPreviousDevice, coordsDeltaDevice,
      velMass, m_KineticEnergyPressureTensor.getDeviceData());
  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  m_KineticEnergyPressureTensor.transferFromDevice();

  m_PressureTensor[0] =
      (virialTensor[0] + m_KineticEnergyPressureTensor[0]) * volumeFactor;
  m_PressureTensor[1] =
      (virialTensor[3] + m_KineticEnergyPressureTensor[1]) * volumeFactor;
  m_PressureTensor[2] =
      (virialTensor[4] + m_KineticEnergyPressureTensor[2]) * volumeFactor;
  m_PressureTensor[3] =
      (virialTensor[6] + m_KineticEnergyPressureTensor[3]) * volumeFactor;
  m_PressureTensor[4] =
      (virialTensor[7] + m_KineticEnergyPressureTensor[4]) * volumeFactor;
  m_PressureTensor[5] =
      (virialTensor[8] + m_KineticEnergyPressureTensor[5]) * volumeFactor;

  m_PressureScalar[0] =
      (m_PressureTensor[0] + m_PressureTensor[2] + m_PressureTensor[5]) / 3.0;
  m_PressureScalar.transferToDevice();

  m_AveragePressureScalar[0] =
      (m_StepId / (m_StepId + 1.0)) * m_AveragePressureScalar[0] +
      (1.0 / (m_StepId + 1.0)) * m_PressureScalar[0];
  for (int i = 0; i < 6; i++) {
    m_AveragePressureTensor[i] =
        (m_StepId / (m_StepId + 1.0)) * m_AveragePressureTensor[i] +
        (1.0 / (m_StepId + 1.0)) * m_PressureTensor[i];
  }

  if (m_DebugPrintFrequency > 0 && m_StepId % m_DebugPrintFrequency == 0) {
    for (int i = 0; i < 6; i++) {
      int j;
      switch (i) {
      case 0:
        j = 0;
        break;
      case 1:
        j = 3;
        break;
      case 2:
        j = 4;
        break;
      case 3:
        j = 6;
        break;
      case 4:
        j = 7;
        break;
      case 5:
        j = 8;
        break;
      default:
        break;
      }

      std::cout << "pressure[" << i << "] = " << m_PressureTensor[i] << "   "
                << virialTensor[j] << "   " << m_KineticEnergyPressureTensor[i]
                << std::endl;
    }

    std::cout << "averagePressureScalar = " << m_AveragePressureScalar[0]
              << std::endl;
  }

  // end :MovePressureCalculationToEndTesting
  // volume = context->getVolume();

  // *********************
  // Calculate HFCTE etc
  double pressureTarget = (m_ReferencePressure[0] + m_ReferencePressure[2] +
                           m_ReferencePressure[5]) /
                          3.0;
  double pistonPotentialEnergy =
      pressureTarget * volume * charmm::constants::atmosp;

  if (m_ConstantSurfaceTensionFlag) {
    pressureTarget = m_ReferencePressure[5];
    pistonPotentialEnergy =
        pressureTarget * volume * charmm::constants::atmosp -
        m_SurfaceTension * charmm::constants::surfaceTensionFactor * volume *
            charmm::constants::atmosp / boxDimensions[2];
  }

  // *********************
  if (m_NoseHooverFlag) {
    m_NoseHooverPistonPosition += m_NoseHooverPistonVelocity * m_TimeStep +
                                  0.5 * m_NoseHooverPistonForce * m_TimeStep;
  }
  updateSPKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      numAtoms, xyzq, coords);

  copy_DtoD_async<double4>(coordsDeltaDevice, coordsDeltaPreviousDevice,
                           numAtoms, *m_IntegratorStream);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  m_Context->setBoxDimensions(boxDimensions);
  surfaceTensionInstantaneous =
      0.5 * boxDimensions[2] *
      (m_PressureTensor[5] -
       0.5 * (m_PressureTensor[0] + m_PressureTensor[2])) /
      charmm::constants::surfaceTensionFactor;

  // HFCTE calculation calculated only in debug mode

  // if (debugPrintFrequency > 0 && stepId % debugPrintFrequency == 0) {
  double pistonKineticEnergy = 0.0;
  for (int i = 0; i < m_PistonDegreesOfFreedom; i++) {
    pistonKineticEnergy += 0.5 * m_OnStepPistonVelocity[i] *
                           m_OnStepPistonVelocity[i] * m_PistonMass[i];
  }
  m_Context->calculateKineticEnergy();
  auto ke = m_Context->getKineticEnergy();
  // exit if the kinetic energy is nan
  // if (ke != ke) {
  if (std::isnan(ke)) {
    throw std::runtime_error("NAN detected in kinetic energy");
    exit(1);
  }
  auto peContainer = m_Context->getPotentialEnergy();
  peContainer.transferFromDevice();
  auto pe = peContainer[0];

  m_DeltaPressureHalfStepKinetic.setToValue(0.0);
  calculateHalfStepKineticPressureKernel<<<numBlocks, numThreads, 0,
                                           *m_IntegratorStream>>>(
      numAtoms, vcell, m_CoordsDeltaPredicted.getDeviceData(), velMass,
      m_DeltaPressureHalfStepKinetic.getDeviceData());
  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  m_DeltaPressureHalfStepKinetic.transferFromDevice();
  double totken =
      (m_DeltaPressureHalfStepKinetic[0] + m_DeltaPressureHalfStepKinetic[2] +
       m_DeltaPressureHalfStepKinetic[5]) /
      (0.5 * charmm::constants::patmos / volume);
  // if (halfStepKineticEnergy1StepPrevious[0] == 0.0) {
  //   double epdiff;
  // }

  double hfcten =
      (m_HalfStepKineticEnergy1StepPrevious[0] - ke - pistonKineticEnergy +
       m_PotentialEnergyPrevious[0] - pe - pistonPotentialEnergy) /
          3.0 +
      (m_HalfStepKineticEnergy2StepsPrevious[0] - totken) / 12; // TODO fill it

  if (m_DebugPrintFrequency > 0 && m_StepId % m_DebugPrintFrequency == 0) {
    std::cout << "Kinetic energy = " << ke << std::endl;

    std::cout << "Potential energy = " << pe << std::endl;
    std::cout << "Piston potential energy = " << pistonPotentialEnergy
              << std::endl;
    std::cout << "Piston kinetic energy = " << pistonKineticEnergy << std::endl;
    std::cout << "Total energy = "
              << pe + ke + pistonPotentialEnergy + pistonKineticEnergy + hfcten
              << std::endl;

    std::cout << "HFCTE = " << hfcten << std::endl;

    std::cout << "Temperature : " << m_Context->computeTemperature() << "\n";
    std::cout << "Surface tension instantaneous : "
              << surfaceTensionInstantaneous << "\n";

    std::cout << "Volume : " << std::fixed << volume << "\n";
    std::cout << "Inv piston mass: ";
    for (int i = 0; i < m_PistonDegreesOfFreedom; i++) {
      std::cout << m_InversePistonMass[i] << " ";
    }
    std::cout << "\n";
  }
  m_HalfStepKineticEnergy2StepsPrevious[0] =
      m_HalfStepKineticEnergy1StepPrevious[0];
  m_HalfStepKineticEnergy1StepPrevious[0] =
      totken; // fill it current next half step ke
  m_PotentialEnergyPrevious[0] = pe + pistonPotentialEnergy;
  //}

  m_PotentialEnergyPrevious[0] = pe + pistonPotentialEnergy;
  //}

  m_StepId++;
}

// void CudaLangevinPistonIntegrator::setOnStepPistonVelocity(
//     const std::vector<double> _onStepPistonVelocity) {

//   assert((_onStepPistonVelocity.size() == pistonDegreesOfFreedom,
//           "Wrong size in setOnStepPistonVelocity"));

//   CudaContainer<double> temp;
//   temp.allocate(_onStepPistonVelocity.size());
//   temp.setHostArray(_onStepPistonVelocity);
//   temp.transferToDevice(); // technically not needed ? as calculation on
//   piston
//                            // dofs are done on host side
//   setOnStepPistonVelocity(temp);
// }

// void CudaLangevinPistonIntegrator::setHalfStepPistonVelocity(
//     const std::vector<double> _halfStepPistonVelocity) {

//   assert((_halfStepPistonVelocity.size() == pistonDegreesOfFreedom,
//           "Wrong size in setHalfStepPistonVelocity"));

//   CudaContainer<double> temp;
//   temp.allocate(_halfStepPistonVelocity.size());
//   temp.setHostArray(_halfStepPistonVelocity);
//   temp.transferToDevice(); // technically not needed ? as calculation on
//   piston
//                            // dofs are done on host side
//   setHalfStepPistonVelocity(temp);
// }

// void CudaLangevinPistonIntegrator::setOnStepPistonPosition(
//     const std::vector<double> _onStepPistonPosition) {

//   assert((_onStepPistonPosition.size() == pistonDegreesOfFreedom,
//           "Wrong size in setOnStepPistonPosition"));

//   CudaContainer<double> temp;
//   temp.allocate(_onStepPistonPosition.size());
//   temp.setHostArray(_onStepPistonPosition);
//   temp.transferToDevice(); // technically not needed ? as calculation on
//   piston
//                            // dofs are done on host side
//   setOnStepPistonPosition(temp);
// }

// void CudaLangevinPistonIntegrator::setHalfStepPistonPosition(
//     const std::vector<double> _halfStepPistonPosition) {

//   assert((_halfStepPistonPosition.size() == pistonDegreesOfFreedom,
//           "Wrong size in setHalfStepPistonPosition"));

//   CudaContainer<double> temp;
//   temp.allocate(_halfStepPistonPosition.size());
//   temp.setHostArray(_halfStepPistonPosition);
//   temp.transferToDevice(); // technically not needed ? as calculation on
//   piston
//                            // dofs are done on host side
//   setHalfStepPistonPosition(temp);
// }

std::map<std::string, std::string>
CudaLangevinPistonIntegrator::getIntegratorDescriptors(void) {
  std::map<std::string, std::string> ret;
  ret["type"] = "LangevinPiston";
  ret["timeStep"] = std::to_string(m_TimeStep);
  ret["bathTemperature"] = std::to_string(m_BathTemperature);
  ret["referencePressure"] = std::to_string(m_ReferencePressure[0]);
  ret["pistonMass"] = std::to_string(m_PistonMass[0]);
  ret["noseHooverPistonMass"] = std::to_string(m_NoseHooverPistonMass);
  return ret;
}

double CudaLangevinPistonIntegrator::computeNoseHooverPistonMass(void) {
  CudaContainer<double4> velmassCC = m_Context->getVelocityMass();
  velmassCC.transferFromDevice();
  std::vector<double4> velmass = velmassCC.getHostArray();
  double totalMass = 0.0;
  for (std::size_t i = 0; i < velmass.size(); i++)
    totalMass += 1. / velmass[i].w;
  return totalMass / 50.0;
}

void CudaLangevinPistonIntegrator::allocatePistonVariables(void) {
  m_OnStepPistonPosition.resize(m_PistonDegreesOfFreedom);
  m_OnStepPistonPosition.setToValue(0.0);
  m_HalfStepPistonPosition.resize(m_PistonDegreesOfFreedom);
  m_HalfStepPistonPosition.setToValue(0.0);
  m_OnStepPistonVelocity.resize(m_PistonDegreesOfFreedom);
  m_OnStepPistonVelocity.setToValue(0.0);
  m_HalfStepPistonVelocity.resize(m_PistonDegreesOfFreedom);
  m_HalfStepPistonVelocity.setToValue(0.0);

  m_PistonMass.resize(m_PistonDegreesOfFreedom);
  m_PistonMass.setToValue(0.0);
  m_InversePistonMass.resize(m_PistonDegreesOfFreedom);
  m_InversePistonMass.setToValue(0.0);
  m_PistonDeltaPressure.resize(m_PistonDegreesOfFreedom);
  m_PistonDeltaPressure.setToValue(0.0);
  m_PressurePistonPositionDelta.resize(m_PistonDegreesOfFreedom);
  m_PressurePistonPositionDelta.setToValue(0.0);
  m_PressurePistonPositionDeltaPrevious.resize(m_PistonDegreesOfFreedom);
  m_PressurePistonPositionDeltaPrevious.setToValue(0.0);
  m_PressurePistonPositionDeltaStored.resize(m_PistonDegreesOfFreedom);
  m_PressurePistonPositionDeltaStored.setToValue(0.0);

  return;
}

void CudaLangevinPistonIntegrator::removeCenterOfMassMotion(void) {
  auto pbc = m_Context->getForceManager()->getPeriodicBoundaryCondition();

  // TODO : do this in the kernel rather than the host side

  int numAtoms = m_Context->getNumAtoms();
  auto velocityMass = m_Context->getVelocityMass();
  auto coords = m_Context->getCoordinatesCharges();
  auto boxDimensions = m_Context->getBoxDimensions();
  coords.transferFromDevice();
  m_CoordsDeltaPrevious.transferFromDevice();

  // Remove the center of mass velocity
  float3 cdpcom = make_float3(0.0, 0.0, 0.0);

  float totalMass = 0.0;
  for (int i = 0; i < numAtoms; i++) {
    auto mass = 1 / velocityMass[i].w;

    cdpcom.x += m_CoordsDeltaPrevious[i].x * mass;

    if (pbc == PBC::P21) {
      // cdpcom.y -= coordsDeltaPrevious[i].y * mass;
      // cdpcom.z -= coordsDeltaPrevious[i].z * mass;
    } else {
      cdpcom.y += m_CoordsDeltaPrevious[i].y * mass;
      cdpcom.z += m_CoordsDeltaPrevious[i].z * mass;
    }

    totalMass += mass;
  }
  cdpcom.x /= totalMass;
  cdpcom.y /= totalMass;
  cdpcom.z /= totalMass;

  for (int i = 0; i < numAtoms; i++) {
    m_CoordsDeltaPrevious[i].x -= cdpcom.x;

    if (pbc == PBC::P21) {
      // coordsDeltaPrevious[i].y += cdpcom.y;
      // coordsDeltaPrevious[i].z += cdpcom.z;
    } else {
      m_CoordsDeltaPrevious[i].y -= cdpcom.y;
      m_CoordsDeltaPrevious[i].z -= cdpcom.z;
    }
  }

  m_CoordsDeltaPrevious.transferToDevice();

  return;
}

__global__ void averageNetForceKernel(int numAtoms, int stride,
                                      const double *__restrict__ force) {
  return;
}

void CudaLangevinPistonIntegrator::removeCenterOfMassAverageNetForce(void) {
  int numAtoms = m_Context->getNumAtoms();
  auto force = m_Context->getForces();

  int stride = m_Context->getForceStride();

  int numThreads = 128;
  int numBlocks = (numAtoms + numThreads - 1) / numThreads;
  averageNetForceKernel<<<numBlocks, numThreads>>>(numAtoms, stride,
                                                   force->xyz());

  return;
}
