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
#include <stdexcept>

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

  // 0 -> New "Jung" T
  // 1 -> Old T
  m_AverageWindowSize = 0;
  m_KineticEnergy.resize(2);
  m_AverageTemperature.resize(2);

  m_KineticEnergy.setToValue(0.0);
  m_AverageTemperature.setToValue(0.0);

  m_UsingOldTemperature = false;
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
    const bool usingOldTemperature) {
  m_UsingOldTemperature = usingOldTemperature;
  return;
}

void CudaNoseHooverThermostatIntegrator::resetAverageTemperature(void) {
  m_AverageWindowSize = 0;
  m_AverageTemperature.setToValue(0.0);
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

double CudaNoseHooverThermostatIntegrator::getInstantaneousTemperature(void) {
  const double ndegf = static_cast<double>(m_Context->getDegreesOfFreedom());
  m_KineticEnergy.transferToHost();
  if (m_UsingOldTemperature)
    return (m_KineticEnergy[1] / (0.5 * ndegf * charmm::constants::kBoltz));
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
  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

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

void CudaNoseHooverThermostatIntegrator::initializeFromRestartFile(
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
  // std::vector<double> HDOT(6, 0.0);
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

  // Not needed for Nose-Hoover thermostat
  line.clear();
  std::getline(fin, line);
  // HDOT[0] = apo::fortSciStrToCDouble(line.substr(0, 22));
  // HDOT[1] = apo::fortSciStrToCDouble(line.substr(22, 22));
  // HDOT[2] = apo::fortSciStrToCDouble(line.substr(44, 22));
  line.clear();
  std::getline(fin, line);
  // HDOT[3] = apo::fortSciStrToCDouble(line.substr(0, 22));
  // HDOT[4] = apo::fortSciStrToCDouble(line.substr(22, 22));
  // HDOT[5] = apo::fortSciStrToCDouble(line.substr(44, 22));

  line.clear();
  std::getline(fin, line);
  // PNH = apo::fortSciStrToCDouble(line.substr(0, 22)); // Not needed for NHT
  PNHV = apo::fortSciStrToCDouble(line.substr(22, 22));
  PNHF = apo::fortSciStrToCDouble(line.substr(44, 22));

  // Not needed for Nose-Hoover thermostat
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

  // Not needed for Nose-Hoover thermostat
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

  // // Not needed for Nose-Hoover thermostat
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

  // Not needed for Nose-Hoover thermostat
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

  // Not needed for Nose-Hoover thermostat
  line.clear();
  std::getline(fin, line);
  // GRAD1A = apo::fortSciStrToCDouble(line.substr(0, 22));
  // GRAD1B = apo::fortSciStrToCDouble(line.substr(22, 22));
  // GRAD2A = apo::fortSciStrToCDouble(line.substr(44, 22));
  line.clear();
  std::getline(fin, line);
  // GRAD2B = apo::fortSciStrToCDouble(line.substr(0, 22));

  m_Context->setBoxDimensions({XTLABC[0], XTLABC[2], XTLABC[5]});

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
  // int NSAVC = 0;  // Not needed for Nose-Hoover Thermostat
  // int NSAVV = 0;  // Not needed for Nose-Hoover Thermostat
  // int JHSTRT = 0; // Not needed for Nose-Hoover Thermostat
  int NDEGF = 0;
  // std::uint64_t SEED = 0;    // Not needed for Nose-Hoover Thermostat
  // std::string RNGSTATE = ""; // Not needed for Nose-Hoover Thermostat

  line.clear();
  std::getline(fin, line);
  NATOM = std::stoi(line.substr(0, 12));
  NPRIV = std::stoull(line.substr(12, 12));
  NSTEP = std::stoi(line.substr(24, 12));
  // NSAVC = std::stoi(line.substr(36, 12));
  // NSAVV = std::stoi(line.substr(48, 12));
  // JHSTRT = std::stoi(line.substr(60, 12));
  NDEGF = std::stoi(line.substr(72, 12));
  // SEED = std::stoull(line.substr(84, 22));
  // RNGSTATE = line.substr(106, std::string::npos);

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

    // Actually store the change in velocity (not the force)
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
    const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double fact = 0.5 / timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
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

  if (m_StepsSinceNeighborListUpdate % m_NonbondedListUpdateFrequency == 0) {
    if (m_Context->getForceManager()->getPeriodicBoundaryCondition() ==
        PBC::P21) {
      // Find a better place for this
      int numGroups =
          m_Context->getForceManager()->getPsf()->getGroups().size();
      int2 *groups =
          m_Context->getForceManager()->getPsf()->getGroups().getDeviceData();
      float boxDimX = static_cast<float>(m_Context->getBoxDimensions()[0]);

      constexpr int numThreads = 256;
      const int numBlocks = (numGroups + numThreads - 1) / numThreads;
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

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  if ((m_DebugPrintFrequency > 0) &&
      (m_CurrentPropagatedStep % m_DebugPrintFrequency == 0)) {
    m_Context->calculateForces(false, true, true);
  } else {
    m_Context->calculateForces(false, false, false);
  }

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

  const PBC pbc = m_Context->getForceManager()->getPeriodicBoundaryCondition();
  const int numAtoms = m_Context->getNumAtoms();
  CudaContainer<double4> &velMass = m_Context->getVelocityMass();

  velMass.transferToHost();
  m_CoordsDeltaPrevious.transferToHost();

  double3 com = make_double3(0.0, 0.0, 0.0);
  double totalMass = 0.0;
  for (int i = 0; i < numAtoms; i++) {
    const double mass = 1.0 / velMass[i].w;

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
