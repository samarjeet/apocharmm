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
#include "CudaLangevinThermostatIntegrator.h"
#include "gpu_utils.h"
#include <iostream>
#include <random>
#include <stdexcept>

CudaLangevinThermostatIntegrator::CudaLangevinThermostatIntegrator(
    const double timeStep)
    : CudaIntegrator(timeStep) {
  m_UsingHolonomicConstraints = true;

  m_ReferenceTemperature = 300.0;
  this->setThermostatFriction(0.0);

  std::random_device rd{};
  m_Seed = rd();
  m_RngSequencePos = 0;
  m_RngStates = nullptr;

  // 0 -> New "Jung" T
  // 1 -> Old T
  m_AverageWindowSize = 0;
  m_KineticEnergy.resize(2);
  m_AverageTemperature.resize(2);
}

CudaLangevinThermostatIntegrator::~CudaLangevinThermostatIntegrator(void) {
  this->dealloc();
}

void CudaLangevinThermostatIntegrator::setReferenceTemperature(
    const double referenceTemperature) {
  m_ReferenceTemperature = referenceTemperature;
  return;
}

void CudaLangevinThermostatIntegrator::setThermostatFriction(
    const double thermostatFriction) {
  m_ThermostatFriction = thermostatFriction;
  m_ThermostatGamma = m_TimeStep * m_Timfac * thermostatFriction;
  // JEG260409: If the friction changes before running a new simulation (i.e.
  // not from a restart file or continuing a previous one) AND the CHARMM
  // context has previous been set, we need to call this->initialize() again so
  // m_CoordsDelta and m_CoordsDeltaPrevious can have correct values.
  if ((m_Context != nullptr) && (m_TotNumSteps == 0))
    this->initialize();
  return;
}

void CudaLangevinThermostatIntegrator::setThermostatRngSeed(
    const std::uint64_t seed) {
  m_Seed = seed;
  this->initializeRng();
  return;
}

void CudaLangevinThermostatIntegrator::setRngSequencePos(
    const unsigned long long int rngSequencePos) {
  m_RngSequencePos = rngSequencePos;
  this->initializeRng();
  return;
}

void CudaLangevinThermostatIntegrator::resetAverageTemperature(void) {
  m_AverageWindowSize = 0;
  m_AverageTemperature.setToValue(0.0);
  return;
}

double CudaLangevinThermostatIntegrator::getReferenceTemperature(void) const {
  return m_ReferenceTemperature;
}

double CudaLangevinThermostatIntegrator::getThermostatFriction(void) const {
  return m_ThermostatFriction;
}

std::uint64_t
CudaLangevinThermostatIntegrator::getThermostatRngSeed(void) const {
  return m_Seed;
}

unsigned long long int
CudaLangevinThermostatIntegrator::getRngSequencePos(void) const {
  return m_RngSequencePos;
}

int CudaLangevinThermostatIntegrator::getAverageWindowSize(void) const {
  return m_AverageWindowSize;
}

const CudaContainer<double> &
CudaLangevinThermostatIntegrator::getKineticEnergy(void) const {
  return m_KineticEnergy;
}

const CudaContainer<double> &
CudaLangevinThermostatIntegrator::getAverageTemperature(void) const {
  return m_AverageTemperature;
}

CudaContainer<double> &
CudaLangevinThermostatIntegrator::getKineticEnergy(void) {
  return m_KineticEnergy;
}

CudaContainer<double> &
CudaLangevinThermostatIntegrator::getAverageTemperature(void) {
  return m_AverageTemperature;
}

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
                     const double gamma, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;
  const double halfGamma = 0.5 * gamma;
  const double alpha0 = timeStep * sqrt(1.0 + halfGamma);
  const double alpha1 = timeStep * (1.0 - halfGamma) / sqrt(1.0 + halfGamma);

  for (int i = idx; i < numAtoms; i += stride) {
    const double fx = forces[0 * forceStride + i];
    const double fy = forces[1 * forceStride + i];
    const double fz = forces[2 * forceStride + i];
    const double fact = 0.5 * timeStep2 * velMass[i].w;

    coordsDelta[i].x = alpha0 * velMass[i].x - fact * fx;
    coordsDelta[i].y = alpha0 * velMass[i].y - fact * fy;
    coordsDelta[i].z = alpha0 * velMass[i].z - fact * fz;

    coordsDeltaPrevious[i].x = alpha1 * velMass[i].x + fact * fx;
    coordsDeltaPrevious[i].y = alpha1 * velMass[i].y + fact * fy;
    coordsDeltaPrevious[i].z = alpha1 * velMass[i].z + fact * fz;
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

void CudaLangevinThermostatIntegrator::initialize(void) {
  const int numAtoms = m_Context->getNumAtoms();
  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

  this->initializeRng();

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
      velMass, numAtoms, forces, forceStride, m_ThermostatGamma, m_TimeStep);

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

void CudaLangevinThermostatIntegrator::initializeFromRestartFile(
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
  bool isApoRstFile = false;
  line.clear();
  std::getline(fin, line);
  if (line.length() >= 33) // Check for APO flag
    isApoRstFile = (line.substr(30, 3) == "APO");

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
  // double PNHV = 0.0, PNHF = 0.0;
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

  // Not needed for Langevin Thermostat
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
  // PNH = apo::fortSciStrToCDouble(line.substr(0, 22));   // Not needed for LT
  // PNHV = apo::fortSciStrToCDouble(line.substr(22, 22)); // Not needed for LT
  // PNHF = apo::fortSciStrToCDouble(line.substr(44, 22)); // Not needed for LT

  // Not needed for Langevin Thermostat
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

  // Not needed for Langevin Thermostat
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

  // Not needed for Langevin Thermostat
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

  // Not needed for Langevin Thermostat
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

  // Not needed for Langevin Thermostat
  line.clear();
  std::getline(fin, line);
  // GRAD1A = apo::fortSciStrToCDouble(line.substr(0, 22));
  // GRAD1B = apo::fortSciStrToCDouble(line.substr(22, 22));
  // GRAD2A = apo::fortSciStrToCDouble(line.substr(44, 22));
  line.clear();
  std::getline(fin, line);
  // GRAD2B = apo::fortSciStrToCDouble(line.substr(0, 22));

  m_Context->setBoxDimensions({XTLABC[0], XTLABC[2], XTLABC[5]});

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
  // int NSAVC = 0;  // Not needed for Langevin Thermostat
  // int NSAVV = 0;  // Not needed for Langevin Thermostat
  // int JHSTRT = 0; // Not needed for Langevin Thermostat
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
  this->setThermostatRngSeed(SEED);
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

__global__ static void PositionUpdateKernel(
    double4 *__restrict__ coordsCharges, double4 *__restrict__ coordsDelta,
    curandStatePhilox4_32_10_t *__restrict__ rngStates,
    const double4 *__restrict__ coordsDeltaPrevious,
    const double4 *__restrict__ velMass, const int numAtoms,
    const double *__restrict__ forces, const int forceStride,
    const double gamma, const double kbt, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;
  const double halfGamma = 0.5 * gamma;
  const double denomGamma = 1.0 / (1.0 + halfGamma);
  const double alpha = (1.0 - halfGamma) * denomGamma;

  for (int i = idx; i < numAtoms; i += stride) {
    curandStatePhilox4_32_10_t state = rngStates[i];
    const double rdf = sqrt(2.0 * gamma * kbt / velMass[i].w) / timeStep;
    const float4 rn = curand_normal4(&state);
    const double fact = timeStep2 * velMass[i].w * denomGamma;
    double fx = forces[0 * forceStride + i];
    double fy = forces[1 * forceStride + i];
    double fz = forces[2 * forceStride + i];
    fx += rdf * static_cast<double>(rn.x);
    fy += rdf * static_cast<double>(rn.y);
    fz += rdf * static_cast<double>(rn.z);

    coordsDelta[i].x = alpha * coordsDeltaPrevious[i].x - fact * fx;
    coordsDelta[i].y = alpha * coordsDeltaPrevious[i].y - fact * fy;
    coordsDelta[i].z = alpha * coordsDeltaPrevious[i].z - fact * fz;

    coordsCharges[i].x += coordsDelta[i].x;
    coordsCharges[i].y += coordsDelta[i].y;
    coordsCharges[i].z += coordsDelta[i].z;

    rngStates[i] = state;
  }

  return;
}

__global__ static void VelocityUpdateKernel(
    double4 *__restrict__ coordsDelta, double4 *__restrict__ velMass,
    const double4 *__restrict__ coordsCharges,
    const double4 *__restrict__ coordsRef,
    const double4 *__restrict__ coordsDeltaPrevious, const int numAtoms,
    const double gamma, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double fact = 0.5 * sqrt(1.0 + 0.5 * gamma) / timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    coordsDelta[i].x = coordsCharges[i].x - coordsRef[i].x;
    coordsDelta[i].y = coordsCharges[i].y - coordsRef[i].y;
    coordsDelta[i].z = coordsCharges[i].z - coordsRef[i].z;

    velMass[i].x = fact * (coordsDelta[i].x + coordsDeltaPrevious[i].x);
    velMass[i].y = fact * (coordsDelta[i].y + coordsDeltaPrevious[i].y);
    velMass[i].z = fact * (coordsDelta[i].z + coordsDeltaPrevious[i].z);
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

void CudaLangevinThermostatIntegrator::propagateOneStep(void) {
  const double kbt = charmm::constants::kBoltz * m_ReferenceTemperature;
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
          m_Context->getForceManager()->getPSF()->getGroups().size();
      int2 *groups =
          m_Context->getForceManager()->getPSF()->getGroups().getDeviceData();
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

  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

  PositionUpdateKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      coordsCharges, m_CoordsDelta.getDeviceData(), m_RngStates,
      m_CoordsDeltaPrevious.getDeviceData(), velMass, numAtoms, forces,
      forceStride, m_ThermostatGamma, kbt, m_TimeStep);

  m_RngSequencePos += 4; // curand_normal4 iterates 4 steps

  if (m_UsingHolonomicConstraints) {
    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());
  }

  VelocityUpdateKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      m_CoordsDelta.getDeviceData(), velMass, coordsCharges,
      m_CoordsRef.getDeviceData(), m_CoordsDeltaPrevious.getDeviceData(),
      numAtoms, m_ThermostatGamma, m_TimeStep);

  cudaCheck(
      cudaMemsetAsync(static_cast<void *>(m_KineticEnergy.getDeviceData()), 0,
                      2 * sizeof(double), *m_IntegratorStream));

  ComputeKineticEnergyKernel<<<1, 1024, 0, *m_IntegratorStream>>>(
      m_KineticEnergy.getDeviceData(), velMass, m_CoordsDelta.getDeviceData(),
      m_CoordsDeltaPrevious.getDeviceData(), numAtoms, m_TimeStep);

  UpdateAverageTemperatureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
      m_AverageTemperature.getDeviceData(), m_KineticEnergy.getDeviceData(),
      m_Context->getDegreesOfFreedom(), charmm::constants::kBoltz,
      m_AverageWindowSize);

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

void CudaLangevinThermostatIntegrator::initializeRng(void) {
  if (m_Context == nullptr)
    return;

  const int numAtoms = m_Context->getNumAtoms();
  this->alloc(numAtoms);
  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;
  InitializeRngKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      m_RngStates, numAtoms, m_Seed, 0, m_RngSequencePos);

  return;
}

void CudaLangevinThermostatIntegrator::removeCenterOfMassMotion(void) {
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

void CudaLangevinThermostatIntegrator::alloc(const int n) {
  if (m_RngStates != nullptr) // Deallocate to be safe
    this->dealloc();

  // Allocate memory for RNG
  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&m_RngStates),
                       n * sizeof(curandStatePhilox4_32_10_t)));

  return;
}

void CudaLangevinThermostatIntegrator::dealloc(void) {
  if (m_RngStates != nullptr) {
    // Deallocate memory for RNG
    cudaCheck(cudaFree(static_cast<void *>(m_RngStates)));
    m_RngStates = nullptr;
  }
  return;
}
