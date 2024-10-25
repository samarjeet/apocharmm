// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "Constants.h"
#include "CudaBAOABIntegrator.h"
#include "gpu_utils.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>

CudaBAOABIntegrator::CudaBAOABIntegrator(const double timeStep)
    : CudaIntegrator(timeStep) {
  m_Friction = 0.0;
  m_DevPHILOXStates = nullptr;
  m_StepsSinceLastReport = 0;
  m_BathTemperature = 0.0;
  m_IntegratorTypeName = "BAOAB";
}

CudaBAOABIntegrator::CudaBAOABIntegrator(const double timeStep,
                                         const double bathTemperature,
                                         const double friction)
    : CudaBAOABIntegrator(timeStep) {
  m_Friction = friction;
  m_BathTemperature = bathTemperature;
}

CudaBAOABIntegrator::~CudaBAOABIntegrator(void) {
  cudaCheck(cudaFree(m_DevPHILOXStates));
}

__global__ static void setup_kernel(int numAtoms,
                                    curandStatePhilox4_32_10_t *state) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms)
    curand_init(1234, index, 0, &state[index]);
}

__global__ static void init(double kbt, const double gamma, const int numAtoms,
                            const int stride, const double timeStep,
                            // double4 *__restrict__ coords,
                            double4 *__restrict__ coordsDelta,
                            double4 *__restrict__ coordsDeltaPrevious,
                            const double4 *__restrict__ velMass,
                            const double *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    double fx = force[index];
    double fy = force[index + stride];
    double fz = force[index + 2 * stride];

    double fact = timeStep * timeStep * velMass[index].w * 0.5;
    double alpha = timeStep * (1 - 0.5 * gamma) / sqrt(1 + gamma * 0.5);

    coordsDeltaPrevious[index].x = velMass[index].x * alpha - fx * fact;
    coordsDeltaPrevious[index].y = velMass[index].y * alpha - fy * fact;
    coordsDeltaPrevious[index].z = velMass[index].z * alpha - fz * fact;

    alpha = timeStep * sqrt(1 + 0.5 * gamma);

    coordsDelta[index].x = alpha * velMass[index].x + fact * fx;
    coordsDelta[index].y = alpha * velMass[index].y + fact * fy;
    coordsDelta[index].z = alpha * velMass[index].z + fact * fz;
  }
}

void CudaBAOABIntegrator::initialize(void) {
  int numAtoms = m_Context->getNumAtoms();

  m_CoordsDelta.resize(numAtoms);
  m_CoordsDeltaPrevious.resize(numAtoms);

  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&m_DevPHILOXStates),
                       numAtoms * sizeof(curandStatePhilox4_32_10_t)));

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;
  setup_kernel<<<numBlocks, numThreads>>>(numAtoms, m_DevPHILOXStates);
  cudaCheck(cudaDeviceSynchronize());

  auto coords = m_Context->getCoordinatesCharges().getDeviceData();
  auto coordsDeltaDevice = m_CoordsDelta.getDeviceData();
  auto coordsDeltaPreviousDevice = m_CoordsDeltaPrevious.getDeviceData();
  auto velMass = m_Context->getVelocityMass().getDeviceData();

  m_Context->calculateForces(); // false, false, false);
  auto force = m_Context->getForces();

  int stride = m_Context->getForceStride();
  double kbt = charmm::constants::kBoltz * m_BathTemperature;
  const double gamma = m_TimeStep * m_Timfac * m_Friction;

  init<<<numBlocks, numThreads>>>(
      kbt, gamma, numAtoms, stride, m_TimeStep, // coords,
      coordsDeltaDevice, coordsDeltaPreviousDevice, velMass, force->xyz());
  cudaCheck(cudaDeviceSynchronize());

  return;
}

void CudaBAOABIntegrator::setFriction(const double friction) {
  m_Friction = friction;
  return;
}

double CudaBAOABIntegrator::getFriction(void) const { return m_Friction; }

void CudaBAOABIntegrator::setBathTemperature(const double bathTemperature) {
  m_BathTemperature = bathTemperature;
  return;
}

double CudaBAOABIntegrator::getBathTemperature(void) const {
  return m_BathTemperature;
}

__global__ static void step1(double kbt, const double gamma, const int numAtoms,
                             const int stride, const double timeStep,
                             double4 *__restrict__ coords,
                             double4 *__restrict__ coordsDelta,
                             const double4 *__restrict__ coordsDeltaPrevious,
                             double4 *__restrict__ velMass,
                             const double *__restrict__ force,
                             curandStatePhilox4_32_10_t *state) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    curandStatePhilox4_32_10_t localState = state[index];
    float4 random = curand_normal4(&localState);

    double fx = force[index];
    double fy = force[index + stride];
    double fz = force[index + 2 * stride];

    double rdf = sqrt(2.0 * gamma * kbt / velMass[index].w) / timeStep;
    fx += random.x * rdf;
    fy += random.y * rdf;
    fz += random.z * rdf;

    double alpha = (1 - 0.5 * gamma) / (1 + 0.5 * gamma);
    double fact = timeStep * timeStep * velMass[index].w / (1 + gamma * 0.5);

    coordsDelta[index].x = alpha * coordsDeltaPrevious[index].x - fact * fx;
    coordsDelta[index].y = alpha * coordsDeltaPrevious[index].y - fact * fy;
    coordsDelta[index].z = alpha * coordsDeltaPrevious[index].z - fact * fz;

    coords[index].x += coordsDelta[index].x;
    coords[index].y += coordsDelta[index].y;
    coords[index].z += coordsDelta[index].z;

    state[index] = localState;
  }
}

__global__ static void
step2(double kbt, const double gamma, const int numAtoms, const int stride,
      const double timeStep, const double4 *__restrict__ coordsRef,
      const double4 *__restrict__ coords, double4 *__restrict__ coordsDelta,
      double4 *__restrict__ coordsDeltaPrevious, double4 *__restrict__ velMass,
      const double *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {

    coordsDelta[index].x = coords[index].x - coordsRef[index].x;
    coordsDelta[index].y = coords[index].y - coordsRef[index].y;
    coordsDelta[index].z = coords[index].z - coordsRef[index].z;

    double fact = 0.5 * sqrt(1 + gamma * 0.5) / timeStep;

    velMass[index].x =
        (coordsDelta[index].x + coordsDeltaPrevious[index].x) * fact;
    velMass[index].y =
        (coordsDelta[index].y + coordsDeltaPrevious[index].y) * fact;
    velMass[index].z =
        (coordsDelta[index].z + coordsDeltaPrevious[index].z) * fact;

    coordsDeltaPrevious[index].x = coordsDelta[index].x;
    coordsDeltaPrevious[index].y = coordsDelta[index].y;
    coordsDeltaPrevious[index].z = coordsDelta[index].z;
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

// TODO : do this in the imageCentering kernel in CharmmContext itself
static __global__ void invertDeltaAsymmetric(int numGroups,
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

void CudaBAOABIntegrator::propagateOneStep(void) {
  auto coords = m_Context->getCoordinatesCharges().getDeviceData();
  auto xyzq = m_Context->getXYZQ()->getDeviceXYZQ();
  auto coordsDeltaDevice = m_CoordsDelta.getDeviceData();
  auto coordsDeltaPreviousDevice = m_CoordsDeltaPrevious.getDeviceData();
  auto velMass = m_Context->getVelocityMass().getDeviceData();

  int numAtoms = m_Context->getNumAtoms();
  int stride = m_Context->getForceStride();

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

  if (m_DebugPrintFrequency > 0 &&
      (m_StepsSinceLastReport + 1) % m_DebugPrintFrequency == 0) {
    m_Context->calculateForces(false, true, false);
  } else {
    m_Context->calculateForces(false, false, false);
  }

  auto force = m_Context->getForces();

  double kbt = charmm::constants::kBoltz * m_BathTemperature;
  const double gamma = m_TimeStep * m_Timfac * m_Friction;

  if (m_UsingHolonomicConstraints) {
    copy_DtoD_async<double4>(coords, m_CoordsRef.getDeviceData(), numAtoms,
                             *m_IntegratorMemcpyStream);
  }

  int numThreads = 512;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  step1<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      kbt, gamma, numAtoms, stride, m_TimeStep, coords, coordsDeltaDevice,
      coordsDeltaPreviousDevice, velMass, force->xyz(), m_DevPHILOXStates);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorMemcpyStream));
  if (m_UsingHolonomicConstraints) {
    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceData());
  }

  step2<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      kbt, gamma, numAtoms, stride, m_TimeStep, m_CoordsRef.getDeviceData(),
      coords, coordsDeltaDevice, coordsDeltaPreviousDevice, velMass,
      force->xyz());

  updateSPKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
      numAtoms, xyzq, coords);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  m_StepsSinceLastReport++;
  if (m_DebugPrintFrequency > 0 &&
      m_StepsSinceLastReport % m_DebugPrintFrequency == 0) {
    m_Context->calculateKineticEnergy();
    auto ke = m_Context->getKineticEnergy();
    std::cout << "kinetic energy = " << ke << std::endl;

    // m_Context->getForceManager()->setPrintEnergyDecomposition(true);
    // m_Context->calculateForces(false, true, false);
    // m_Context->getForceManager()->setPrintEnergyDecomposition(false);

    auto peContainer = m_Context->getPotentialEnergy();
    peContainer.transferFromDevice();
    auto pe = peContainer[0];
    std::cout << "potential energy = " << pe << std::endl;
    std::cout << "total energy = " << pe + ke << std::endl;

    std::cout << "[LangTherm]Temp : " << m_Context->computeTemperature()
              << "\n";
    m_StepsSinceLastReport = 0;
  }

  return;
}

std::map<std::string, std::string>
CudaBAOABIntegrator::getIntegratorDescriptors(void) {
  std::map<std::string, std::string> ret;
  ret["type"] = "BAOAB";
  ret["timeStep"] = std::to_string(m_TimeStep);
  ret["bathTemperature"] = std::to_string(m_BathTemperature);
  ret["friction"] = std::to_string(m_Friction);
  return ret;
}

const CudaContainer<double4> &
CudaBAOABIntegrator::getCoordsDeltaPrevious(void) const {
  return m_CoordsDeltaPrevious;
}

CudaContainer<double4> &CudaBAOABIntegrator::getCoordsDeltaPrevious(void) {
  return m_CoordsDeltaPrevious;
}

void CudaBAOABIntegrator::setCoordsDeltaPrevious(
    const std::vector<std::vector<double>> &coordsDeltaPrevious) {
  assert((coordsDeltaPrevious.size() == m_Context->getNumAtoms(),
          "Wrong size in setCoordsDeltaPrevious"));
  std::vector<double4> cdpd4;
  for (std::size_t i = 0; i < coordsDeltaPrevious.size(); i++) {
    cdpd4.emplace_back(make_double4(coordsDeltaPrevious[i][0],
                                    coordsDeltaPrevious[i][1],
                                    coordsDeltaPrevious[i][2], 0.0));
  }
  m_CoordsDeltaPrevious = cdpd4;
  // coordsDeltaPrevious.setHostArray(cdpCC);
  // coordsDeltaPrevious.transferToDevice();

  return;
}
