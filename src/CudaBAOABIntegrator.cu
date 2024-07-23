// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "Constants.h"
#include "CudaBAOABIntegrator.h"
#include "gpu_utils.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>

CudaBAOABIntegrator::CudaBAOABIntegrator(ts_t timeStep)
    : CudaIntegrator(timeStep) {
  friction = 0.0;
  stepsSinceLastReport = 0;
}

CudaBAOABIntegrator::CudaBAOABIntegrator(ts_t timeStep, double bathTemperature,
                                         double friction)
    : CudaIntegrator(timeStep) {
  setFriction(friction);
  setBathTemperature(bathTemperature);
  stepsSinceLastReport = 0;
}

CudaBAOABIntegrator::~CudaBAOABIntegrator() {
  // cudaCheck(cudaFree(devPHILOXStates));
}

__global__ static void setup_kernel(int numAtoms,
                                    curandStatePhilox4_32_10_t *state) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms)
    curand_init(1234, index, 0, &state[index]);
}

__global__ static void init(double kbt, const double gamma, const int numAtoms,
                            const int stride, const ts_t timeStep,
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

void CudaBAOABIntegrator::initialize() {
  int numAtoms = context->getNumAtoms();

  coordsDelta.allocate(numAtoms);
  coordsDeltaPrevious.allocate(numAtoms);

  cudaCheck(cudaMalloc((void **)&devPHILOXStates,
                       numAtoms * sizeof(curandStatePhilox4_32_10_t)));

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;
  setup_kernel<<<numBlocks, numThreads>>>(numAtoms, devPHILOXStates);
  cudaCheck(cudaDeviceSynchronize());

  auto coords = context->getCoordinatesCharges().getDeviceArray().data();
  auto coordsDeltaDevice = coordsDelta.getDeviceArray().data();
  auto coordsDeltaPreviousDevice = coordsDeltaPrevious.getDeviceArray().data();

  auto velMass = context->getVelocityMass().getDeviceArray().data();

  context->calculateForces(); // false, false, false);
  auto force = context->getForces();

  int stride = context->getForceStride();
  double kbt = charmm::constants::kBoltz * bathTemperature;
  const double gamma = timeStep * timfac * friction;

  init<<<numBlocks, numThreads>>>(
      kbt, gamma, numAtoms, stride, timeStep, // coords,
      coordsDeltaDevice, coordsDeltaPreviousDevice, velMass, force->xyz());
  cudaCheck(cudaDeviceSynchronize());
}

__global__ static void step1(double kbt, const double gamma, const int numAtoms,
                             const int stride, const ts_t timeStep,
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
      const ts_t timeStep, const double4 *__restrict__ coordsRef,
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

void CudaBAOABIntegrator::propagateOneStep() {

  auto coords = context->getCoordinatesCharges().getDeviceArray().data();

  auto xyzq = context->getXYZQ()->getDeviceXYZQ();
  auto coordsDeltaDevice = coordsDelta.getDeviceArray().data();
  auto coordsDeltaPreviousDevice = coordsDeltaPrevious.getDeviceArray().data();

  auto velMass = context->getVelocityMass().getDeviceArray().data();

  int numAtoms = context->getNumAtoms();
  int stride = context->getForceStride();

  if (stepsSinceNeighborListUpdate % nonbondedListUpdateFrequency == 0) {

    if (context->getForceManager()->getPeriodicBoundaryCondition() ==
        PBC::P21) {
      auto groups = context->getForceManager()->getPSF()->getGroups();

      // find a better place for this
      int numGroups = groups.size();
      int numThreads = 128;
      int numBlocks = (numGroups - 1) / numThreads + 1;

      auto boxDimensions = context->getBoxDimensions();
      float3 box = {(float)boxDimensions[0], (float)boxDimensions[1],
                    (float)boxDimensions[2]};

      invertDeltaAsymmetric<<<numBlocks, numThreads, 0, *integratorStream>>>(
          numGroups, groups.getDeviceArray().data(), box.x, xyzq, stride,
          coordsDeltaPreviousDevice);
      cudaCheck(cudaStreamSynchronize(*integratorStream));
    }
    context->resetNeighborList();
  }

  if (debugPrintFrequency > 0 &&
      (stepsSinceLastReport + 1) % debugPrintFrequency == 0) {
    context->calculateForces(false, true, false);
  } else {
    context->calculateForces(false, false, false);
  }

  auto force = context->getForces();

  double kbt = charmm::constants::kBoltz * bathTemperature;
  const double gamma = timeStep * timfac * friction;

  if (usingHolonomicConstraints) {
    copy_DtoD_async<double4>(coords, coordsRef.getDeviceArray().data(),
                             numAtoms, *integratorMemcpyStream);
  }

  int numThreads = 512;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  step1<<<numBlocks, numThreads, 0, *integratorStream>>>(
      kbt, gamma, numAtoms, stride, timeStep, coords, coordsDeltaDevice,
      coordsDeltaPreviousDevice, velMass, force->xyz(), devPHILOXStates);

  cudaCheck(cudaStreamSynchronize(*integratorMemcpyStream));
  if (usingHolonomicConstraints) {
    holonomicConstraint->handleHolonomicConstraints(
        coordsRef.getDeviceArray().data());
  }

  step2<<<numBlocks, numThreads, 0, *integratorStream>>>(
      kbt, gamma, numAtoms, stride, timeStep, coordsRef.getDeviceArray().data(),
      coords, coordsDeltaDevice, coordsDeltaPreviousDevice, velMass,
      force->xyz());

  updateSPKernel<<<numBlocks, numThreads, 0, *integratorStream>>>(numAtoms,
                                                                  xyzq, coords);

  cudaCheck(cudaStreamSynchronize(*integratorStream));
  stepsSinceLastReport++;
  if (debugPrintFrequency > 0 &&
      stepsSinceLastReport % debugPrintFrequency == 0) {
    context->calculateKineticEnergy();
    auto ke = context->getKineticEnergy();
    std::cout << "kinetic energy = " << ke << std::endl;

    // context->getForceManager()->setPrintEnergyDecomposition(true);
    // context->calculateForces(false, true, false);
    // context->getForceManager()->setPrintEnergyDecomposition(false);

    auto peContainer = context->getPotentialEnergy();
    peContainer.transferFromDevice();
    auto pe = peContainer[0];
    std::cout << "potential energy = " << pe << std::endl;
    std::cout << "total energy = " << pe + ke << std::endl;

    std::cout << "[LangTherm]Temp : " << context->computeTemperature() << "\n";
    stepsSinceLastReport = 0;
  }
}

CudaContainer<double4> CudaBAOABIntegrator::getCoordsDeltaPrevious() {
  return coordsDeltaPrevious;
}

std::map<std::string, std::string>
CudaBAOABIntegrator::getIntegratorDescriptors() {
  std::map<std::string, std::string> ret;
  ret["type"] = "LangevinThermostat";
  ret["timeStep"] = std::to_string(timeStep);
  ret["bathTemperature"] = std::to_string(bathTemperature);
  ret["friction"] = std::to_string(friction);
  return ret;
}

void CudaBAOABIntegrator::setCoordsDeltaPrevious(
    std::vector<std::vector<double>> _coordsDeltaPreviousIn) {
  assert((_coordsDeltaPreviousIn.size() == context->getNumAtoms(),
          "Wrong size in setCoordsDeltaPrevious"));
  std::vector<double4> cdpCC;
  for (int i = 0; i < _coordsDeltaPreviousIn.size(); ++i) {
    double4 temp = {_coordsDeltaPreviousIn[i][0], _coordsDeltaPreviousIn[i][1],
                    _coordsDeltaPreviousIn[i][2], 0.0};
    cdpCC.push_back(temp);
  }
  coordsDeltaPrevious.setHostArray(cdpCC);
  coordsDeltaPrevious.transferToDevice();
}