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

CudaLangevinPistonIntegrator::CudaLangevinPistonIntegrator(ts_t timeStep)
    : CudaLangevinPistonIntegrator(timeStep, CRYSTAL::ORTHORHOMBIC) {}

CudaLangevinPistonIntegrator::CudaLangevinPistonIntegrator(ts_t timeStep,
                                                           CRYSTAL _crystalType)
    : CudaIntegrator(timeStep) {
  // forceScale = 1.0;
  // velScale = 1.0;
  // noiseScale = 0.0;
  pgamma = 0.0;
  stepId = 0; // Remove this

  stepsSinceLastReport = 0;

  noseHooverFlag = true;
  // setNoseHooverPistonMass(computeNoseHooverPistonMass());
  noseHooverPistonPosition = 0.0;
  noseHooverPistonForce = 0.0; // tentative bugfix
  noseHooverPistonForcePrevious = 0.0;

  noseHooverPistonVelocity = 0.0;         // tentative bugfix;
  noseHooverPistonVelocityPrevious = 0.0; // tentative bugfix

  setBathTemperature(300.0);

  usingHolonomicConstraints = true;
  constantSurfaceTensionFlag = false;

  kineticEnergyPressureTensor.allocate(6);
  pressureTensor.allocate(6);
  if (usingHolonomicConstraints) {
    holonomicVirial.allocate(6);
    holonomicVirial.set({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  }

  crystalDimensions.allocate(6);
  crystalDimensionsPrevious.allocate(6);
  inverseCrystalDimensions.allocate(6);

  referencePressure.allocate(6);
  referencePressure.set({1.0, 0.0, 1.0, 0.0, 0.0, 1.0});

  deltaPressure.allocate(6);
  deltaPressureNonChanging.allocate(6);
  deltaPressureHalfStepKinetic.allocate(6);

  palpha = 1.0;
  // pbfact = timeStep * timeStep;
  pvfact = 1.0 / timeStep;

  std::random_device rd{};
  seed = rd();
  rng.seed(seed);

  switch (_crystalType) {
  case CRYSTAL::ORTHORHOMBIC:
    pistonDegreesOfFreedom = 3;
    break;
  case CRYSTAL::TETRAGONAL:
    pistonDegreesOfFreedom = 2;
    break;
  case CRYSTAL::CUBIC:
    pistonDegreesOfFreedom = 1;
    break;
  default:
    throw std::invalid_argument(
        "Invalid crystal type. Please use CRYSTAL::ORTHORHOMBIC, "
        "CRYSTAL::TETRAGONAL or CRYSTAL::CUBIC.");
  }

  // pistonDegreesOfFreedom = 3;
  allocatePistonVariables();
  for (int i = 0; i < pistonDegreesOfFreedom; i++) {
    pistonMass[i] = 500.0;
    inversePistonMass[i] = 1.0 / pistonMass[i];
  }
  inversePistonMass.transferToDevice();

  onStepCrystalFactor.allocate(crystalDegreesOfFreedom);
  halfStepCrystalFactor.allocate(crystalDegreesOfFreedom);

  pressureScalar.allocate(1);

  pressureScalar.allocate(1);
  // Allocations for HFCTEN calculation
  halfStepKineticEnergy.allocate(1);
  halfStepKineticEnergy1StepPrevious.allocate(1);
  halfStepKineticEnergy1StepPrevious.set({0.0});
  halfStepKineticEnergy2StepsPrevious.allocate(1);
  potentialEnergyPrevious.allocate(1);
  potentialEnergyPrevious.set({0.0});
  hfctenTerm.allocate(1);

  // debugTotalPressure = 0.0;
  averagePressureScalar.allocate(1);
  averagePressureScalar.set({0.0});
  averagePressureTensor.allocate(6);
  averagePressureTensor.set({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

  pistonFrictionSetFlag = false;
}

void CudaLangevinPistonIntegrator::allocatePistonVariables() {
  std::vector<double> tempZero(pistonDegreesOfFreedom, 0.0);

  onStepPistonPosition.allocate(pistonDegreesOfFreedom);
  onStepPistonPosition.set(tempZero);
  halfStepPistonPosition.allocate(pistonDegreesOfFreedom);
  halfStepPistonPosition.set(tempZero);
  onStepPistonVelocity.allocate(pistonDegreesOfFreedom);
  onStepPistonVelocity.set(tempZero);
  halfStepPistonVelocity.allocate(pistonDegreesOfFreedom);
  halfStepPistonVelocity.set(tempZero);

  pistonMass.resize(pistonDegreesOfFreedom);
  inversePistonMass.allocate(pistonDegreesOfFreedom);
  inversePistonMass.set(tempZero);
  pistonDeltaPressure.allocate(pistonDegreesOfFreedom);
  pistonDeltaPressure.set(tempZero);
  pressurePistonPositionDelta.allocate(pistonDegreesOfFreedom);
  pressurePistonPositionDelta.set(tempZero);
  pressurePistonPositionDeltaPrevious.allocate(pistonDegreesOfFreedom);
  pressurePistonPositionDeltaPrevious.set(tempZero);
  pressurePistonPositionDeltaStored.allocate(pistonDegreesOfFreedom);
  pressurePistonPositionDeltaStored.set(tempZero);
}

CudaLangevinPistonIntegrator::~CudaLangevinPistonIntegrator() {
  // cudaCheck(cudaFree(devPHILOXStates));
}

void CudaLangevinPistonIntegrator::setPressure(
    std::vector<double> _referencePressure) {
  referencePressure.set(_referencePressure);
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
    std::vector<double> _pistonMass) {
  assert(
      _pistonMass.size() == pistonDegreesOfFreedom &&
      "size of pistonMass vector and pistonDegreesOfFreedom should be equal.");

  for (int i = 0; i < pistonDegreesOfFreedom; i++) {
    if (_pistonMass[i] == 0.0) {
      _pistonMass[i] = std::numeric_limits<double>::max();
      inversePistonMass[i] = 0.0;
      pistonMass[i] = 0.0;
    } else {
      inversePistonMass[i] = 1.0 / _pistonMass[i];
      pistonMass[i] = _pistonMass[i];
    }
  }
  inversePistonMass.transferToDevice();
}

double CudaLangevinPistonIntegrator::computeNoseHooverPistonMass() {
  CudaContainer<double4> velmassCC = context->getVelocityMass();
  velmassCC.transferFromDevice();
  std::vector<double4> velmass = velmassCC.getHostArray();
  double totalMass = 0.0;
  for (int i = 0; i < velmass.size(); i++) {
    totalMass += 1. / velmass[i].w;
  }
  return totalMass / 50.0;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonMass(double _nhMass) {
  noseHooverPistonMass = _nhMass;
}

void CudaLangevinPistonIntegrator::setCrystalType(CRYSTAL _crystalType) {
  crystalType = _crystalType;
  switch (crystalType) {
  case CRYSTAL::ORTHORHOMBIC:
    pistonDegreesOfFreedom = 3;
    break;
  case CRYSTAL::TETRAGONAL:
    pistonDegreesOfFreedom = 2;
    break;
  case CRYSTAL::CUBIC:
    pistonDegreesOfFreedom = 1;
    break;
  default:
    break;
  }
  allocatePistonVariables();
}

CRYSTAL CudaLangevinPistonIntegrator::getCrystalType(void) const {
  return crystalType;
}

void CudaLangevinPistonIntegrator::setSurfaceTension(double st) {

  constantSurfaceTensionFlag = true;

  // Since we only have an orthorhombic box, only Z perpendicular to X-Y are
  // apt.
  surfaceTension = 2 * st;
}

std::vector<double> CudaLangevinPistonIntegrator::getReferencePressure() {
  referencePressure.transferFromDevice();
  return referencePressure.getHostArray();
}

void CudaLangevinPistonIntegrator::setPistonFriction(double _friction) {
  pgamma = _friction;
  double pgam = timfac * timeStep * pgamma;
  palpha = (1 - pgam * 0.5) / (1 + pgam * 0.5);
  pbfact = timeStep * timeStep / (1 + pgam * 0.5);
  pvfact = 0.5 / timeStep;

  double kbt = charmm::constants::kBoltz * bathTemperature;
  assert(pistonDegreesOfFreedom != 0);
  for (int i = 0; i < pistonDegreesOfFreedom; i++)
    prfwd.push_back(sqrt(2 * inversePistonMass[i] * pgam * kbt) / timeStep);

  pistonFrictionSetFlag = true;
}

/** @brief Updates the single-precision coordinates container (xyzq) */
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

/*
__global__ static void setup_kernel(int numAtoms,
                                    curandStatePhilox4_32_10_t *state) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // if (index < numAtoms)
  //   curand_init(1234, index, 0, &state[index]);
}
*/
__global__ static void init(double kbt, const int numAtoms, const int stride,
                            const ts_t timeStep,
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

void CudaLangevinPistonIntegrator::setBoxDimensions(
    std::vector<double> boxDimensionsOriginal) {
  boxDimensions.clear();
  for (int i = 0; i < crystalDegreesOfFreedom; ++i) {
    boxDimensions.push_back(boxDimensionsOriginal[i]);
  }
}

void CudaLangevinPistonIntegrator::initialize() {
  int numAtoms = context->getNumAtoms();

  if (not hasPistonFrictionSet()) {
    throw std::invalid_argument(
        "Piston friction not set. Please set piston friction before "
        "using the Langevin piston integrator.");
  }

  //  Get the mass of the system and divide that by 50. (charmm-gui does that)
  setNoseHooverPistonMass(computeNoseHooverPistonMass());
  // Reset Nose-Hoover piston variables (if integrator is reused from earlier,
  // e.g.)
  noseHooverPistonForce = 0.0;
  noseHooverPistonForcePrevious = 0.0;
  noseHooverPistonVelocity = 0.0;
  noseHooverPistonVelocityPrevious = 0.0;
  onStepPistonVelocity.setToValue(0.0);
  halfStepPistonVelocity.setToValue(0.0);
  onStepPistonPosition.setToValue(0.0);
  halfStepPistonPosition.setToValue(0.0);
  pistonDeltaPressure.setToValue(0.0);
  pressurePistonPositionDelta.setToValue(0.0);
  pressurePistonPositionDelta.setToValue(0.0);
  pressurePistonPositionDeltaPrevious.setToValue(0.0);
  pressurePistonPositionDeltaStored.setToValue(0.0);
  deltaPressure.setToValue(0.0);

  coordsDelta.allocate(numAtoms);
  coordsDeltaPrevious.allocate(numAtoms);
  coordsDeltaPredicted.allocate(numAtoms);

  auto coordsRefDevice = coordsRef.getDeviceArray().data();
  if (usingHolonomicConstraints) {
    holonomicConstraintForces.allocate(numAtoms);
  }

  auto boxDimensionsOriginal = context->getBoxDimensions();
  setBoxDimensions(boxDimensionsOriginal);

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  auto coords = context->getCoordinatesCharges().getDeviceArray().data();

  auto xyzq = context->getXYZQ()->getDeviceXYZQ();

  auto coordsDeltaDevice = coordsDelta.getDeviceArray().data();
  auto coordsDeltaPreviousDevice = coordsDeltaPrevious.getDeviceArray().data();

  auto velMass = context->getVelocityMass().getDeviceArray().data();

  if (usingHolonomicConstraints) {
    copy_DtoD_async<double4>(coords, coordsRefDevice, numAtoms,
                             *integratorStream);
    cudaCheck(cudaStreamSynchronize(*integratorStream));

    holonomicConstraint->handleHolonomicConstraints(coordsRefDevice);
    updateSPKernel<<<numBlocks, numThreads, 0, *integratorStream>>>(
        numAtoms, xyzq, coords);
    copy_DtoD_async<double4>(coords, coordsRefDevice, numAtoms,
                             *integratorStream);
    cudaCheck(cudaStreamSynchronize(*integratorStream));
  }

  context->calculateForces();
  auto force = context->getForces();

  int stride = context->getForceStride();
  double kbt = charmm::constants::kBoltz * bathTemperature;

  init<<<numBlocks, numThreads, 0, *integratorStream>>>(
      kbt, numAtoms, stride, timeStep, // coords,
      coordsDeltaDevice, coordsDeltaPreviousDevice, velMass, force->xyz());
  cudaCheck(cudaStreamSynchronize(*integratorStream));

  if (usingHolonomicConstraints) {
    backStepInitializationKernel<<<numBlocks, numThreads, 0,
                                   *integratorStream>>>(
        numAtoms, coords, coordsDeltaPreviousDevice);

    holonomicConstraint->handleHolonomicConstraints(coordsRefDevice);

    backStepInitializationKernel2<<<numBlocks, numThreads, 0,
                                    *integratorStream>>>(
        numAtoms, coords, coordsRefDevice, coordsDeltaPreviousDevice);
  }
  cudaCheck(cudaStreamSynchronize(*integratorStream));
}

/* Integrate the forces NOT TAKING THE BAROSTAT INTO ACCOUNT, to compute
the non-barostatted half-step velocities. These will be used as initial
value for the predictor corrector.
*/
__global__ static void nonBarostatHalfStepVelocityUpdate(
    double kbt, const int numAtoms, const int stride, const ts_t timeStep,
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

/** @brief One iteration of the predictor-corrector for the crystal dof
 * velocity, the predicted coordinate change and the predicted coordinates */
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
 * on-step velocity */
__global__ static void onStepVelocityCalculation(
    const int numAtoms, const ts_t timeStep, double4 *__restrict__ coordsDelta,
    double4 *__restrict__ coordsDeltaPrevious, double4 *__restrict__ velMass) {

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

/** @brief Computes the kinetic energy contribution to the pressure tensor
 * using previous and next half step velocities. One might think the on-step
 * velocity would be a better thing to use, but it would be unsound (Brooks
 * 1987) */
static __global__ void calculateAverageKineticPressureKernel(
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
static __global__ void calculateHalfStepKineticPressureKernel(
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
/** @brief Projects deltaPressure tensor onto piston dofs */
// projectDeltaPressureOverBoxDim
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

__global__ void averageNetForceKernel(int numAtoms, int stride,
                                      const double *__restrict__ force) {}
void CudaLangevinPistonIntegrator::removeCenterOfMassAverageNetForce() {

  int numAtoms = context->getNumAtoms();
  auto force = context->getForces();

  int stride = context->getForceStride();

  int numThreads = 128;
  int numBlocks = (numAtoms + numThreads - 1) / numThreads;
  averageNetForceKernel<<<numBlocks, numThreads>>>(numAtoms, stride,
                                                   force->xyz());
}

void CudaLangevinPistonIntegrator::removeCenterOfMassMotion() {

  auto pbc = context->getForceManager()->getPeriodicBoundaryCondition();

  // TODO : do this in the kernel rather than the host side

  int numAtoms = context->getNumAtoms();
  auto velocityMass = context->getVelocityMass();
  auto coords = context->getCoordinatesCharges();
  auto boxDimensions = context->getBoxDimensions();
  coords.transferFromDevice();
  coordsDeltaPrevious.transferFromDevice();

  // Remove the center of mass velocity
  float3 cdpcom = {0.0, 0.0, 0.0};

  float totalMass = 0.0;
  for (int i = 0; i < numAtoms; ++i) {
    auto mass = 1 / velocityMass[i].w;

    cdpcom.x += coordsDeltaPrevious[i].x * mass;

    if (pbc == PBC::P21) {
      // cdpcom.y -= coordsDeltaPrevious[i].y * mass;
      // cdpcom.z -= coordsDeltaPrevious[i].z * mass;
    } else {
      cdpcom.y += coordsDeltaPrevious[i].y * mass;
      cdpcom.z += coordsDeltaPrevious[i].z * mass;
    }

    totalMass += mass;
  }
  cdpcom.x /= totalMass;
  cdpcom.y /= totalMass;
  cdpcom.z /= totalMass;

  for (int i = 0; i < numAtoms; ++i) {
    coordsDeltaPrevious[i].x -= cdpcom.x;

    if (pbc == PBC::P21) {
      // coordsDeltaPrevious[i].y += cdpcom.y;
      // coordsDeltaPrevious[i].z += cdpcom.z;
    } else {
      coordsDeltaPrevious[i].y -= cdpcom.y;
      coordsDeltaPrevious[i].z -= cdpcom.z;
    }
  }

  coordsDeltaPrevious.transferToDevice();
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

#include <iomanip> // JEG240724: TEMP DELETE LATER

void CudaLangevinPistonIntegrator::propagateOneStep() {
  auto coords = context->getCoordinatesCharges().getDeviceArray().data();

  auto xyzq = context->getXYZQ()->getDeviceXYZQ();
  auto coordsDeltaDevice = coordsDelta.getDeviceArray().data();
  auto coordsDeltaPreviousDevice = coordsDeltaPrevious.getDeviceArray().data();
  auto coordsRefDevice = coordsRef.getDeviceArray().data();

  auto velMass = context->getVelocityMass().getDeviceArray().data();

  int numDegreesOfFreedom = context->getDegreesOfFreedom();

  double referenceKineticEnergy = 0.5 * context->getDegreesOfFreedom() *
                                  charmm::constants::kBoltz * bathTemperature;

  std::vector<double> sixZeros(6, 0.0);

  if (debugPrintFrequency > 0 && stepId % debugPrintFrequency == 0) {
    std::cout << "\n Step id : " << stepId << "\n---\n";
    for (int i = 0; i < crystalDegreesOfFreedom; ++i) {
      std::cout << "boxDimensions[" << i << "] = " << boxDimensions[i]
                << std::endl;
    }
  }

  auto volume = context->getVolume();
  int numAtoms = context->getNumAtoms();
  int stride = context->getForceStride();
  double kbt = charmm::constants::kBoltz * bathTemperature;
  // const double gamma = timeStep * timfac * friction;

  std::normal_distribution<double> dist(0, 1);

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;
  int numBlocksReduction = 64;

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

  if (stepId % removeCenterOfMassFrequency == 0) {
    removeCenterOfMassMotion();
  }

  copy_DtoD_async<double4>(coords, coordsRef.getDeviceArray().data(), numAtoms,
                           *integratorStream);

  context->calculateForces(false, true, true);
  auto force = context->getForces();

  // if (stepId % removeCenterOfMassFrequency == 0) {
  //  removeCenterOfMassAverageNetForce();
  //}

  noseHooverPistonVelocityPrevious = noseHooverPistonVelocity;
  noseHooverPistonForcePrevious = noseHooverPistonForce;

  nonBarostatHalfStepVelocityUpdate<<<numBlocks, numThreads, 0,
                                      *integratorStream>>>(
      kbt, numAtoms, stride, timeStep, coords, coordsDeltaDevice,
      coordsDeltaPreviousDevice, velMass, force->xyz());

  cudaCheck(cudaStreamSynchronize(*integratorStream));
  auto virialTensor = context->getVirial();
  virialTensor.transferFromDevice();

  // TODO :  Use profiler to determine where we do this computation
  if (usingHolonomicConstraints) {
    holonomicVirial.set(sixZeros);

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaStreamSynchronize(*integratorStream));

    holonomicConstraint->handleHolonomicConstraints(coordsRefDevice);
    cudaCheck(cudaDeviceSynchronize());

    computeHolonomicConstraintForces<<<numBlocks, numThreads, 0,
                                       *integratorStream>>>(
        numAtoms, timeStep, velMass, coordsRefDevice, coords, coordsDeltaDevice,
        holonomicConstraintForces.getDeviceArray().data());

    computeHolonomicConstraintVirial<<<numBlocksReduction, numThreads, 0,
                                       *integratorStream>>>(
        numAtoms, coordsRefDevice,
        holonomicConstraintForces.getDeviceArray().data(),
        holonomicVirial.getDeviceArray().data());

    updateCoordsDeltaAfterConstraint<<<numBlocks, numThreads, 0,
                                       *integratorStream>>>(
        numAtoms, coordsRefDevice, coords, coordsDeltaDevice);

    cudaCheck(cudaStreamSynchronize(*integratorStream));
    holonomicVirial.transferFromDevice();

    virialTensor[0] += holonomicVirial[0];
    virialTensor[1] += holonomicVirial[1];
    virialTensor[2] += holonomicVirial[3];
    virialTensor[3] += holonomicVirial[1];
    virialTensor[4] += holonomicVirial[2];
    virialTensor[5] += holonomicVirial[4];
    virialTensor[6] += holonomicVirial[3];
    virialTensor[7] += holonomicVirial[4];
    virialTensor[8] += holonomicVirial[5];

    virialTensor.transferToDevice();
    cudaCheck(cudaDeviceSynchronize());
  }

  kineticEnergyPressureTensor.set(sixZeros);
  calculateAverageKineticPressureKernel<<<numBlocksReduction, numThreads, 0,
                                          *integratorStream>>>(
      numAtoms, timeStep, coordsDeltaPreviousDevice, coordsDeltaDevice, velMass,
      kineticEnergyPressureTensor.getDeviceArray().data());
  cudaCheck(cudaStreamSynchronize(*integratorStream));
  kineticEnergyPressureTensor.transferFromDevice();

  double volumeFactor = charmm::constants::patmos / volume;
  pressureTensor[0] =
      (virialTensor[0] + kineticEnergyPressureTensor[0]) * volumeFactor;
  pressureTensor[1] =
      (virialTensor[3] + kineticEnergyPressureTensor[1]) * volumeFactor;
  pressureTensor[2] =
      (virialTensor[4] + kineticEnergyPressureTensor[2]) * volumeFactor;
  pressureTensor[3] =
      (virialTensor[6] + kineticEnergyPressureTensor[3]) * volumeFactor;
  pressureTensor[4] =
      (virialTensor[7] + kineticEnergyPressureTensor[4]) * volumeFactor;
  pressureTensor[5] =
      (virialTensor[8] + kineticEnergyPressureTensor[5]) * volumeFactor;

  pressureScalar[0] =
      (pressureTensor[0] + pressureTensor[2] + pressureTensor[5]) / 3.0;
  pressureScalar.transferToDevice();

  if (constantSurfaceTensionFlag) {
    referencePressure[0] =
        (referencePressure[5] - surfaceTension *
                                    charmm::constants::surfaceTensionFactor /
                                    boxDimensions[2]);
    referencePressure[2] = referencePressure[0];
  }

  deltaPressure[0] = pressureTensor[0] - referencePressure[0];
  deltaPressure[1] = pressureTensor[1] - referencePressure[1];
  deltaPressure[2] = pressureTensor[2] - referencePressure[2];
  deltaPressure[3] = pressureTensor[3] - referencePressure[3];
  deltaPressure[4] = pressureTensor[4] - referencePressure[4];
  deltaPressure[5] = pressureTensor[5] - referencePressure[5];
  deltaPressure.transferToDevice();

  auto vcell =
      0.25 * charmm::constants::patmos / volume / (timeStep * timeStep);
  deltaPressureHalfStepKinetic.set(sixZeros);
  calculateHalfStepKineticPressureKernel<<<numBlocksReduction, numThreads, 0,
                                           *integratorStream>>>(
      numAtoms, vcell, coordsDeltaDevice, velMass,
      deltaPressureHalfStepKinetic.getDeviceArray().data());
  cudaCheck(cudaStreamSynchronize(*integratorStream));

  deltaPressureHalfStepKinetic.transferFromDevice();

  for (int i = 0; i < 6; ++i) {
    deltaPressureNonChanging[i] =
        deltaPressure[i] - deltaPressureHalfStepKinetic[i];
  }

  for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
    pressurePistonPositionDeltaStored[i] = pressurePistonPositionDelta[i];
  }

  copy_DtoD_async<double4>(coordsDeltaDevice,
                           coordsDeltaPredicted.getDeviceArray().data(),
                           numAtoms, *integratorStream);

  cudaCheck(cudaStreamSynchronize(*integratorStream));
  std::vector<double> boxDimensionsStored;

  for (int i = 0; i < crystalDegreesOfFreedom; ++i) {
    boxDimensionsStored.push_back(boxDimensions[i]);
  }

  double surfaceTensionInstantaneous =
      0.5 * boxDimensions[2] *
      (pressureTensor[5] - 0.5 * (pressureTensor[0] + pressureTensor[2])) /
      charmm::constants::surfaceTensionFactor;

  // predictor corrector loop
  ///////////////////////////////

  for (int iter = 0; iter < maxPredictorCorrectorSteps; ++iter) {

    for (int i = 0; i < crystalDegreesOfFreedom; ++i) {
      boxDimensions[i] = boxDimensionsStored[i];
    }

    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      pressurePistonPositionDelta[i] = pressurePistonPositionDeltaStored[i];
    }

    // TODO : move to the device
    projectDeltaPressureToPistonDof(crystalType, boxDimensions, deltaPressure,
                                    pistonDeltaPressure);

    double fact = volume * pbfact * charmm::constants::atmosp;

    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      pressurePistonPositionDeltaPrevious[i] = pressurePistonPositionDelta[i];
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

      pressurePistonPositionDelta[i] =
          palpha * pressurePistonPositionDelta[i] +
          inversePistonMass[i] * pistonDeltaPressure[i] * fact +
          pbfact * prfwd[i] * dist(rng);

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

      onStepPistonVelocity[i] = (pressurePistonPositionDelta[i] +
                                 pressurePistonPositionDeltaPrevious[i]) /
                                (2.0 * timeStep);

      // std::cout << "onStepPistonVelocity[" << i
      //           << "] = " << onStepPistonVelocity[i] << std::endl;
    }
    projectCrystalDimensionsToPistonPosition(crystalType, boxDimensions,
                                             onStepPistonPosition);

    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      onStepPistonPosition[i] += pressurePistonPositionDelta[i];
      halfStepPistonPosition[i] =
          onStepPistonPosition[i] - pressurePistonPositionDelta[i] / 2.0;
    }

    // Calculate nose hoover thermal piston velocity and position
    if (noseHooverFlag) {

      // TODO : change this to on step kinetic energy
      /*double onStepKineticEnergy =
          (deltaPressureHalfStepKinetic[0] + deltaPressureHalfStepKinetic[2] +
           deltaPressureHalfStepKinetic[5]) /
          (0.5 * charmm::constants::patmos / volume);
      */
      double onStepKineticEnergy =
          context->computeTemperature() *
          (0.5 * numDegreesOfFreedom * charmm::constants::kBoltz);
      noseHooverPistonForce = 2.0 * timeStep *
                              (onStepKineticEnergy - referenceKineticEnergy) /
                              noseHooverPistonMass;
      if (noseHooverPistonForcePrevious == 0.0) {
        noseHooverPistonForcePrevious = noseHooverPistonForce;
      }

      noseHooverPistonVelocity =
          noseHooverPistonVelocityPrevious +
          (noseHooverPistonForce + noseHooverPistonForcePrevious) / 2.0;
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
    //
    projectPistonQuantitiesToCrystalQuantities(
        crystalType, timeStep, onStepPistonPosition, halfStepPistonPosition,
        onStepPistonVelocity, pressurePistonPositionDelta, onStepCrystalFactor,
        halfStepCrystalFactor, boxDimensions);

    onStepCrystalFactor.transferToDevice();
    halfStepCrystalFactor.transferToDevice();

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
    //        std::cout << "coordsDeltaPrevious: "
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

    predictorCorrectorKernel<<<numBlocks, numThreads, 0, *integratorStream>>>(
        noseHooverFlag, numAtoms, timeStep, noseHooverPistonVelocity,
        coordsRef.getDeviceArray().data(), velMass, coords,
        coordsDeltaPreviousDevice, coordsDeltaDevice,
        coordsDeltaPredicted.getDeviceArray().data(),
        onStepCrystalFactor.getDeviceArray().data(),
        halfStepCrystalFactor.getDeviceArray().data());

    deltaPressureHalfStepKinetic.transferFromDevice();

    deltaPressureHalfStepKinetic.set(sixZeros);

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
                                             *integratorStream>>>(
        numAtoms, vcell, coordsDeltaPredicted.getDeviceArray().data(), velMass,
        deltaPressureHalfStepKinetic.getDeviceArray().data());
    cudaCheck(cudaStreamSynchronize(*integratorStream));
    // xx, yy terms have been checked on CPU side and correspond

    deltaPressureHalfStepKinetic.transferFromDevice();

    for (int i = 0; i < 6; ++i) {
      deltaPressure[i] =
          deltaPressureNonChanging[i] + deltaPressureHalfStepKinetic[i];
    }
  }
  //
  // end predcorr loop
  /////////////////////

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
  if (usingHolonomicConstraints) {

    holonomicConstraint->handleHolonomicConstraints(
        coordsRef.getDeviceArray().data());

    prepareCoordsRefForHolonomicConstraintsKernel<<<numBlocks, numThreads, 0,
                                                    *integratorStream>>>(
        numAtoms, coordsRef.getDeviceArray().data(), coords,
        halfStepCrystalFactor.getDeviceArray().data());

    // Swapping coordsRef and coords as the holonomic constraint will update
    // the coords by calling it from the CharmmContext

    double4 *temporary = coordsRef.getDeviceArray().data();
    double4 *temporary2 = coords;
    coordsRefDevice = coords;
    coords = temporary;
    context->getCoordinatesCharges().getDeviceArray().set(temporary);

    holonomicConstraint->handleHolonomicConstraints(coordsRefDevice);

    // temporary = coordsRef.getDeviceArray().data();
    //  coordsRef.getDeviceArray().set(coords);
    // coordsRefDevice = coords;
    coordsRef.getDeviceArray().set(temporary);
    context->getCoordinatesCharges().getDeviceArray().set(temporary2);

    coords = context->getCoordinatesCharges().getDeviceArray().data();
    coordsRefDevice = coordsRef.getDeviceArray().data();

    updateCoordsDeltaPredictedKernel<<<numBlocks, numThreads, 0,
                                       *integratorStream>>>(
        numAtoms, coordsDeltaPredicted.getDeviceArray().data(), coords,
        coordsRef.getDeviceArray().data());
  }

  copy_DtoD_async<double4>(coordsDeltaPredicted.getDeviceArray().data(),
                           coordsDeltaDevice, numAtoms, *integratorStream);

  onStepVelocityCalculation<<<numBlocks, numThreads, 0, *integratorStream>>>(
      numAtoms, timeStep, coordsDeltaDevice, coordsDeltaPreviousDevice,
      velMass);

  cudaCheck(cudaStreamSynchronize(*integratorStream));
  // Calculate nose hoover thermal piston velocity and position
  if (noseHooverFlag) {

    // TODO : change this to on step kinetic energy
    /*double onStepKineticEnergy =
        (deltaPressureHalfStepKinetic[0] + deltaPressureHalfStepKinetic[2] +
         deltaPressureHalfStepKinetic[5]) /
        (0.5 * charmm::constants::patmos / volume);
    */
    double onStepKineticEnergy =
        context->computeTemperature() *
        (0.5 * numDegreesOfFreedom * charmm::constants::kBoltz);
    noseHooverPistonForce = 2.0 * timeStep *
                            (onStepKineticEnergy - referenceKineticEnergy) /
                            noseHooverPistonMass;

    noseHooverPistonVelocity =
        noseHooverPistonVelocityPrevious +
        (noseHooverPistonForce + noseHooverPistonForcePrevious) / 2.0;
  }

  // start : MovePressureCalculationToEndTesting
  kineticEnergyPressureTensor.set(sixZeros);
  calculateAverageKineticPressureKernel<<<numBlocksReduction, numThreads, 0,
                                          *integratorStream>>>(
      numAtoms, timeStep, coordsDeltaPreviousDevice, coordsDeltaDevice, velMass,
      kineticEnergyPressureTensor.getDeviceArray().data());
  cudaCheck(cudaStreamSynchronize(*integratorStream));
  kineticEnergyPressureTensor.transferFromDevice();

  pressureTensor[0] =
      (virialTensor[0] + kineticEnergyPressureTensor[0]) * volumeFactor;
  pressureTensor[1] =
      (virialTensor[3] + kineticEnergyPressureTensor[1]) * volumeFactor;
  pressureTensor[2] =
      (virialTensor[4] + kineticEnergyPressureTensor[2]) * volumeFactor;
  pressureTensor[3] =
      (virialTensor[6] + kineticEnergyPressureTensor[3]) * volumeFactor;
  pressureTensor[4] =
      (virialTensor[7] + kineticEnergyPressureTensor[4]) * volumeFactor;
  pressureTensor[5] =
      (virialTensor[8] + kineticEnergyPressureTensor[5]) * volumeFactor;

  pressureScalar[0] =
      (pressureTensor[0] + pressureTensor[2] + pressureTensor[5]) / 3.0;
  pressureScalar.transferToDevice();

  averagePressureScalar[0] =
      (stepId / (stepId + 1.0)) * averagePressureScalar[0] +
      (1.0 / (stepId + 1.0)) * pressureScalar[0];
  for (int i = 0; i < 6; i++) {
    averagePressureTensor[i] =
        (stepId / (stepId + 1.0)) * averagePressureTensor[i] +
        (1.0 / (stepId + 1.0)) * pressureTensor[i];
  }

  if (debugPrintFrequency > 0 && stepId % debugPrintFrequency == 0) {
    for (int i = 0; i < 6; ++i) {
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

      std::cout << "pressure[" << i << "] = " << pressureTensor[i] << "   "
                << virialTensor[j] << "   " << kineticEnergyPressureTensor[i]
                << std::endl;
    }

    std::cout << "averagePressureScalar = " << averagePressureScalar[0]
              << std::endl;
  }

  // end :MovePressureCalculationToEndTesting
  // volume = context->getVolume();

  // *********************
  // Calculate HFCTE etc
  double pressureTarget =
      (referencePressure[0] + referencePressure[2] + referencePressure[5]) /
      3.0;
  double pistonPotentialEnergy =
      pressureTarget * volume * charmm::constants::atmosp;

  if (constantSurfaceTensionFlag) {
    pressureTarget = referencePressure[5];
    pistonPotentialEnergy =
        pressureTarget * volume * charmm::constants::atmosp -
        surfaceTension * charmm::constants::surfaceTensionFactor * volume *
            charmm::constants::atmosp / boxDimensions[2];
  }

  // *********************
  if (noseHooverFlag) {
    noseHooverPistonPosition += noseHooverPistonVelocity * timeStep +
                                0.5 * noseHooverPistonForce * timeStep;
  }
  updateSPKernel<<<numBlocks, numThreads, 0, *integratorStream>>>(numAtoms,
                                                                  xyzq, coords);

  copy_DtoD_async<double4>(coordsDeltaDevice, coordsDeltaPreviousDevice,
                           numAtoms, *integratorStream);

  cudaCheck(cudaStreamSynchronize(*integratorStream));

  context->setBoxDimensions(boxDimensions);
  surfaceTensionInstantaneous =
      0.5 * boxDimensions[2] *
      (pressureTensor[5] - 0.5 * (pressureTensor[0] + pressureTensor[2])) /
      charmm::constants::surfaceTensionFactor;

  // HFCTE calculation calculated only in debug mode

  // if (debugPrintFrequency > 0 && stepId % debugPrintFrequency == 0) {
  double pistonKineticEnergy = 0.0;
  for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
    pistonKineticEnergy +=
        0.5 * onStepPistonVelocity[i] * onStepPistonVelocity[i] * pistonMass[i];
  }
  context->calculateKineticEnergy();
  auto ke = context->getKineticEnergy();
  // exit if the kinetic energy is nan
  // if (ke != ke) {
  if (std::isnan(ke)) {
    throw std::runtime_error("NAN detected in kinetic energy");
    exit(1);
  }
  auto peContainer = context->getPotentialEnergy();
  peContainer.transferFromDevice();
  auto pe = peContainer[0];

  deltaPressureHalfStepKinetic.set(sixZeros);
  calculateHalfStepKineticPressureKernel<<<numBlocks, numThreads, 0,
                                           *integratorStream>>>(
      numAtoms, vcell, coordsDeltaPredicted.getDeviceArray().data(), velMass,
      deltaPressureHalfStepKinetic.getDeviceArray().data());
  cudaCheck(cudaStreamSynchronize(*integratorStream));
  deltaPressureHalfStepKinetic.transferFromDevice();
  double totken =
      (deltaPressureHalfStepKinetic[0] + deltaPressureHalfStepKinetic[2] +
       deltaPressureHalfStepKinetic[5]) /
      (0.5 * charmm::constants::patmos / volume);
  // if (halfStepKineticEnergy1StepPrevious[0] == 0.0) {
  //   double epdiff;
  // }

  double hfcten =
      (halfStepKineticEnergy1StepPrevious[0] - ke - pistonKineticEnergy +
       potentialEnergyPrevious[0] - pe - pistonPotentialEnergy) /
          3.0 +
      (halfStepKineticEnergy2StepsPrevious[0] - totken) / 12; // TODO fill it

  if (debugPrintFrequency > 0 && stepId % debugPrintFrequency == 0) {
    std::cout << "Kinetic energy = " << ke << std::endl;

    std::cout << "Potential energy = " << pe << std::endl;
    std::cout << "Piston potential energy = " << pistonPotentialEnergy
              << std::endl;
    std::cout << "Piston kinetic energy = " << pistonKineticEnergy << std::endl;
    std::cout << "Total energy = "
              << pe + ke + pistonPotentialEnergy + pistonKineticEnergy + hfcten
              << std::endl;

    std::cout << "HFCTE = " << hfcten << std::endl;

    std::cout << "Temperature : " << context->computeTemperature() << "\n";
    std::cout << "Surface tension instantaneous : "
              << surfaceTensionInstantaneous << "\n";

    std::cout << "Volume : " << std::fixed << volume << "\n";
    std::cout << "Inv piston mass: ";
    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      std::cout << inversePistonMass[i] << " ";
    }
    std::cout << "\n";
  }
  halfStepKineticEnergy2StepsPrevious[0] =
      halfStepKineticEnergy1StepPrevious[0];
  halfStepKineticEnergy1StepPrevious[0] =
      totken; // fill it current next half step ke
  potentialEnergyPrevious[0] = pe + pistonPotentialEnergy;
  //}

  potentialEnergyPrevious[0] = pe + pistonPotentialEnergy;
  //}

  stepId++;
}

double CudaLangevinPistonIntegrator::getPressureScalar() {
  // averagePressureScalar.transferFromDevice();
  return averagePressureScalar[0];
}

std::vector<double> CudaLangevinPistonIntegrator::getPressureTensor() {
  // averagePressureTensor.transferFromDevice();
  return averagePressureTensor.getHostArray();
}

double CudaLangevinPistonIntegrator::getInstantaneousPressureScalar() {
  // pressureScalar.transferFromDevice();
  return pressureScalar[0];
}

std::vector<double>
CudaLangevinPistonIntegrator::getInstantaneousPressureTensor() {
  // pressureTensor.transferFromDevice();
  return pressureTensor.getHostArray();
}

CudaContainer<double4> CudaLangevinPistonIntegrator::getCoordsDeltaPrevious() {
  return coordsDeltaPrevious;
}

CudaContainer<double4> CudaLangevinPistonIntegrator::getCoordsDelta() {
  return coordsDelta;
}

void CudaLangevinPistonIntegrator::setCoordsDeltaPrevious(
    std::vector<std::vector<double>> _coordsDeltaIn) {
  assert((_coordsDeltaIn.size() == context->getNumAtoms(),
          "Wrong size in setCoordsDeltaPrevious"));
  std::vector<double4> cdpCC;
  for (int i = 0; i < _coordsDeltaIn.size(); ++i) {
    double4 temp;
    temp.x = _coordsDeltaIn[i][0];
    temp.y = _coordsDeltaIn[i][1];
    temp.z = _coordsDeltaIn[i][2];
    temp.w = 0.0;
    cdpCC.push_back(temp);
  }
  coordsDeltaPrevious.setHostArray(cdpCC);
  coordsDeltaPrevious.transferToDevice();
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
CudaLangevinPistonIntegrator::getIntegratorDescriptors() {
  std::map<std::string, std::string> ret;
  ret["type"] = "LangevinPiston";
  ret["timeStep"] = std::to_string(timeStep);
  ret["bathTemperature"] = std::to_string(bathTemperature);
  ret["referencePressure"] = std::to_string(referencePressure[0]);
  ret["pistonMass"] = std::to_string(pistonMass[0]);
  ret["noseHooverPistonMass"] = std::to_string(noseHooverPistonMass);
  return ret;
}
