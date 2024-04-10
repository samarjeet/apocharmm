// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include "Constants.h"
#include "CudaNoseHooverThermostatIntegrator.h"
#include "gpu_utils.h"
#include <chrono>
#include <iostream>

CudaNoseHooverThermostatIntegrator::CudaNoseHooverThermostatIntegrator(
    ts_t timeStep)
    : CudaIntegrator(timeStep) {
  // std::cout << "Setting up the nose-hoover integrator.\n";
  chainLength = 5;

  bathTemperature = 300.0;
}

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
void CudaNoseHooverThermostatIntegrator::initialize() {

  int numAtoms = context->getNumAtoms();

  // chainPositions.allocate(chainLength);
  // chainVelocities.allocate(chainLength);
  noseHooverPistonMass = 500.0; // TODO : set this
  noseHooverPistonForce = 0.0;
  noseHooverPistonForcePrevious = 0.0;
  noseHooverPistonVelocity = 0.0;
  noseHooverPistonVelocityPrevious = 0.0;

  coordsDelta.allocate(numAtoms);
  coordsDeltaPrevious.allocate(numAtoms);

  auto coordsRefDevice = coordsRef.getDeviceArray().data();
  if (usingHolonomicConstraints) {
    // holonomicConstraintForces.allocate(numAtoms);
  }

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
    // cudaCheck(cudaDeviceSynchronize());

    holonomicConstraint->handleHolonomicConstraints(coordsRefDevice);
    updateSPKernel<<<numBlocks, numThreads, 0, *integratorStream>>>(
        numAtoms, xyzq, coords);
    copy_DtoD_async<double4>(coords, coordsRefDevice, numAtoms,
                             *integratorStream);
    cudaCheck(cudaStreamSynchronize(*integratorStream));
    // cudaCheck(cudaDeviceSynchronize());
  }

  context->calculateForces();
  auto force = context->getForces();

  int stride = context->getForceStride();
  double kbt = charmm::constants::kBoltz * bathTemperature;

  init<<<numBlocks, numThreads, 0, *integratorStream>>>(
      kbt, numAtoms, stride, timeStep, // coords,
      coordsDeltaDevice, coordsDeltaPreviousDevice, velMass, force->xyz());
  cudaCheck(cudaStreamSynchronize(*integratorStream));
  // cudaCheck(cudaDeviceSynchronize());

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

  stepId = 0;
}

// change this to save delta

// static __global__ void firstHalfKickAndDrift(const int numAtoms,
//                                              const int stride,
//                                              const ts_t timeStep, float4
//                                              *xyzq, double4 *velMass, const
//                                              double *__restrict__ force) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < numAtoms) {
//     float fx = force[index];
//     float fy = force[index + stride];
//     float fz = force[index + 2 * stride];
//
//     velMass[index].x -= 0.5 * timeStep * fx * velMass[index].w;
//     velMass[index].y -= 0.5 * timeStep * fy * velMass[index].w;
//     velMass[index].z -= 0.5 * timeStep * fz * velMass[index].w;
//
//     xyzq[index].x += timeStep * velMass[index].x;
//     xyzq[index].y += timeStep * velMass[index].y;
//     xyzq[index].z += timeStep * velMass[index].z;
//   }
// }
//
// static __global__ void secondHalfKick(const int numAtoms, const int stride,
//                                       const ts_t timeStep, float4 *xyzq,
//                                       double4 *velMass, const double *force)
//                                       {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < numAtoms) {
//
//     float fx = force[index];
//     float fy = force[index + stride];
//     float fz = force[index + 2 * stride];
//
//     velMass[index].x -= 0.5 * timeStep * fx * velMass[index].w;
//     velMass[index].y -= 0.5 * timeStep * fy * velMass[index].w;
//     velMass[index].z -= 0.5 * timeStep * fz * velMass[index].w;
//   }
// }
//
//__global__ void ke(const double4 *velMass, int numAtoms, double *d_ke) {
//   extern __shared__ double sdata[];
//
//   unsigned int id = threadIdx.x;
//   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
//
//   if (tid < numAtoms) {
//     sdata[id] =
//         (velMass[tid].x * velMass[tid].x + velMass[tid].y * velMass[tid].y +
//          velMass[tid].z * velMass[tid].z) /
//         velMass[tid].w;
//   } else {
//     sdata[id] = 0.0f;
//   }
//   __syncthreads();
//
//   for (int s = 1; s < blockDim.x; s *= 2) {
//     if (id % (s * 2) == 0) {
//       sdata[id] += sdata[id + s];
//     }
//     __syncthreads();
//
//     if (id == 0) {
//       d_ke[blockIdx.x] = (double)sdata[0];
//     }
//   }
// }
//
/*
 Velocity-Verlet drift step propagator
*/
// static __global__ void u1Propagator(float deltaT, int numAtoms,
//                                     const double4 *__restrict__ velMass,
//                                     double4 *__restrict__ xyzq) {

//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < numAtoms) {
//     xyzq[index].x += deltaT * velMass[index].x;
//     xyzq[index].y += deltaT * velMass[index].y;
//     xyzq[index].z += deltaT * velMass[index].z;
//   }
// }

// /*
// Velocty-Verlet kick step propagator
// */
// static __global__ void u2Propagator(float deltaT, int numAtoms, int stride,
//                                     double4 *__restrict__ velMass,
//                                     const double *__restrict__ force,
//                                     const double4 *__restrict__ xyzq) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < numAtoms) {

//     const double fx = force[index];
//     const double fy = force[index + stride];
//     const double fz = force[index + 2 * stride];

//     velMass[index].x -= deltaT * fx * velMass[index].w;
//     velMass[index].y -= deltaT * fy * velMass[index].w;
//     velMass[index].z -= deltaT * fz * velMass[index].w;
//   }
// }

///*
// Thermostat propagator
// chain positions
//*/
//  __global__ void
// u3Propagator(float deltaT, int numAtoms, double4 *__restrict__ velMass,
//              int chainLength, double *__restrict__ chainPositions,
//              const double *__restrict__ chainVelocities) {
//   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
//
//   if (index < numAtoms) {
//     velMass[index].x *= exp(-deltaT * chainVelocities[0]);
//     velMass[index].y *= exp(-deltaT * chainVelocities[0]);
//     velMass[index].z *= exp(-deltaT * chainVelocities[0]);
//   }
//
//   if (index == 0) {
//     for (int i = 0; i < chainLength; i++) {
//       chainPositions[i] += deltaT * chainVelocities[i];
//     }
//   }
// }
//
///*
// Thermostat propagator
// chain velocities are updated
//*/
//__global__ void u434Propagator(float deltaT, int numAtoms, double kT,
//                                       int ndegf, double4 *__restrict__
//                                       velMass, int chainLength, double
//                                       *__restrict__ chainPositions, double
//                                       *__restrict__ chainVelocities, const
//                                       double *__restrict__ chainMasses) {
//
//   // Calculate p * p / m
//   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < numAtoms) {
//   }
//
//   double kineticEnergy = 0.0; // TODO : calculate it
//   int M = chainLength - 1;
//   double G =
//       ((chainVelocities[M - 1] * chainVelocities[M - 1]) * chainMasses[M - 1]
//       -
//        kT) /
//       chainMasses[M];
//
//   chainVelocities[M] += deltaT * G;
//
//   for (int m = M - 1; m >= 0; m--) {
//     G = ((chainVelocities[m - 1] * chainVelocities[m - 1]) *
//              chainMasses[m - 1] -
//          kT) /
//         chainMasses[m];
//     double scale = exp(-deltaT * chainVelocities[m]);
//     chainVelocities[m] = scale * (scale * chainVelocities[m] + deltaT * G);
//   }
//
//   // put u3 here
//
//   G = (2 * kineticEnergy - ndegf * kT) / chainMasses[0];
//   for (int m = 0; m < M - 1; m++) {
//     double scale = exp(-deltaT * chainVelocities[m + 1]);
//     chainVelocities[m] = scale * (scale * chainVelocities[m] + deltaT * G);
//     G = (chainVelocities[m] * chainVelocities[m] * chainMasses[m] - kT) /
//         chainMasses[m + 1];
//   }
//   chainVelocities[M] += deltaT * G;
// }

__global__ static void coordsHalfStepVelocityUpdate(
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

void CudaNoseHooverThermostatIntegrator::propagateOneStep() {

  auto coords = context->getCoordinatesCharges().getDeviceArray().data();

  auto xyzq = context->getXYZQ()->getDeviceXYZQ();
  auto coordsDeltaDevice = coordsDelta.getDeviceArray().data();
  auto coordsDeltaPreviousDevice = coordsDeltaPrevious.getDeviceArray().data();
  auto coordsRefDevice = coordsRef.getDeviceArray().data();

  auto velMass = context->getVelocityMass().getDeviceArray().data();

  int numDegreesOfFreedom = context->getDegreesOfFreedom();

  double referenceKineticEnergy = 0.5 * context->getDegreesOfFreedom() *
                                  charmm::constants::kBoltz * bathTemperature;

  if (debugPrintFrequency > 0 && stepId % debugPrintFrequency == 0) {
    std::cout << "Step id : " << stepId << std::endl;
    // std::cout << "\n Step id : " << stepId << "\n---\n";
  }

  int numAtoms = context->getNumAtoms();
  int stride = context->getForceStride();
  double kbt = charmm::constants::kBoltz * bathTemperature;

  if (stepsSinceNeighborListUpdate % nonbondedListUpdateFrequency == 0) {
    /*
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
    */
    context->resetNeighborList();
  }

  if (stepId % removeCenterOfMassFrequency == 0) {
    // TODO : activate this
    // removeCenterOfMassMotion();
  }

  copy_DtoD_async<double4>(coords, coordsRef.getDeviceArray().data(), numAtoms,
                           *integratorStream);

  context->calculateForces(false, true, true);
  auto force = context->getForces();

  noseHooverPistonVelocityPrevious = noseHooverPistonVelocity;
  noseHooverPistonForcePrevious = noseHooverPistonForce;

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;
  // int numBlocksReduction = 64;

  coordsHalfStepVelocityUpdate<<<numBlocks, numThreads, 0, *integratorStream>>>(
      kbt, numAtoms, stride, timeStep, coords, coordsDeltaDevice,
      coordsDeltaPreviousDevice, velMass, force->xyz());

  cudaCheck(cudaStreamSynchronize(*integratorStream));

  // TODO :  Use profiler to determine where we do this computation

  if (usingHolonomicConstraints) {

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaStreamSynchronize(*integratorStream));

    holonomicConstraint->handleHolonomicConstraints(coordsRefDevice);
    cudaCheck(cudaDeviceSynchronize());

    // computeHolonomicConstraintForces<<<numBlocks, numThreads, 0,
    //                                    *integratorStream>>>(
    //     numAtoms, timeStep, velMass, coordsRefDevice, coords,
    //     coordsDeltaDevice,
    //     holonomicConstraintForces.getDeviceArray().data());

    updateCoordsDeltaAfterConstraint<<<numBlocks, numThreads, 0,
                                       *integratorStream>>>(
        numAtoms, coordsRefDevice, coords, coordsDeltaDevice);

    cudaCheck(cudaStreamSynchronize(*integratorStream));
    cudaCheck(cudaDeviceSynchronize());
  }
  // Calculate nose hoover thermal piston velocity and position

  // TODO : change this to on step kinetic energy
  /*double onStepKineticEnergy =
      (deltaPressureHalfStepKinetic[0] + deltaPressureHalfStepKinetic[2] +
       deltaPressureHalfStepKinetic[5]) /
      (0.5 * charmm::constants::patmos / volume);
  */

  onStepVelocityCalculation<<<numBlocks, numThreads, 0, *integratorStream>>>(
      numAtoms, timeStep, coordsDeltaDevice, coordsDeltaPreviousDevice,
      velMass);
  cudaCheck(cudaStreamSynchronize(*integratorStream));

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

  // onStepKineticEnergy = context->computeTemperature() *
  //                       (0.5 * numDegreesOfFreedom *
  //                       charmm::constants::kBoltz);
  // noseHooverPistonForce = 2.0 * timeStep *
  //                         (onStepKineticEnergy - referenceKineticEnergy) /
  //                         noseHooverPistonMass;

  // noseHooverPistonVelocity =
  //     noseHooverPistonVelocityPrevious +
  //     (noseHooverPistonForce + noseHooverPistonForcePrevious) / 2.0;

  noseHooverPistonPosition += noseHooverPistonVelocity * timeStep +
                              0.5 * noseHooverPistonForce * timeStep;

  updateSPKernel<<<numBlocks, numThreads, 0, *integratorStream>>>(numAtoms,
                                                                  xyzq, coords);

  cudaCheck(cudaStreamSynchronize(*integratorStream));

  copy_DtoD_async<double4>(coordsDeltaDevice, coordsDeltaPreviousDevice,
                           numAtoms, *integratorStream);

  cudaCheck(cudaStreamSynchronize(*integratorStream));

  context->calculateKineticEnergy();
  auto ke = context->getKineticEnergy();
  // exit if the kinetic energy is nan
  // if (ke != ke) {
  if (std::isnan(ke)) {
    throw std::runtime_error("NAN detected in kinetic energy");
    exit(1);
  }

  if (debugPrintFrequency > 0 && stepId % debugPrintFrequency == 0) {
    auto peContainer = context->getPotentialEnergy();
    peContainer.transferFromDevice();
    auto pe = peContainer[0];

    std::cout << "Kinetic energy = " << ke << std::endl;

    std::cout << "Potential energy = " << pe << std::endl;
    // std::cout << "Total energy = "
    //           << pe + ke + pistonPotentialEnergy + pistonKineticEnergy +
    //           hfcten
    //           << std::endl;

    // std::cout << "HFCTE = " << hfcten << std::endl;

    std::cout << "Temperature : " << context->computeTemperature() << "\n";
    std::cout << "\n";
  }

  // old code

  /*
 u4 u3 u4
 */

  // int numThreads = 128;
  // int numBlocks = (numAtoms - 1) / numThreads + 1;

  // u2Propagator<<<numBlocks, numThreads>>>(timeStep / 2.0, numAtoms, stride,
  //                                         velMass, force->xyz(), coords);

  // cudaCheck(cudaDeviceSynchronize()); // TODO : remove these

  // u1Propagator<<<numBlocks, numThreads>>>(timeStep, numAtoms, velMass,
  // coords); cudaCheck(cudaDeviceSynchronize()); // TODO : remove these

  // updateSPKernel<<<numBlocks, numThreads, 0, *integratorStream>>>(numAtoms,
  //                                                                 xyzq,
  //                                                                 coords);
  // cudaCheck(cudaDeviceSynchronize()); // TODO : remove these
  // if (stepsSinceNeighborListUpdate % 20 == 0) {
  //   context->resetNeighborList();
  // }

  // context->calculateForces();
  // force = context->getForces();

  // u2Propagator<<<numBlocks, numThreads>>>(timeStep / 2.0, numAtoms, stride,
  //                                         velMass, force->xyz(), coords);
  // cudaCheck(cudaDeviceSynchronize()); // TODO : remove these

  ++stepId;
}
