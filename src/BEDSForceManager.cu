// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "BEDSForceManager.h"
#include "Constants.h"

#include <cassert>
#include <cmath>
#include <numeric>
// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

BEDSForceManager::BEDSForceManager() {
  // storePECounter = 0;
  // storePotentialEnergyFrequency = 1000;
  //  setting default s value
  setSValue(0.05);
}

BEDSForceManager::BEDSForceManager(std::shared_ptr<ForceManager> fm1,
                                   std::shared_ptr<ForceManager> fm2) {
  // storePECounter = 0;
  // storePotentialEnergyFrequency = 1000;
  this->addForceManager(fm1);
  this->addForceManager(fm2);
}

void BEDSForceManager::initialize() {
  ForceManagerComposite::initialize();
  energyOffsets.resize(lambdas.size());
  weights.resize(lambdas.size());
  // children[1]->computeDirectSpaceForces = false;
}

/*
static __global__ void
updateWeightsKernel(int numChildren, double beta_s,
                    const double *__restrict__ childrenPEs,
                    const double *__restrict__ energyOffsets,
                    double *referenceHamiltonian, double *weights) {
  double sum = 0.0;
  for (int i = 0; i < numChildren; ++i) {
    weights[i] = std::exp(-beta_s * (childrenPEs[i] - energyOffsets[i]));
    sum += weights[i];
  }
  referenceHamiltonian[0] = -std::log(sum) / (beta_s);

  for (int i = 0; i < numChildren; ++i) {
    weights[i] /= sum;
  }
}
void BEDSForceManager::updateWeights() {
  // TODO : Get the right temperature
  double temperature = 298.17;
  double beta = charmm::constants::kBoltz * temperature;

  // assert(weights.size() == children.size());
  updateWeightsKernel<<<1, 1, 0, *compositeStream>>>(
      children.size(), beta * sValue,
      totalPotentialEnergy.getDeviceArray().data(),
      energyOffsets.getDeviceArray().data(),
      totalPotentialEnergy.getDeviceArray().data(),
      weights.getDeviceArray().data());
}

static __global__ void weighForcesKernel(int numAtoms, int stride, int childId,
                                         const double *__restrict__ weights,
                                         const double *__restrict__ childForce,
                                         double *__restrict__ totalForce) {

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  // for (int i = index; i < numAtoms; i += blockDim.x) {
  int i = index;
  if (index < numAtoms) {

    totalForce[i] += (weights[childId] * childForce[i]);
    totalForce[i + stride] += (weights[childId] * childForce[i + stride]);
    totalForce[i + 2 * stride] +=
        (weights[childId] * childForce[i + 2 * stride]);
  }
}
*/

static __global__ void calculateWeightsKernel(
    double beta_s, int stride, int numLambdas,
    const float *__restrict__ lambdas, const double *__restrict__ childrenPEs,
    const double *__restrict__ energyOffsets, double *__restrict__ weights) {

  // int index = threadIdx.x + blockDim.x * blockIdx.x;
  double sum = 0.0;
  for (int i = 0; i < numLambdas; ++i) {
    auto lambdaEnergy =
        (1 - lambdas[i]) * childrenPEs[0] + lambdas[i] * childrenPEs[1];
    // if (index == 0)
    //   printf("%f \n", lambdaEnergy);
    weights[i] = std::exp(-beta_s * (lambdaEnergy - energyOffsets[i]));
    sum += weights[i];
  }

  auto referencePE = -std::log(sum) / beta_s;
  for (int i = 0; i < numLambdas; ++i) {
    weights[i] /= sum;
  }
  /*weights.transferFromDevice();

  for (int i = 0; i < children.size(); ++i) {
    std::cout << "weight[" << i << "] = " << weights[i] << " \n";
  }
  std::cout << "\n";
  */
}

static __global__ void weighForcesKernel(
    int numAtoms, double beta_s, int stride, int numLambdas,
    const float *__restrict__ lambdas, const double *__restrict__ childrenPEs,
    const double *__restrict__ energyOffsets, double *__restrict__ weights,
    const double *__restrict__ childForce0,
    const double *__restrict__ childForce1, double *__restrict__ totalForce) {

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < numAtoms) {
    for (int i = 0; i < numLambdas; ++i) {
      // update totalForce
      auto fx = (1 - lambdas[i]) * childForce0[index] +
                lambdas[i] * childForce1[index];
      auto fy = (1 - lambdas[i]) * childForce0[index + stride] +
                lambdas[i] * childForce1[index + stride];
      auto fz = (1 - lambdas[i]) * childForce0[index + 2 * stride] +
                lambdas[i] * childForce1[index + 2 * stride];

      totalForce[index] += (weights[i] * fx);
      totalForce[index + stride] += (weights[i] * fy);
      totalForce[index + 2 * stride] += (weights[i] * fz);
    }
  }
}

void BEDSForceManager::weighForces() {
  double temperature = 298.17;
  double beta = charmm::constants::kBoltz * temperature;

  totalForceValues->clear(*compositeStream);
  int forceStride = totalForceValues->stride();

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  // Assuming two end states for now
  calculateWeightsKernel<<<1, 1, 0, *compositeStream>>>(
      beta * sValue, forceStride, lambdas.size(),
      lambdas.getDeviceArray().data(),
      totalPotentialEnergy.getDeviceArray().data(),
      energyOffsets.getDeviceArray().data(), weights.getDeviceArray().data());

  weighForcesKernel<<<numBlocks, numThreads, 0, *compositeStream>>>(
      numAtoms, beta * sValue, forceStride, lambdas.size(),
      lambdas.getDeviceArray().data(),
      totalPotentialEnergy.getDeviceArray().data(),
      energyOffsets.getDeviceArray().data(), weights.getDeviceArray().data(),
      children[0]->getForces()->xyz(), children[1]->getForces()->xyz(),
      totalForceValues->xyz());
}

float BEDSForceManager::calc_force(const float4 *xyzq, bool reset,
                                   bool calcEnergy, bool calcVirial) {

  calcEnergy = true; // need the energies to weight the forces
  ForceManagerComposite::calc_force(xyzq, reset, calcEnergy, calcVirial);
  // updateWeights();
  // In branch EDS, weights on the forces are calcualted on the fly
  weighForces();
  cudaStreamSynchronize(*compositeStream);
  return 0.0;
}

std::shared_ptr<Force<double>> BEDSForceManager::getForces() {
  return totalForceValues;
}

void BEDSForceManager::setSValue(float svalue) { sValue = svalue; }

void BEDSForceManager::setLambdas(std::vector<float> lambdasIn) {
  assertm(lambdasIn[0] == 0.0 && lambdasIn[lambdasIn.size() - 1] == 1.0,
          "0th lambda should be 0.0 and last lambda should be 1.0.");
  lambdas.set(lambdasIn);

  // TODO : remove these allocations from initialize
  weights.resize(lambdas.size());
  energyOffsets.resize(lambdas.size());
  lambdaPotentialEnergies.resize(lambdas.size());
}

void BEDSForceManager::setEndStateEnergyOffsets(
    std::vector<double> _energyOffsets) {

  // for the current version this should be 2
  assertm(_energyOffsets.size() == children.size(),
          "Wrong energyOffsets size -- BEDSForceManager might not be "
          "initialized.");
  assertm(lambdas.size() != 0,
          "Make sure the lambda values are set before setting end-point energy "
          "offsets. API will change soon to fix this requirement.\n");
  std::vector<double> allEnergyOffsets(lambdas.size(), 0.0);
  allEnergyOffsets[0] = _energyOffsets[0];
  allEnergyOffsets[allEnergyOffsets.size() - 1] = _energyOffsets[1];

  for (int i = 0; i < lambdas.size(); ++i) {
    allEnergyOffsets[i] =
        (1 - lambdas[i]) * _energyOffsets[0] + lambdas[i] * _energyOffsets[1];
  }

  energyOffsets = allEnergyOffsets;
}

CudaContainer<double> BEDSForceManager::getLambdaPotentialEnergies() {

  // Doing these on the host side itself as they are not costly
  // Move them to the device in the second pass
  // weights.transferFromDevice();
  totalPotentialEnergy.transferFromDevice();
  for (int i = 0; i < lambdas.size(); ++i) {
    lambdaPotentialEnergies[i] = (1 - lambdas[i]) * totalPotentialEnergy[0] +
                                 lambdas[i] * totalPotentialEnergy[1];
  }

  return lambdaPotentialEnergies;
}
