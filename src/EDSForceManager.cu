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
#include "EDSForceManager.h"

#include <cassert>
#include <cmath>
#include <numeric>
// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

EDSForceManager::EDSForceManager() {
  storePECounter = 0;
  storePotentialEnergyFrequency = 1000;
  // setting default s value
  setSValue(0.05);
}

EDSForceManager::EDSForceManager(std::shared_ptr<ForceManager> fm1,
                                 std::shared_ptr<ForceManager> fm2) {
  storePECounter = 0;
  storePotentialEnergyFrequency = 1000;
  this->addForceManager(fm1);
  this->addForceManager(fm2);
}

void EDSForceManager::initialize() {
  ForceManagerComposite::initialize();
  energyOffsets.allocate(children.size());
  weights.allocate(children.size());
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
*/
static __global__ void
updateWeightsKernel(int numChildren, double beta_s,
                    const double *__restrict__ childrenPEs,
                    const double *__restrict__ energyOffsets,
                    double *referenceHamiltonian, double *weights) {
  double sum = 0.0;
  double maxAbs = 0.0;
  for (int i = 0; i < numChildren; ++i) {
    if (std::abs(childrenPEs[i] - energyOffsets[i]) > maxAbs) {
      maxAbs = std::abs(childrenPEs[i] - energyOffsets[i]);
    }
  }

  for (int i = 0; i < numChildren; ++i) {
    double weightDenominator = 0.0;
    for (int j = 0; j < numChildren; ++j) {
      if (i != j) {
        double deltaE = childrenPEs[j] - childrenPEs[i];
        double deltaOffset = energyOffsets[j] - energyOffsets[i];
        weightDenominator += std::exp(-beta_s * (deltaE - deltaOffset));
      } else {
        weightDenominator += 1.0;
      }
    }
    weights[i] = 1.0 / weightDenominator;

    // weights[i] = std::exp(-beta_s * (childrenPEs[i] - energyOffsets[i]));
    sum += std::exp(-beta_s * (childrenPEs[i] - energyOffsets[i] + maxAbs));
  }
  referenceHamiltonian[0] = -std::log(sum) / (beta_s)-maxAbs;

  /*
  for (int i = 0; i < numChildren; ++i) {
    weights[i] /= sum;
  }
  */
}

void EDSForceManager::updateWeights() {
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
  cudaCheck(cudaStreamSynchronize(*compositeStream));

  /*weights.transferFromDevice();

  for (int i = 0; i < children.size(); ++i) {
    std::cout << "weight[" << i << "] = " << weights[i] << " \n";
  }
  std::cout << "\n";
  */
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

void EDSForceManager::weighForces() {

  totalForceValues->clear(*compositeStream);
  int forceStride = totalForceValues->stride();

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  for (int i = 0; i < children.size(); ++i) {
    weighForcesKernel<<<numBlocks, numThreads, 0, *compositeStream>>>(
        numAtoms, forceStride, i, weights.getDeviceArray().data(),
        children[i]->getForces()->xyz(), totalForceValues->xyz());
    // cudaStreamSynchronize(*compositeStream);
  }
}

float EDSForceManager::calc_force(const float4 *xyzq, bool reset,
                                  bool calcEnergy, bool calcVirial) {

  calcEnergy = true; // need the energies to weight the forces
  ForceManagerComposite::calc_force(xyzq, reset, calcEnergy, calcVirial);
  updateWeights();
  weighForces();
  cudaStreamSynchronize(*compositeStream);
  return 0.0;
}

std::shared_ptr<Force<double>> EDSForceManager::getForces() {
  return totalForceValues;
}

void EDSForceManager::setSValue(float svalue) { sValue = svalue; }

void EDSForceManager::setEnergyOffsets(std::vector<double> _energyOffsets) {
  assertm(
      energyOffsets.size() == _energyOffsets.size(),
      "Wrong energyOffsets size -- EDSForceManager might not be initialized.");
  energyOffsets.setHostArray(_energyOffsets);
  energyOffsets.transferToDevice();
}

float EDSForceManager::getSValue() { return sValue; }

CudaContainer<double> EDSForceManager::getEnergyOffsets() {
  return energyOffsets;
}