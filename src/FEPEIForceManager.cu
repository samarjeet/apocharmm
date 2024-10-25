// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "FEPEIForceManager.h"
#include <cassert>
#include <numeric>

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

FEPEIForceManager::FEPEIForceManager() {}

void FEPEIForceManager::initialize() { ForceManagerComposite::initialize(); }

void FEPEIForceManager::setLambdas(std::vector<float> lambdasIn) {
  assertm(lambdasIn[0] == 0.0 && lambdasIn[lambdasIn.size() - 1] == 1.0,
          "0th lambda should be 0.0 and last lambda should be 1.0.");
  lambdas.set(lambdasIn);
  lambdaPotentialEnergies.resize(lambdas.size());
}

void FEPEIForceManager::setSelectorVec(std::vector<float> lambdaIn) {
  // Ensure that only one of entries is 1.0 and
  // the sum of the lambdas is 1.0
  assert(std::accumulate(lambdaIn.begin(), lambdaIn.end(), 0.0));
  auto it = std::find(lambdaIn.begin(), lambdaIn.end(), 1.0);
  assert(it != lambdaIn.end());
  nonZeroLambdaIndex = it - lambdaIn.begin();
  for (int i = 0; i < lambdaIn.size(); ++i) {
    if (i != nonZeroLambdaIndex)
      assert(lambdaIn[i] == 0.0);
  }
  ForceManagerComposite::setSelectorVec(lambdaIn);
}

static __global__ void weighForcesKernel(
    int numAtoms, int stride, int numLambdas, const float *__restrict__ lambdas,
    const double *__restrict__ childrenPEs, double lambda,
    const double *__restrict__ childForce0,
    const double *__restrict__ childForce1, double *__restrict__ totalForce) {

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < numAtoms) {
    // update totalForce
    auto fx = (1 - lambda) * childForce0[index] + lambda * childForce1[index];
    auto fy = (1 - lambda) * childForce0[index + stride] +
              lambda * childForce1[index + stride];
    auto fz = (1 - lambda) * childForce0[index + 2 * stride] +
              lambda * childForce1[index + 2 * stride];

    totalForce[index] = fx;
    totalForce[index + stride] = fy;
    totalForce[index + 2 * stride] = fz;
  }
}

void FEPEIForceManager::weighForces() {

  totalForceValues->clear(*compositeStream);
  int forceStride = totalForceValues->stride();

  int numThreads = 128;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  weighForcesKernel<<<numBlocks, numThreads, 0, *compositeStream>>>(
      numAtoms, forceStride, lambdas.size(), lambdas.getDeviceArray().data(),
      totalPotentialEnergy.getDeviceArray().data(), lambda,
      children[0]->getForces()->xyz(), children[1]->getForces()->xyz(),
      totalForceValues->xyz());
}

float FEPEIForceManager::calc_force(const float4 *xyzq, bool reset,
                                    bool calcEnergy, bool calcVirial) {

  calcEnergy = true; // need the energies to weight the forces
  ForceManagerComposite::calc_force(xyzq, reset, calcEnergy, calcVirial);
  weighForces();
  cudaCheck(cudaStreamSynchronize(*compositeStream));
  return 0.0;
}

std::shared_ptr<Force<double>> FEPEIForceManager::getForces() {

  return totalForceValues;
}

CudaContainer<double> FEPEIForceManager::getVirial() {
  std::cout << "[FEPEIForceManager] Don't call me. Instead call "
               "getVirialInChild with <int> childId\n"
            << "For now, returns child[0]'s virial.";
  return children[0]->getVirial();
}

CudaContainer<double> FEPEIForceManager::getLambdaPotentialEnergies() {
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

// void ForceManagerComposite::storePotentialEnergy(){
//  ++storePECounter;
//  if (storePECounter % storePotentialEnergyFrequency == 0){
//
//  }
//}
//
