// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "ForceManager.h"
#include "MBARForceManager.h"
#include <cassert>
#include <numeric>

MBARForceManager::MBARForceManager() { nonZeroLambdaIndex = -1; }

MBARForceManager::MBARForceManager(
    std::vector<std::shared_ptr<ForceManager>> fmList)
    : ForceManagerComposite(fmList) {
  nonZeroLambdaIndex = -1;
};

void MBARForceManager::initialize() {
  ForceManagerComposite::initialize();

  nonZeroLambdaIndex = 0;
  /*if (nonZeroLambdaIndex == -1) {
    throw std::invalid_argument(
        "selectorVec not set (see MBARForceManager::setSelectorVec)");
  }*/
};

__global__ void
updateChargeInCharmmContext(int numAtoms, const double *__restrict__ d_charges,
                            float4 *__restrict__ xyzq,
                            double4 *__restrict__ coordsCharge) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    xyzq[index].w = (float)d_charges[index];
    coordsCharge[index].w = d_charges[index];
  }
}

void MBARForceManager::setSelectorVec(std::vector<float> lambdaIn) {
  // Ensure that only one of entries is 1.0 and
  // the sum of the lambdas is 1.0
  assert(std::accumulate(lambdaIn.begin(), lambdaIn.end(), 0.0));
  auto it = std::find(lambdaIn.begin(), lambdaIn.end(), 1.0);
  assert(it != lambdaIn.end());
  nonZeroLambdaIndex = it - lambdaIn.begin();
  for (int i = 0; i < lambdaIn.size(); ++i) {
    if (i != nonZeroLambdaIndex)
      // assert((lambdaIn[i] == 0.0) && "Only one of the entries in the
      // selectorVec should be 1.");
      if (lambdaIn[i] != 0.0) {
        throw std::invalid_argument(
            "Only one of the entries in the selectorVec of a MBARForceManager "
            "should be 1.");
      }
  }
  ForceManagerComposite::setSelectorVec(lambdaIn);

  // Testing this design
  auto charges = children[nonZeroLambdaIndex]->getPSF()->getAtomCharges();
  double *d_charges;
  cudaMalloc(&d_charges, sizeof(double) * charges.size());
  cudaMemcpy(d_charges, charges.data(), sizeof(double) * charges.size(),
             cudaMemcpyHostToDevice);

  int numThreads = 128;
  int numBlocks = (charges.size() + numThreads - 1) / numThreads;

  auto xyzq = context->getXYZQ()->getDeviceXYZQ();
  auto coordsCharge = context->getCoordinatesCharges().getDeviceArray().data();

  updateChargeInCharmmContext<<<numBlocks, numThreads>>>(
      charges.size(), d_charges, xyzq, coordsCharge);
  cudaDeviceSynchronize(); // remove this

  cudaFree(d_charges);
}

float MBARForceManager::calc_force(const float4 *xyzq, bool reset,
                                   bool calcEnergy, bool calcVirial) {

  children[nonZeroLambdaIndex]->calc_force(xyzq, reset, calcEnergy, calcVirial);
  auto childPotentialEnergy =
      children[nonZeroLambdaIndex]->getPotentialEnergy();

  cudaMemcpyAsync(totalPotentialEnergy.getDeviceArray().data() +
                      nonZeroLambdaIndex,
                  childPotentialEnergy.getDeviceArray().data(), sizeof(double),
                  cudaMemcpyDeviceToDevice, *compositeStream);
  return 0.0;
}

std::shared_ptr<Force<double>> MBARForceManager::getForces() {

  return children[nonZeroLambdaIndex]->getForces();
}

CudaContainer<double> MBARForceManager::getVirial() {
  return children[nonZeroLambdaIndex]->getVirial();
}