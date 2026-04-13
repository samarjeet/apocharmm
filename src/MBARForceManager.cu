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

void MBARForceManager::setSelectorVec(const std::vector<float> &lambdas) {
  // Ensure that only one of entries is 1.0 and
  // the sum of the lambdas is 1.0
  assert(std::accumulate(lambdas.begin(), lambdas.end(), 0.0));
  auto it = std::find(lambdas.begin(), lambdas.end(), 1.0);
  assert(it != lambdas.end());
  nonZeroLambdaIndex = it - lambdas.begin();
  for (int i = 0; i < lambdas.size(); ++i) {
    if (i != nonZeroLambdaIndex)
      // assert((lambdas[i] == 0.0) && "Only one of the entries in the
      // selectorVec should be 1.");
      if (lambdas[i] != 0.0) {
        throw std::invalid_argument(
            "Only one of the entries in the selectorVec of a MBARForceManager "
            "should be 1.");
      }
  }

  ForceManagerComposite::setSelectorVec(lambdas);

  // Testing this design
  auto charges = m_Children[nonZeroLambdaIndex]->getPsf()->getAtomCharges();
  double *d_charges;
  cudaMalloc(&d_charges, sizeof(double) * charges.size());
  cudaMemcpy(d_charges, charges.data(), sizeof(double) * charges.size(),
             cudaMemcpyHostToDevice);

  int numThreads = 128;
  int numBlocks = (charges.size() + numThreads - 1) / numThreads;

  auto xyzq = m_Context->getXYZQ()->getDeviceXYZQ();
  auto coordsCharge =
      m_Context->getCoordinatesCharges().getDeviceArray().data();

  updateChargeInCharmmContext<<<numBlocks, numThreads>>>(
      charges.size(), d_charges, xyzq, coordsCharge);
  cudaDeviceSynchronize(); // remove this

  cudaFree(d_charges);
}

void MBARForceManager::calcForce(const float4 *xyzq, bool reset,
                                 bool calcEnergy, bool calcVirial) {

  m_Children[nonZeroLambdaIndex]->calcForce(xyzq, reset, calcEnergy,
                                            calcVirial);
  auto childPotentialEnergy =
      m_Children[nonZeroLambdaIndex]->getPotentialEnergy();

  cudaMemcpyAsync(m_TotalPotentialEnergy.getDeviceArray().data() +
                      nonZeroLambdaIndex,
                  childPotentialEnergy.getDeviceArray().data(), sizeof(double),
                  cudaMemcpyDeviceToDevice, *m_CompositeStream);
  return;
}

std::shared_ptr<Force<double>> MBARForceManager::getForces() {

  return m_Children[nonZeroLambdaIndex]->getForces();
}

CudaContainer<double> &MBARForceManager::getVirial() {
  return m_Children[nonZeroLambdaIndex]->getVirial();
}
