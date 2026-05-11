// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CharmmContext.h"
#include "ForceManagerComposite.h"
#include "gpu_utils.h"
#include <chrono>
#include <iostream>
#include <string>

ForceManagerComposite::ForceManagerComposite(void) : ForceManager() {
  this->setLambda(1.0f);
}

ForceManagerComposite::ForceManagerComposite(
    const std::vector<std::shared_ptr<ForceManager>> &fmList) {
  for (int i = 0; i < fmList.size(); i++)
    this->addForceManager(fmList[i]);
}

void ForceManagerComposite::setLambda(const float lambda) {
  if (m_Children.size() != 2) {
    throw std::invalid_argument(
        "ERROR: Cannot have more than 2 children force managers");
  }
  m_Lambda = lambda;
  this->setSelectorVec({1 - lambda, lambda});
  return;
}

void ForceManagerComposite::setSelectorVec(const std::vector<float> &lambdas) {
  m_Lambdas = lambdas;
  return;
}

void ForceManagerComposite::addForceManager(std::shared_ptr<ForceManager> fm) {
  const int numAtoms = m_Psf->getNumAtoms();

  if (m_Children.size() > 0)
    assert(numAtoms == fm->getNumAtoms());

  m_Children.push_back(fm);

  // add xyzq as well
  // we only need the charges but it is available only along with xyz
  std::vector<float4> coords(numAtoms);
  const std::vector<double> &charges = fm->getPsf()->getCharges();
  for (int i = 0; i < numAtoms; i++)
    coords[i] = make_float4(0.0f, 0.0f, 0.0f, static_cast<float>(charges[i]));

  // auto tmpxyzq = std::make_shared<XYZQ>();
  // tmpxyzq->set_ncoord(numAtoms);
  // tmpxyzq->set_xyzq(numAtoms, coords.data(), 0);
  // m_XYZQs.push_back(tmpxyzq);
  m_XYZQs.push_back(coords);

  return;
}

void ForceManagerComposite::setBoxDimensions(const std::vector<double> &size) {
  for (auto child : m_Children)
    child->setBoxDimensions(size);
  return;
}

void ForceManagerComposite::setKappa(const float kappa) {
  for (auto child : m_Children)
    child->setKappa(kappa);
  return;
}

void ForceManagerComposite::setCutoff(const float cutoff) {
  for (auto child : m_Children)
    child->setCutoff(cutoff);
  return;
}

void ForceManagerComposite::setCtonnb(const float ctonnb) {
  for (auto child : m_Children)
    child->setCtonnb(ctonnb);
  return;
}

void ForceManagerComposite::setCtofnb(const float ctofnb) {
  for (auto child : m_Children)
    child->setCtofnb(ctofnb);
  return;
}

void ForceManagerComposite::setFFTGrid(const int nfftx, const int nffty,
                                       const int nfftz) {
  for (auto child : m_Children)
    child->setFFTGrid(nfftx, nffty, nfftz);
  return;
}

void ForceManagerComposite::setPmeSplineOrder(const int pmeSplineOrder) {
  for (auto child : m_Children)
    child->setPmeSplineOrder(pmeSplineOrder);
  return;
}

std::shared_ptr<CharmmPSF> ForceManagerComposite::getPsf(void) {
  // Returns the first one !?
  return m_Children[0]->getPsf();
}

bool ForceManagerComposite::isInitialized(void) const {
  for (auto child : m_Children) {
    if (!child->isInitialized())
      return false;
  }
  return true;
}

// Combine the two end states
/* *
__global__ static void CombineKernel(float *__restrict__ forces,
                                     const float *__restrict__ forces1,
                                     const float *__restrict__ forces2,
                                     const int forceStride, const int numAtoms,
                                     const float lambda) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    const int ix = 0 * forceStride + i;
    const int iy = 1 * forceStride + i;
    const int iz = 2 * forceStride + i;
    forces[ix] = (1.0f - lambda) * forces1[ix] + lambda * forces2[ix];
    forces[iy] = (1.0f - lambda) * forces1[ix] + lambda * forces2[iy];
    forces[iz] = (1.0f - lambda) * forces1[iz] + lambda * forces2[iz];
  }

  return;
}
* */

// Combine multiple end states
__global__ static void CombineKernel(double *__restrict__ forces,
                                     const double *__restrict__ childForces,
                                     const int forceStride, const int numAtoms,
                                     const float lambda) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    const int ix = 0 * forceStride + i;
    const int iy = 1 * forceStride + i;
    const int iz = 2 * forceStride + i;
    forces[ix] += static_cast<double>(lambda) * childForces[ix];
    forces[iy] += static_cast<double>(lambda) * childForces[iy];
    forces[iz] += static_cast<double>(lambda) * childForces[iz];
  }

  return;
}

std::shared_ptr<Force<double>> ForceManagerComposite::getForces(void) {
  const int numAtoms = m_Psf->getNumAtoms();

  m_TotalForceValues->clear();

  for (std::size_t i = 0; i < m_Children.size(); i++) {
    if (m_Lambdas[i] != 0.0f) {
      constexpr int numThreads = 512;
      const int numBlocks = (numAtoms + numThreads - 1) / numThreads;
      CombineKernel<<<numBlocks, numThreads, 0, *m_CompositeStream>>>(
          m_TotalForceValues->xyz(), m_Children[i]->getForces()->xyz(),
          this->getForceStride(), numAtoms, m_Lambdas[i]);
    }
  }

  cudaStreamSynchronize(*m_CompositeStream);

  return m_TotalForceValues;
}

std::shared_ptr<Force<double>>
ForceManagerComposite::getForcesInChild(const int childIdx) {
  assert(static_cast<std::size_t>(childIdx) < m_Children.size());
  return m_Children[childIdx]->getForces();
}

const std::vector<double> &ForceManagerComposite::getBoxDimensions(void) const {
  return m_Children[0]->getBoxDimensions();
}

std::vector<double> &ForceManagerComposite::getBoxDimensions(void) {
  return m_Children[0]->getBoxDimensions();
}

CudaContainer<double> &ForceManagerComposite::getPotentialEnergy(void) {
  // JEG260413: This does the same thing as ForceManager::getPotentialEnergy().
  // So I'm not sure why we need to override it. Even if classes that inherit
  // from ForceManagerComposite can override it if they do something different.
  return m_TotalPotentialEnergy;
}

float ForceManagerComposite::getPotentialEnergies(void) {
  return 0.0f; // Dummy fix
}

CudaContainer<double> &ForceManagerComposite::getVirial(void) {
  throw std::runtime_error(
      "Don't call me. Instead call getVirialInChild with <int> childIdx");
  return m_Children[0]->getVirial();
}

bool ForceManagerComposite::isComposite(void) const { return true; }

float ForceManagerComposite::getLambda(void) const { return m_Lambda; }

int ForceManagerComposite::getCompositeSize(void) const {
  return static_cast<int>(m_Children.size());
}

const std::vector<std::shared_ptr<ForceManager>> &
ForceManagerComposite::getChildren(void) const {
  return m_Children;
}

std::vector<std::shared_ptr<ForceManager>> &
ForceManagerComposite::getChildren(void) {
  return m_Children;
}

void ForceManagerComposite::initialize(void) {
  for (auto child : m_Children)
    child->initialize();

  m_TotalForceValues = std::make_shared<Force<double>>();
  m_TotalForceValues->realloc(m_Psf->getNumAtoms(), 1.5f);
  // TODO move this to MBAR FM

  // TODO :: create a stream for each FM
  // it will be used while copying xyz to their respective xyzq s
  m_CompositeStream = std::make_shared<cudaStream_t>();
  cudaStreamCreate(m_CompositeStream.get());

  m_TotalPotentialEnergy.resize(m_Children.size());

  m_ChildrenPotentialEnergy.resize(m_Children.size());

  return;
}

void ForceManagerComposite::resetNeighborList(const float4 *xyzq) {
  for (auto child : m_Children)
    child->resetNeighborList(xyzq);
  return;
}

CudaContainer<double>
ForceManagerComposite::computeAllChildrenPotentialEnergy(const float4 *xyzq) {
  // Compute potential energy for each child
  for (std::size_t i = 0; i < m_Children.size(); i++) {
    m_Children[i]->calcForce(xyzq, false, true, false);
    m_Children[i]->getPotentialEnergy().transferFromDevice();
    m_ChildrenPotentialEnergy[i] = m_Children[i]->getPotentialEnergy()[0];
    // std::cout << "Child " << i << " potential energy : " <<
    // m_ChildrenPotentialEnergy[i] << std::endl;
  }
  // TODO: this is kind of a hotfix. Not sure this is ideal.
  m_ChildrenPotentialEnergy.transferToDevice();
  return m_ChildrenPotentialEnergy;
}

// Given two XYZ pointers, copies coordinates from "src" to "dest"
__global__ static void copyXYZKernel(float4 *__restrict__ dest,
                                     const float4 *__restrict__ src,
                                     const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    dest[i].x = src[i].x;
    dest[i].y = src[i].y;
    dest[i].z = src[i].z;
  }

  return;
}

void ForceManagerComposite::calcForce(const float4 *xyzq, const bool reset,
                                      const bool calcEnergy,
                                      const bool calcVirial) {
  const int numChildren = static_cast<int>(m_Children.size());

  for (int i = 0; i < numChildren; i++) {
    const int numAtoms = m_Psf->getNumAtoms();
    constexpr int numThreads = 128;
    const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

    // Copy xyz part of XYZQ to Device's XYZQ
    copyXYZKernel<<<numBlocks, numThreads, 0, *m_CompositeStream>>>(
        m_XYZQs[i].getDeviceArray().data(), xyzq, numAtoms);
  }
  cudaStreamSynchronize(*m_CompositeStream);

  for (int i = 0; i < numChildren; i++) {
    m_Children[i]->calcForcePart1(m_XYZQs[i].getDeviceArray().data(), reset,
                                  calcEnergy, calcVirial);
  }

  for (int i = 0; i < numChildren; i++) {
    m_Children[i]->calcForcePart2(m_XYZQs[i].getDeviceArray().data(), reset,
                                  calcEnergy, calcVirial);
  }

  for (int i = 0; i < numChildren; i++) {
    m_Children[i]->calcForcePart3(m_XYZQs[i].getDeviceArray().data(), reset,
                                  calcEnergy, calcVirial);
  }

  for (int i = 0; i < numChildren; i++) {
    // TODO : this is not the best way. Profile it to see the impact
    cudaCheck(cudaMemcpyAsync(
        m_TotalPotentialEnergy.getDeviceArray().data() + i,
        m_Children[i]->getPotentialEnergy().getDeviceArray().data(),
        sizeof(double), cudaMemcpyDeviceToDevice, *m_CompositeStream));
  }

  return;
}
