// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "HarmonicRestraintForce.h"

#include "cuda_utils.h"
#include "gpu_utils.h"
#include <stdexcept>
#include <string>

template <typename AT, typename CT>
HarmonicRestraintForce<AT, CT>::HarmonicRestraintForce(void)
    : m_NumAtoms(-1), m_ForceConstants(), m_ReferenceCoordinates(),
      m_BoxDimensions(), m_EnergyVirial(nullptr), m_Forces(nullptr),
      m_Stream(nullptr) {
  m_BoxDimensions.resize(3);
  m_BoxDimensions.set(0.0);
  m_EnergyVirial = std::make_shared<CudaEnergyVirial>();
  m_EnergyVirial->insert("harm");
  m_Stream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_Stream.get()));
}

template <typename AT, typename CT>
HarmonicRestraintForce<AT, CT>::~HarmonicRestraintForce(void) {
  this->dealloc();
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setForceConstant(
    const double forceConstant) {
  m_ForceConstants.set(forceConstant);
  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setForceConstants(
    const std::vector<double> &forceConstants) {
  if (forceConstants.size() != static_cast<std::size_t>(m_NumAtoms)) {
    std::string msg = "ERROR: HarmonicRestraintForce::setForceConstants(const "
                      "std::vector<double> &): Size of input vector must match "
                      "the total number of atoms\n";
    msg += "                NATOM = " + std::to_string(m_NumAtoms) + "\n";
    msg += "forceConstants.size() = " + std::to_string(forceConstants.size()) +
           "\n";
    if (m_NumAtoms == -1)
      msg += "HINT: Try calling initialize(numAtoms, boxDimensions) first\n";
    throw std::invalid_argument(msg);
  }
  m_ForceConstants = forceConstants;
  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setReferenceCoordinates(
    const std::vector<double4> &referenceCoordinates) {
  if (referenceCoordinates.size() != static_cast<std::size_t>(m_NumAtoms)) {
    std::string msg =
        "ERROR: HarmonicRestraintForce::setReferenceCoordinates(const "
        "std::vector<double4> &): Size of input vector must match "
        "the total number of atoms\n";
    msg += "                      NATOM = " + std::to_string(m_NumAtoms) + "\n";
    msg += "referenceCoordinates.size() = " +
           std::to_string(referenceCoordinates.size()) + "\n";
    if (m_NumAtoms == -1)
      msg += "HINT: Try calling initialize(numAtoms, boxDimensions) first\n";
    throw std::invalid_argument(msg);
  }

  for (int i = 0; i < m_NumAtoms; i++) {
    m_ReferenceCoordinates[i].x = referenceCoordinates[i].x;
    m_ReferenceCoordinates[i].y = referenceCoordinates[i].y;
    m_ReferenceCoordinates[i].z = referenceCoordinates[i].z;
  }
  m_ReferenceCoordinates.transferToDevice();

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setReferenceCoordinates(
    const std::vector<std::vector<double>> &referenceCoordinates) {
  if (referenceCoordinates.size() != static_cast<std::size_t>(m_NumAtoms)) {
    std::string msg =
        "ERROR: HarmonicRestraintForce::setReferenceCoordinates(const "
        "std::vector<double4> &): Size of input vector must match "
        "the total number of atoms\n";
    msg += "                      NATOM = " + std::to_string(m_NumAtoms) + "\n";
    msg += "referenceCoordinates.size() = " +
           std::to_string(referenceCoordinates.size()) + "\n";
    if (m_NumAtoms == -1)
      msg += "HINT: Try calling initialize(numAtoms, boxDimensions) first\n";
    throw std::invalid_argument(msg);
  }

  for (int i = 0; i < m_NumAtoms; i++) {
    m_ReferenceCoordinates[i].x = referenceCoordinates[i][0];
    m_ReferenceCoordinates[i].y = referenceCoordinates[i][1];
    m_ReferenceCoordinates[i].z = referenceCoordinates[i][2];
  }
  m_ReferenceCoordinates.transferToDevice();

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setMasses(
    const std::vector<double> &masses) {
  if (masses.size() != static_cast<std::size_t>(m_NumAtoms)) {
    std::string msg =
        "ERROR: HarmonicRestraintForce::setReferenceCoordinates(const "
        "std::vector<double4> &): Size of input vector must match "
        "the total number of atoms\n";
    msg += "        NATOM = " + std::to_string(m_NumAtoms) + "\n";
    msg += "masses.size() = " + std::to_string(masses.size()) + "\n";
    if (m_NumAtoms == -1)
      msg += "HINT: Try calling initialize(numAtoms, boxDimensions) first\n";
    throw std::invalid_argument(msg);
  }

  for (int i = 0; i < m_NumAtoms; i++)
    m_ReferenceCoordinates[i].w = masses[i];
  m_ReferenceCoordinates.transferToDevice();

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setBoxDimensions(
    const std::vector<double> &boxDimensions) {
  if ((m_BoxDimensions[0] == boxDimensions[0]) &&
      (m_BoxDimensions[1] == boxDimensions[1]) &&
      (m_BoxDimensions[2] == boxDimensions[2]))
    return;

  // JEG260512: This only works for cubic, tetragonal, and orthorhombic boxes.
  // If we ever want to do other crystals (e.g. triclinic) we would need to
  // update all of the code to use the A, B, C, alpha, beta, gamma matrix.
  const double dx = boxDimensions[0] / m_BoxDimensions[0];
  const double dy = boxDimensions[1] / m_BoxDimensions[1];
  const double dz = boxDimensions[2] / m_BoxDimensions[2];

  for (int i = 0; i < m_NumAtoms; i++) {
    m_ReferenceCoordinates[i].x *= dx;
    m_ReferenceCoordinates[i].y *= dy;
    m_ReferenceCoordinates[i].z *= dz;
  }
  m_ReferenceCoordinates.transferToDevice();

  m_BoxDimensions = boxDimensions;

  return;
}

template <typename AT, typename CT>
std::shared_ptr<CudaEnergyVirial>
HarmonicRestraintForce<AT, CT>::getEnergyVirial(void) {
  return m_EnergyVirial;
}

template <typename AT, typename CT>
std::shared_ptr<Force<AT>> HarmonicRestraintForce<AT, CT>::getForce(void) {
  return m_Forces;
}

template <typename AT, typename CT>
std::shared_ptr<cudaStream_t> HarmonicRestraintForce<AT, CT>::getStream(void) {
  return m_Stream;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::initialize(
    const int numAtoms, const std::vector<double> &boxDimensions) {
  if (m_NumAtoms != numAtoms) {
    m_NumAtoms = numAtoms;
    m_ForceConstants.resize(numAtoms);
    m_ReferenceCoordinates.resize(numAtoms);
    m_Forces = std::make_shared<Force<AT>>();
    m_Forces->realloc(numAtoms, 1.5f);

    m_ForceConstants.set(0.0);
    m_ReferenceCoordinates.set(make_double4(0.0, 0.0, 0.0, 1.0));
  }

  this->setBoxDimensions(boxDimensions);

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::clear(void) {
  m_EnergyVirial->clear(*m_Stream);
  m_Forces->clear(*m_Stream);
  return;
}

template <typename AT, typename CT, bool calcEnergy, bool calcVirial>
__global__ static void HarmonicRestraintForceKernel(
    AT *__restrict__ forces, const int forceStride, double *__restrict__ energy,
    const double *__restrict__ forceConstants, const float4 *__restrict__ xyzq,
    const double4 *__restrict__ coordsRef, const int numAtoms) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  double epot = 0.0;
  for (int i = index; i < numAtoms; i += stride) {
    const CT kf = static_cast<CT>(forceConstants[i]);

    const CT rx = static_cast<CT>(xyzq[i].x);
    const CT ry = static_cast<CT>(xyzq[i].y);
    const CT rz = static_cast<CT>(xyzq[i].z);

    const CT rx0 = static_cast<CT>(coordsRef[i].x);
    const CT ry0 = static_cast<CT>(coordsRef[i].y);
    const CT rz0 = static_cast<CT>(coordsRef[i].z);
    const CT mass = static_cast<CT>(coordsRef[i].w);

    const CT dx = rx - rx0;
    const CT dy = ry - ry0;
    const CT dz = rz - rz0;
    const CT dr2 = dx * dx + dy * dy + dz * dz;

    if (calcEnergy == true)
      epot += static_cast<double>(kf * mass * dr2);

    CT dr = 0.0;
    if (sizeof(CT) == 4)
      dr = sqrtf(dr2);
    else
      dr = sqrt(dr2);
    const CT fr = 2.0 * kf * mass * dr;

    AT fx = static_cast<AT>(0), fy = static_cast<AT>(0),
       fz = static_cast<AT>(0);
    calc_component_force<AT, CT>(fr, dx, dy, dz, fx, fy, fz);

    // Store forces
    write_force<AT>(fx, fy, fz, i, forceStride, forces);

    // JEG260501: We can add this calculation later if we implement the external
    // virial. For now, we only have the internal virial, and absolute harmonic
    // restraints only contribute to the external virial.
    //   if (calcVirial) {
    // #ifdef USE_DP_SFORCE
    //     atomicAdd((unsigned long long int *)&virial->sforce_dp[][0],
    //               static_cast<double>(f * dx));
    //     atomicAdd((unsigned long long int *)&virial->sforce_dp[][0],
    //               static_cast<double>(f * dy));
    //     atomicAdd((unsigned long long int *)&virial->sforce_dp[][0],
    //               static_cast<double>(f * dz));
    // #else
    //     fx /= CONVERT_TO_VIR;
    //     fy /= CONVERT_TO_VIR;
    //     fz /= CONVERT_TO_VIR;
    //     atomicAdd((unsigned long long int *)&virial->sforce_fp[][0],
    //     llitoulli(fx)); atomicAdd((unsigned long long int
    //     *)&virial->sforce_fp[][0], llitoulli(fy)); atomicAdd((unsigned long
    //     long int *)&virial->sforce_fp[][0], llitoulli(fz));
    // #endif
    //   }
  }

  if (calcEnergy == true) {
    epot = BlockReduceSum<double>(epot);
    if (threadIdx.x == 0)
      atomicAdd(energy, epot);
  }

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::calcForce(const float4 *xyzq,
                                               const bool calcEnergy,
                                               const bool calcVirial) {
  constexpr int numThreads = 256;
  const int numBlocks = (m_NumAtoms + numThreads - 1) / numThreads;

  if ((calcEnergy == true) && (calcVirial == true)) {
    HarmonicRestraintForceKernel<AT, CT, true, true>
        <<<numBlocks, numThreads, 0, *m_Stream>>>(
            m_Forces->xyz(), m_Forces->stride(),
            m_EnergyVirial->getEnergyPointer("harm"),
            m_ForceConstants.getDeviceArray().data(), xyzq,
            m_ReferenceCoordinates.getDeviceArray().data(), m_NumAtoms);
  } else if ((calcEnergy == true) && (calcVirial == false)) {
    HarmonicRestraintForceKernel<AT, CT, true, false>
        <<<numBlocks, numThreads, 0, *m_Stream>>>(
            m_Forces->xyz(), m_Forces->stride(),
            m_EnergyVirial->getEnergyPointer("harm"),
            m_ForceConstants.getDeviceArray().data(), xyzq,
            m_ReferenceCoordinates.getDeviceArray().data(), m_NumAtoms);
  } else if ((calcEnergy == false) && (calcVirial == true)) {
    HarmonicRestraintForceKernel<AT, CT, false, true>
        <<<numBlocks, numThreads, 0, *m_Stream>>>(
            m_Forces->xyz(), m_Forces->stride(),
            m_EnergyVirial->getEnergyPointer("harm"),
            m_ForceConstants.getDeviceArray().data(), xyzq,
            m_ReferenceCoordinates.getDeviceArray().data(), m_NumAtoms);
  } else if ((calcEnergy == false) && (calcVirial == false)) {
    HarmonicRestraintForceKernel<AT, CT, false, false>
        <<<numBlocks, numThreads, 0, *m_Stream>>>(
            m_Forces->xyz(), m_Forces->stride(),
            m_EnergyVirial->getEnergyPointer("harm"),
            m_ForceConstants.getDeviceArray().data(), xyzq,
            m_ReferenceCoordinates.getDeviceArray().data(), m_NumAtoms);
  }

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::dealloc(void) {
  if (m_Stream != nullptr) {
    cudaCheck(cudaStreamDestroy(*m_Stream));
    m_Stream.reset();
  }
  return;
}

//
// Explicit instances of HarmonicRestraintForce
//
template class HarmonicRestraintForce<long long int, float>;
template class HarmonicRestraintForce<long long int, double>;
