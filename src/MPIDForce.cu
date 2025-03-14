// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

/**\file*/

#include "MPIDForce.h"
#include <cassert>
#include <iostream>

template <typename AT, typename CT>
MPIDForce<AT, CT>::MPIDForce(CudaEnergyVirial &energyVirial)
    : energyVirial(energyVirial) {
  // TODO Auto-generated constructor stub

  std::cout << "MPIDForce::MPIDForce" << std::endl;

  nonBondedMethod = NonBondedMethod::PME;
  polarizationType = InducedDipoleMethod::Extrapolated;
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::setStream(std::shared_ptr<cudaStream_t> _stream) {
  this->stream = _stream;
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::setNonBondedMethod(NonBondedMethod _nonBondedMethod) {
  this->nonBondedMethod = _nonBondedMethod;
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::setInducedDipoleMethod(
    InducedDipoleMethod _polarizationType) {
  polarizationType = _polarizationType;
  // this->polarizationType = _polarizationType;
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::setCutoff(double _cutoff) {
  this->cutoff = _cutoff;
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::setKappa(double _kappa) {
  this->kappa = _kappa;
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::setFFTGrid(int _nx, int _ny, int _nz) {
  this->nx = _nx;
  this->ny = _ny;
  this->nz = _nz;
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::setDefaultTholeWidth(double _defaultTholeWidth) {
  this->defaultTholeWidth = _defaultTholeWidth;
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::setNumAtoms(int _numAtoms) {
  numAtoms = _numAtoms;
}

template <typename AT, typename CT> int MPIDForce<AT, CT>::getNumAtoms() const {
  return numAtoms;
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::setDipoles(const std::vector<float> &dipoles) {
  assert(dipoles.size() == 3 * numAtoms);
  molecularDipoles.set(dipoles);
  molecularDipoles.transferToDevice();
}

// Might have to change the order of the quadrupoles
template <typename AT, typename CT>
void MPIDForce<AT, CT>::setQuadrupoles(const std::vector<float> &quadrupoles) {
  assert(quadrupoles.size() == 6 * numAtoms);
  molecularQuadrupoles.allocate(5 * numAtoms);
  for (int i = 0; i < numAtoms; i++) {
    int offset = 6 * i;
    molecularQuadrupoles[5 * i] = quadrupoles[offset];         // XX
    molecularQuadrupoles[5 * i + 1] = quadrupoles[offset + 1]; // XY
    molecularQuadrupoles[5 * i + 2] = quadrupoles[offset + 3]; // XZ
    molecularQuadrupoles[5 * i + 3] = quadrupoles[offset + 2]; // YY
    molecularQuadrupoles[5 * i + 4] = quadrupoles[offset + 4]; // YZ
  }
  molecularQuadrupoles.transferToDevice();
}

// Might have to change the order of the octopoles
template <typename AT, typename CT>
void MPIDForce<AT, CT>::setOctopoles(const std::vector<float> &octopoles) {
  assert(octopoles.size() == 10 * numAtoms);
  molecularOctopoles.allocate(7 * numAtoms);
  for (int i = 0; i < numAtoms; i++) {
    int offset = 10 * i;
    molecularOctopoles[7 * i] = octopoles[offset];         // XXX
    molecularOctopoles[7 * i + 1] = octopoles[offset + 1]; // XXY
    molecularOctopoles[7 * i + 2] = octopoles[offset + 4]; // XXZ
    molecularOctopoles[7 * i + 3] = octopoles[offset + 2]; // XYY
    molecularOctopoles[7 * i + 4] = octopoles[offset + 5]; // XYZ
    molecularOctopoles[7 * i + 5] = octopoles[offset + 3]; // YYY
    molecularOctopoles[7 * i + 6] = octopoles[offset + 6]; // YYZ
  }
  molecularOctopoles.transferToDevice();
}

template <typename AT, typename CT> void MPIDForce<AT, CT>::setup() {

  std::cout << "MPIDForce::setup starts" << std::endl;

  labFrameDipoles.allocate(3 * numAtoms);
  labFrameQuadrupoles.allocate(5 * numAtoms);
  labFrameOctopoles.allocate(7 * numAtoms);

  sphericalDipoles.allocate(3 * numAtoms);
  sphericalQuadrupoles.allocate(5 * numAtoms);
  sphericalOctopoles.allocate(7 * numAtoms);
}

// Each thread computes the spherical multipole moments for one atom
__global__ void
computeLabFrameMoments(int numAtoms, const float4 *__restrict__ coordsCharge,
                       const float *__restrict__ molecularDipoles,
                       const float *__restrict__ molecularQuadrupoles,
                       const float *__restrict__ molecularOctopoles,
                       float *__restrict__ labFrameDipoles,
                       float *__restrict__ labFrameQuadrupoles,
                       float *__restrict__ labFrameOctopoles,
                       float *__restrict__ sphericalDipoles,
                       float *__restrict__ sphericalQuadrupoles,
                       float *__restrict__ sphericalOctopoles) {

  for (int atomIndex = blockIdx.x * blockDim.x + threadIdx.x;
       atomIndex < numAtoms;
       atomIndex += blockDim.x * gridDim.x) { // loop over atoms

    // Convert from cartesian to spherical representation
    int offset = 3 * atomIndex;
    sphericalDipoles[offset] = molecularDipoles[offset + 2];     // z -> Q_10
    sphericalDipoles[offset + 1] = molecularDipoles[offset];     // x -> Q_11c
    sphericalDipoles[offset + 2] = molecularDipoles[offset + 1]; // y -> Q_11s

    offset = 5 * atomIndex;
    sphericalQuadrupoles[offset] =
        -3.0f * (molecularQuadrupoles[offset] +
                 molecularQuadrupoles[offset + 3]); // xx + yy (i.e. zz)-> Q_20
    sphericalQuadrupoles[offset + 1] =
        (2.0 * sqrtf(3.0f)) * molecularQuadrupoles[offset + 2]; // xz -> Q_21c
    sphericalQuadrupoles[offset + 2] =
        (2.0 * sqrtf(3.0f)) * molecularQuadrupoles[offset + 4]; // yz -> Q_21s
    sphericalQuadrupoles[offset + 3] =
        sqrtf(3.0f) * (molecularQuadrupoles[offset] -
                       molecularQuadrupoles[offset + 3]); // xx - yy -> Q_22c
    sphericalQuadrupoles[offset + 4] =
        (2.0 * sqrtf(3.0f)) * molecularQuadrupoles[offset + 1]; // xy -> Q_22s

    offset = 7 * atomIndex;

    sphericalOctopoles[offset] =
        -15.0f * (molecularOctopoles[offset + 2] +
                  molecularOctopoles[offset + 6]); // zzz -> Q_30
    sphericalOctopoles[offset + 1] =
        -15.0f * sqrtf(1.5f) *
        (molecularOctopoles[offset] +
         molecularOctopoles[offset + 3]); // xzz -> Q_31c
    sphericalOctopoles[offset + 2] =
        -15.0f * sqrtf(1.5f) *
        (molecularOctopoles[offset + 1] +
         molecularOctopoles[offset + 5]); // yzz -> Q_31s
    sphericalOctopoles[offset + 3] =
        15.0f * sqrtf(0.6f) *
        (molecularOctopoles[offset + 2] +
         molecularOctopoles[offset + 6]); // xxz-yyz -> Q_32c
    sphericalOctopoles[offset + 4] =
        30.0f * sqrtf(0.6f) * (molecularOctopoles[offset + 4]); // xyz -> Q_32s
    sphericalOctopoles[offset + 5] =
        15.0f * sqrtf(0.1f) *
        (molecularOctopoles[offset] -
         3.0 * molecularOctopoles[offset + 3]); // xxx-xyy -> Q_33c
    sphericalOctopoles[offset + 6] =
        15.0f * sqrtf(0.1f) *
        (3.0f * molecularOctopoles[offset + 1] -
         molecularOctopoles[offset + 5]); // xxy-yyy -> Q_33s
  }
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::calculateForce(const float4 *xyzq, bool calcEnergy,
                                       bool calcVirial) {

  std::cout << "MPIDForce::calculateForces starts" << std::endl;

  int numThreads = 256;
  int numBlocks = 64;
  // Compute moments
  computeLabFrameMoments<<<numBlocks, numThreads, 0, *stream>>>(
      numAtoms, xyzq, molecularDipoles.getDeviceArray().data(),
      molecularQuadrupoles.getDeviceArray().data(),
      molecularOctopoles.getDeviceArray().data(),
      labFrameDipoles.getDeviceArray().data(),
      labFrameQuadrupoles.getDeviceArray().data(),
      labFrameOctopoles.getDeviceArray().data(),
      sphericalDipoles.getDeviceArray().data(),
      sphericalQuadrupoles.getDeviceArray().data(),
      sphericalOctopoles.getDeviceArray().data());

  cudaStreamSynchronize(*stream);
}

// for debugging

template <typename AT, typename CT>
void MPIDForce<AT, CT>::printSphericalDipoles(void) {
  std::cout << "Printing spherical dipoles" << std::endl;
  sphericalDipoles.transferFromDevice();
  for (int i = 0; i < numAtoms; i++) {
      std::cout << sphericalDipoles[3*i] << " " << sphericalDipoles[3*i+1] << " "
                << sphericalDipoles[3*i+2] << std::endl;    
  }
}

template <typename AT, typename CT>
void MPIDForce<AT, CT>::printSphericalQuadrupoles(void) {
  std::cout << "Printing spherical quadrupoles" << std::endl;
  sphericalQuadrupoles.transferFromDevice();
  for (int i = 0; i < numAtoms; i++) {
      std::cout << sphericalQuadrupoles[5*i] << " " << sphericalQuadrupoles[5*i+1] << " "
                << sphericalQuadrupoles[5*i+2] << " " << sphericalQuadrupoles[5*i+3] << " "
                << sphericalQuadrupoles[5*i+4] << std::endl;    
  }
}

// Template instantiation
template class MPIDForce<long long int, float>;