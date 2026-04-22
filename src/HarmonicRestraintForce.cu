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

template <typename AT, typename CT>
HarmonicRestraintForce<AT, CT>::HarmonicRestraintForce(void)
    : m_ForceConstants(), m_ReferenceCoordinates(), m_EnergyVirial(),
      m_Forces(nullptr), m_Stream(nullptr) {}

template <typename AT, typename CT>
HarmonicRestraintForce<AT, CT>::~HarmonicRestraintForce(void) {
  this->dealloc();
}

template <typename AT, typename CT>
std::shared_ptr<Force<AT>> HarmonicRestraintForce<AT, CT>::getForce(void) {
  return m_Forces;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::calc_force(const float4 *xyzq,
                                                const bool calcEnergy,
                                                const bool calcVirial) {
  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::dealloc(void) {
  cudaCheck(cudaStreamDestroy(*m_Stream));
  return;
}

//
// Explicit instances of HarmonicRestraintForce
//
template class HarmonicRestraintForce<long long int, float>;
template class HarmonicRestraintForce<long long int, double>;
