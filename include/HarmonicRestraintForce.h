// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include "CudaContainer.h"
#include "CudaEnergyVirial.h"
#include "Force.h"
#include <memory>

//
// Calculates restraint forces
//

template <typename AT, typename CT> class HarmonicRestraintForce {
public:
  HarmonicRestraintForce(void);
  ~HarmonicRestraintForce(void);

public:
  std::shared_ptr<Force<AT>> getForce(void);

public:
  void calc_force(const float4 *xyzq, const bool calcEnergy,
                  const bool calcVirial);

private:
  void dealloc(void);

private:
  CudaContainer<CT> m_ForceConstants;
  CudaContainer<double4> m_ReferenceCoordinates;
  CudaEnergyVirial m_EnergyVirial;
  std::shared_ptr<Force<AT>> m_Forces;
  std::shared_ptr<cudaStream_t> m_Stream;
};
