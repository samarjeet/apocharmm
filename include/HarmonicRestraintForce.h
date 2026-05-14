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
#include <vector>

//
// Calculates restraint forces
//

template <typename AT, typename CT> class HarmonicRestraintForce {
public:
  HarmonicRestraintForce(void);
  ~HarmonicRestraintForce(void);

public:
  void setForceConstant(const double forceConstant);
  void setForceConstants(const std::vector<double> &forceConstants);
  void
  setReferenceCoordinates(const std::vector<double4> &referenceCoordinates);
  void setReferenceCoordinates(
      const std::vector<std::vector<double>> &referenceCoordinates);
  void setMasses(const std::vector<double> &masses);
  void setBoxDimensions(const std::vector<double> &boxDimensions);

public:
  std::shared_ptr<CudaEnergyVirial> getEnergyVirial(void);
  std::shared_ptr<Force<AT>> getForce(void);
  std::shared_ptr<cudaStream_t> getStream(void);

public:
  void initialize(const int numAtoms, const std::vector<double> &boxDimensions);
  void clear(void);
  void calcForce(const float4 *xyzq, const bool calcEnergy,
                 const bool calcVirial);

private:
  void dealloc(void);

private:
  int m_NumAtoms;
  CudaContainer<double> m_ForceConstants;
  CudaContainer<double4> m_ReferenceCoordinates;
  CudaContainer<double> m_BoxDimensions;
  std::shared_ptr<CudaEnergyVirial> m_EnergyVirial;
  std::shared_ptr<Force<AT>> m_Forces;
  std::shared_ptr<cudaStream_t> m_Stream;
};
