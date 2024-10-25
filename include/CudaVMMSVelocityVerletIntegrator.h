// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "CharmmContext.h"
#include "CudaIntegrator.h"
#include <memory>

class CudaVMMSVelocityVerletIntegrator : public CudaIntegrator {
public:
  CudaVMMSVelocityVerletIntegrator(const double timeStep);

  void initialize(void);
  void setCharmmContexts(const std::vector<CharmmContext> &ctxs);
  void setSoluteAtoms(const std::vector<int> &atoms);
  void propagateOneStep(void) override;

private:
  void combineForces(void);

private:
  std::vector<CharmmContext> m_Contexts;
  CudaContainer<int> m_SoluteAtoms;
  std::shared_ptr<Force<float>> m_CombinedForce;

  std::vector<float> m_Weights;
};
