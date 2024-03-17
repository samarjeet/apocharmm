// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include "CharmmContext.h"
#include "CudaIntegrator.h"
#include <memory>

class CudaVMMSVelocityVerletIntegrator : public CudaIntegrator {
private:
  std::vector<CharmmContext> contexts;
  CudaContainer<int> soluteAtoms;
  std::shared_ptr<Force<float>> combinedForce;

  std::vector<float> weights;

  void combineForces();

public:
  CudaVMMSVelocityVerletIntegrator(ts_t timeStep);

  // Put these in the base class
  // void setContext();
  void initialize();
  void setSimulationContexts(std::vector<CharmmContext> ctxs);
  void setSoluteAtoms(std::vector<int> atoms);
  void propagateOneStep() override;
};
