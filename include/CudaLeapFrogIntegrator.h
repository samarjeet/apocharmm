// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include "CharmmContext.h"
#include "CudaIntegrator.h"

class CudaLeapFrogIntegrator : public CudaIntegrator {
private:
  int stepsSinceLastReport;
  std::string integratorTypeName = "CudaLeapFrogIntegrator";

public:
  CudaLeapFrogIntegrator(double timeStep);

  void setCharmmContext(std::shared_ptr<CharmmContext> charmmContext);

  // Put these in the base class
  void initialize() override;
  // void propagate(int nSteps);
  void propagateOneStep() override;

  std::map<std::string, std::string> getIntegratorDescriptors() override;
};
