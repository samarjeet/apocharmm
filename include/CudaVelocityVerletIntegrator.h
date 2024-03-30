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
#include <map>
#include <memory>
class CudaVelocityVerletIntegrator : public CudaIntegrator {
private:
  int stepsSinceLastReport;

  std::string integratorTypeName = "VelocityVerlet";

public:
  CudaVelocityVerletIntegrator(ts_t timeStep);

  // Put these in the base class
  // void setContext();
  void setCharmmContext(std::shared_ptr<CharmmContext> charmmContext);
  void initialize();
  void propagateOneStep() override;
  std::map<std::string, std::string> getIntegratorDescriptors() override;
};
