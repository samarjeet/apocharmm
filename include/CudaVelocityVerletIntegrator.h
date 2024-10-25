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
#include <map>
#include <memory>

class CudaVelocityVerletIntegrator : public CudaIntegrator {
public:
  CudaVelocityVerletIntegrator(const double timeStep);

  void initialize(void);

  void propagateOneStep(void) override;

  std::map<std::string, std::string> getIntegratorDescriptors(void) override;

private:
  int m_StepsSinceLastReport;

  std::string m_IntegratorTypeName = "VelocityVerlet";
};
