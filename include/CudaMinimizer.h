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

#include "CudaIntegrator.h"
#include <memory>
#include <string>

#include "CharmmContext.h"

class CudaMinimizer : public CudaIntegrator {
public:
  CudaMinimizer();

  // This should not be a raw pointer
  // void setCharmmContext(std::shared_ptr<CharmmContext> csc);

  void initialize();
  void minimize(int numSteps);
  void minimize();

  void setVerboseFlag(bool _flag = true);

private:
  int nsteps;
  // std::shared_ptr<CharmmContext> context;
  std::string method;
  bool verboseFlag;
  std::string integratorTypeName = "Minimizer";
};
