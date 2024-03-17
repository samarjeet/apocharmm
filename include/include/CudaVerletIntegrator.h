// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include "CharmmContext.h"
#include "CudaIntegrator.h"

class CudaVerletIntegrator : public CudaIntegrator {
private:
  XYZQ *oldXYZQ, *newXYZQ;

  std::string integratorTypeName = "Verlet";

public:
  /** @brief Verlet integrator. NOT WORKING
   * @todo Completely obsolete.
   */
  CudaVerletIntegrator(double timeStep);

  // Put these in the base class
  void initialize();
  void propagate(int nSteps);
};
