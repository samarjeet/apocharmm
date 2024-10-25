// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once
#include "CharmmContext.h"
#include "CudaIntegrator.h"

class CudaVerletIntegrator : public CudaIntegrator {
public:
  /** @brief Verlet integrator. NOT WORKING
   * @todo Completely obsolete.
   */
  CudaVerletIntegrator(const double timeStep);

  void initialize(void);

  // void propagate(const int nSteps);

private:
  XYZQ *m_OldXYZQ;
  XYZQ *m_NewXYZQ;

  std::string m_IntegratorTypeName = "Verlet";
};
