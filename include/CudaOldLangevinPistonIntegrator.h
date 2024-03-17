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

#include "CudaIntegrator.h"

class CudaLangevinPistonIntegrator : public CudaIntegrator {
private:
  double friction;
public:
  CudaLangevinPistonIntegrator(double timeStep);
  //~CudaLangevinPistonIntegrator();

  // Put these in the base class
  void initialize() override;
  void setFriction(double frictionIn) { friction = frictionIn; }
  void propagateOneStep() override;

};
