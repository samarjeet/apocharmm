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

/** @brief Integrator with Nose-Hoover thermostat. Not working yet
 * @todo finish this ?
 * @attention Should not be used yet
 */
class CudaNoseHooverThermostatIntegrator : public CudaIntegrator {
private:
  double bathTemperature;
  int chainLength;
  CudaContainer<double4> chainPositions;
  CudaContainer<double4> chainVelocities;

  std::string integratorTypeName = "LangevinThermostat";

public:
  CudaNoseHooverThermostatIntegrator(ts_t timeStep);

  // Put these in the base class
  // void setContext();
  void initialize();
  void propagateOneStep() override;
};