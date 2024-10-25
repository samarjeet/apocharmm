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

/** @brief Integrator with Nose-Hoover thermostat. Not working yet
 * @todo finish this ?
 * @attention Should not be used yet
 */
class CudaNoseHooverThermostatIntegrator : public CudaIntegrator {

public:
  CudaNoseHooverThermostatIntegrator(const double timeStep);

  void initialize(void);

  void propagateOneStep(void) override;

  void setNoseHooverPistonMass(const double nhMass);
  double getNoseHooverPistonMass(void); // { return noseHooverPistonMass; }

  void setBathTemperature(
      const double bathTemperature); // { bathTemperature = temp; }

private:
  int m_ChainLength;

  int m_StepId;

  double m_NoseHooverPistonMass;
  double m_NoseHooverPistonPosition;
  double m_NoseHooverPistonVelocity;
  double m_NoseHooverPistonVelocityPrevious;
  double m_NoseHooverPistonForce;
  double m_NoseHooverPistonForcePrevious;
  double m_BathTemperature;

  std::string m_IntegratorTypeName;
};
