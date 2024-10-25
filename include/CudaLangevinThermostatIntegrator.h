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
#include <curand_kernel.h>
#include <map>
#include <memory>

/** @brief Langevin integrator.
 * If run without setting a friction coefficient nor a bath temperature, will
 * propagate as a NVE integrator
 *
 */
class CudaLangevinThermostatIntegrator : public CudaIntegrator {

public:
  /** @brief Base constructor. Uses timeStep (in ps) as input.
   * @param timeStep Time step in ps
   */
  CudaLangevinThermostatIntegrator(const double timeStep);

  /** @brief Creates a Langevin Thermostat integrator using the time step length
   * (ps), the bath temperature (K) and the friction coefficient (ps-1).
   * @param timeStep Time step in ps
   * @param bathTemperature Bath temperature in K
   * @param friction Friction coefficient in ps-1
   */
  CudaLangevinThermostatIntegrator(const double timeStep,
                                   const double bathTemperature,
                                   const double friction);

  ~CudaLangevinThermostatIntegrator(void);

  // Put these in the base class
  // void setContext();
  void initialize(void);

  /**
   * @brief Set the Friction value in ps ^ -1
   *
   * @param frictionIn
   */
  void setFriction(const double friction);

  double getFriction(void) const;

  /**
   * @brief Set the Bath Temperature of the thermostat
   *
   * @param temp
   */
  void setBathTemperature(const double bathTemperature);

  /**
   * @brief Get the Bath Temperature of the thermostat
   *
   * @return double
   */
  double getBathTemperature(void) const;

  void propagateOneStep(void) override;

  std::map<std::string, std::string> getIntegratorDescriptors(void) override;

  const CudaContainer<double4> &getCoordsDeltaPrevious(void) const override;

  CudaContainer<double4> &getCoordsDeltaPrevious(void) override;

  void setCoordsDeltaPrevious(
      const std::vector<std::vector<double>> &coordsDeltaPrevious) override;

private:
  double m_Friction;
  curandStatePhilox4_32_10_t *m_DevPHILOXStates;
  int m_StepsSinceLastReport;
  double m_BathTemperature;
  std::string m_IntegratorTypeName; // = "LangevinThermostat";
};
