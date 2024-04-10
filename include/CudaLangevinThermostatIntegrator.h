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
  CudaLangevinThermostatIntegrator(ts_t timeStep);
  /** @brief Creates a Langevin Thermostat integrator using the time step length
   * (ps), the bath temperature (K) and the friction coefficient (ps-1).
   * @param timeStep Time step in ps
   * @param bathTemperature Bath temperature in K
   * @param friction Friction coefficient in ps-1
   */
  CudaLangevinThermostatIntegrator(ts_t timeStep, double bathTemperature,
                                   double friction);
  ~CudaLangevinThermostatIntegrator();

  // Put these in the base class
  // void setContext();
  void initialize();

  /**
   * @brief Set the Friction value in ps ^ -1
   *
   * @param frictionIn
   */
  void setFriction(double frictionIn) { friction = frictionIn; }

  float getFriction() const { return friction; }
  /**
   * @brief Set the Bath Temperature of the thermostat
   *
   * @param temp
   */
  void setBathTemperature(double temp) { bathTemperature = temp; }

  /**
   * @brief Get the Bath Temperature of the thermostat
   *
   * @return double
   */
  double getBathTemperature() const { return bathTemperature; }

  void propagateOneStep() override;

  std::map<std::string, std::string> getIntegratorDescriptors() override;

  CudaContainer<double4> getCoordsDeltaPrevious() override;

  void setCoordsDeltaPrevious(
      std::vector<std::vector<double>> _coordsDeltaPreviousIn) override;

private:
  double friction;
  curandStatePhilox4_32_10_t *devPHILOXStates;

  // CudaContainer<double4> coordsDeltaPrevious;

  int stepsSinceLastReport;
  double bathTemperature;

  std::string integratorTypeName = "LangevinThermostat";
};
