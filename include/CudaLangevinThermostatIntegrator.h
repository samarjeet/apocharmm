// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include "CharmmContext.h"
#include "CudaIntegrator.h"
#include <cstdint>
#include <curand_kernel.h>

class CudaLangevinThermostatIntegrator : public CudaIntegrator {
public:
  CudaLangevinThermostatIntegrator(const double timeStep);
  ~CudaLangevinThermostatIntegrator(void);

public:
  void setReferenceTemperature(const double referenceTemperature);
  void setThermostatFriction(const double thermostatFriction);
  void setThermostatRngSeed(const std::uint64_t seed);
  void setRngSequencePos(const unsigned long long int sequencePos);
  void resetAverageTemperature(void);

public:
  double getReferenceTemperature(void) const;
  double getThermostatFriction(void) const;
  std::uint64_t getThermostatRngSeed(void) const;
  unsigned long long int getRngSequencePos(void) const;
  int getAverageWindowSize(void) const;
  const CudaContainer<double> &getKineticEnergy(void) const;
  const CudaContainer<double> &getAverageTemperature(void) const;

  CudaContainer<double> &getKineticEnergy(void);
  CudaContainer<double> &getAverageTemperature(void);

public:
  void initialize(void) override;
  void initializeFromRestartFile(const std::string &rstFileName) override;
  void propagateOneStep(void) override;

protected:
  void initializeRng(void);
  void removeCenterOfMassMotion(void);
  void alloc(const int n);
  void dealloc(void);

protected:
  double m_ReferenceTemperature;
  double m_ThermostatFriction;
  double m_ThermostatGamma;

  std::uint64_t m_Seed;
  unsigned long long int m_RngSequencePos;
  curandStatePhilox4_32_10_t *m_RngStates;

  int m_AverageWindowSize;
  CudaContainer<double> m_KineticEnergy;
  CudaContainer<double> m_AverageTemperature;
};
