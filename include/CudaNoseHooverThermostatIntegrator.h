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

class CudaNoseHooverThermostatIntegrator : public CudaIntegrator {
public:
  CudaNoseHooverThermostatIntegrator(const double timeStep);

public:
  void setReferenceTemperature(const double referenceTemperature);
  void setNoseHooverPistonMass(const double noseHooverPistonMass);
  void setNoseHooverPistonVelocity(const double noseHooverPistonVelocity);
  void setNoseHooverPistonVelocityPrevious(
      const double noseHooverPistonVelocityPrevious);
  void setNoseHooverPistonForce(const double noseHooverPistonForce);
  void
  setNoseHooverPistonForcePrevious(const double noseHooverPistonForcePrevious);
  void
  setMaxPredictorCorrectorIterations(const int maxPredictorCorrectorIterations);

public:
  double getReferenceTemperature(void) const;
  const CudaContainer<double> &getNoseHooverPistonMass(void) const;
  const CudaContainer<double> &getNoseHooverPistonVelocity(void) const;
  const CudaContainer<double> &getNoseHooverPistonVelocityPrevious(void) const;
  const CudaContainer<double> &getNoseHooverPistonForce(void) const;
  const CudaContainer<double> &getNoseHooverPistonForcePrevious(void) const;
  int getMaxPredictorCorrectorIterations(void) const;
  const CudaContainer<double> &getKineticEnergy(void) const;
  const CudaContainer<double> &getAverageTemperature(void) const;

  CudaContainer<double> &getNoseHooverPistonMass(void);
  CudaContainer<double> &getNoseHooverPistonVelocity(void);
  CudaContainer<double> &getNoseHooverPistonVelocityPrevious(void);
  CudaContainer<double> &getNoseHooverPistonForce(void);
  CudaContainer<double> &getNoseHooverPistonForcePrevious(void);
  CudaContainer<double> &getKineticEnergy(void);
  CudaContainer<double> &getAverageTemperature(void);

public:
  void initialize(void) override;
  void propagateOneStep(void) override;

protected:
  double computeNoseHooverPistonMass(void);
  void removeCenterOfMassMotion(void);

protected:
  double m_ReferenceTemperature;
  CudaContainer<double> m_NoseHooverPistonMass;
  CudaContainer<double> m_NoseHooverPistonVelocity;
  CudaContainer<double> m_NoseHooverPistonVelocityPrevious;
  CudaContainer<double> m_NoseHooverPistonForce;
  CudaContainer<double> m_NoseHooverPistonForcePrevious;
  CudaContainer<double4> m_CoordsDeltaPredicted;

  int m_MaxPredictorCorrectorIterations;

  CudaContainer<double> m_KineticEnergy;
  CudaContainer<double> m_AverageTemperature;
};
