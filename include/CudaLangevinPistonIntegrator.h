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

class CudaLangevinPistonIntegrator : public CudaIntegrator {
public:
  CudaLangevinPistonIntegrator(const double timeStep);
  ~CudaLangevinPistonIntegrator(void);

public:
  void useNoseHooverThermostat(const bool usingNoseHooverThermostat);
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
  void useOldTemperature(const bool usingOldTemperature);

  void setPressure(const std::vector<double> &referencePressure);
  void setConstantSurfaceTension(const bool constantSurfaceTensionFlag);
  void setCrystalType(const CRYSTAL crystalType);
  void setLangevinPistonMass(const std::vector<double> &mass);
  void setLangevinPistonFrictionSeed(const std::uint64_t seed);
  void setRngSequencePos(const unsigned long long int rngSequencePos);
  void setLangevinPistonFriction(const double pgamma);
  void resetAverages(void);

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
  const CudaContainer<double> &getAverageOldTemperature(void) const;

  CRYSTAL getCrystalType(void) const;
  const CudaContainer<double> &getReferencePressureTensor(void) const;
  const CudaContainer<double> &getLangevinPistonMass(void) const;
  const CudaContainer<double> &getLangevinPistonOnStepPosition(void) const;
  const CudaContainer<double> &getLangevinPistonHalfStepPosition(void) const;
  const CudaContainer<double> &getLangevinPistonOnStepVelocity(void) const;
  const CudaContainer<double> &getLangevinPistonHalfStepVelocity(void) const;
  const CudaContainer<double> &getLangevinPistonDeltaPosition(void) const;
  const CudaContainer<double> &
  getLangevinPistonDeltaPositionPrevious(void) const;
  const CudaContainer<double> &
  getLangevinPistonDeltaPositionPredicted(void) const;
  const CudaContainer<double> &getLangevinPistonDeltaPressure(void) const;
  const CudaContainer<double> &getInstantaneousPressureTensor(void) const;
  const CudaContainer<double> &getInstantaneousPressureScalar(void) const;
  const CudaContainer<double> &getAveragePressureTensor(void) const;
  const CudaContainer<double> &getAveragePressureScalar(void) const;

  CudaContainer<double> &getNoseHooverPistonMass(void);
  CudaContainer<double> &getNoseHooverPistonVelocity(void);
  CudaContainer<double> &getNoseHooverPistonVelocityPrevious(void);
  CudaContainer<double> &getNoseHooverPistonForce(void);
  CudaContainer<double> &getNoseHooverPistonForcePrevious(void);
  CudaContainer<double> &getKineticEnergy(void);
  CudaContainer<double> &getAverageTemperature(void);
  CudaContainer<double> &getAverageOldTemperature(void);
  double getInstantaneousTemperature(void);

  std::uint64_t getLangevinPistonFrictionSeed(void) const;
  unsigned long long int getRngSequencePos(void) const;
  CudaContainer<double> &getReferencePressureTensor(void);
  CudaContainer<double> &getLangevinPistonMass(void);
  CudaContainer<double> &getLangevinPistonOnStepPosition(void);
  CudaContainer<double> &getLangevinPistonHalfStepPosition(void);
  CudaContainer<double> &getLangevinPistonOnStepVelocity(void);
  CudaContainer<double> &getLangevinPistonHalfStepVelocity(void);
  CudaContainer<double> &getLangevinPistonDeltaPosition(void);
  CudaContainer<double> &getLangevinPistonDeltaPositionPrevious(void);
  CudaContainer<double> &getLangevinPistonDeltaPositionPredicted(void);
  CudaContainer<double> &getLangevinPistonDeltaPressure(void);
  CudaContainer<double> &getInstantaneousPressureTensor(void);
  CudaContainer<double> &getInstantaneousPressureScalar(void);
  CudaContainer<double> &getAveragePressureTensor(void);
  CudaContainer<double> &getAveragePressureScalar(void);
  double getInstantaneousSurfaceTension(void);

public:
  void initialize(void) override;
  void initializeFromRestartFile(const std::string &rstFileName) override;
  void propagateOneStep(void) override;

protected:
  double computeNoseHooverPistonMass(void);
  double computeLangevinPistonMass(void);
  void allocateLangevinPistonVariables(void);
  void initializeRng(void);
  void removeCenterOfMassMotion(void);
  void alloc(const int n);
  void dealloc(void);

protected:
  bool m_UsingNoseHooverThermostat;
  double m_ReferenceTemperature;
  CudaContainer<double> m_NoseHooverPistonMass;
  CudaContainer<double> m_NoseHooverPistonVelocity;
  CudaContainer<double> m_NoseHooverPistonVelocityPrevious;
  CudaContainer<double> m_NoseHooverPistonForce;
  CudaContainer<double> m_NoseHooverPistonForcePrevious;
  CudaContainer<double4> m_CoordsDeltaPredicted;

  CudaContainer<double4> m_HolonomicConstraintForces;
  CudaContainer<double> m_HolonomicConstraintVirial;
  CudaContainer<double> m_KineticPressureTensor;
  CudaContainer<double> m_PressureTensor;
  CudaContainer<double> m_PressureScalar;
  CudaContainer<double> m_ReferencePressureTensor;
  CudaContainer<double> m_DeltaPressureTensor;
  CudaContainer<double> m_DeltaKineticPressureTensor;
  CudaContainer<double> m_StaticDeltaPressureTensor;

  CRYSTAL m_CrystalType;
  int m_LangevinPistonDegreesOfFreedom;
  CudaContainer<double> m_LangevinPistonMass;
  CudaContainer<double> m_LangevinPistonInverseMass;
  CudaContainer<double> m_LangevinPistonOnStepPosition;
  CudaContainer<double> m_LangevinPistonHalfStepPosition;
  CudaContainer<double> m_LangevinPistonOnStepVelocity;
  CudaContainer<double> m_LangevinPistonHalfStepVelocity;
  CudaContainer<double> m_LangevinPistonDeltaPosition;
  CudaContainer<double> m_LangevinPistonDeltaPositionPrevious;
  CudaContainer<double> m_LangevinPistonDeltaPositionPredicted;
  CudaContainer<double> m_LangevinPistonDeltaPressure;
  double m_Pgamma;
  double m_Palpha;
  double m_Pbfact;
  CudaContainer<double> m_Prfwd;
  CudaContainer<double> m_OnStepCrystalFactor;
  CudaContainer<double> m_HalfStepCrystalFactor;

  std::uint64_t m_Seed;
  unsigned long long int m_RngSequencePos;
  curandStatePhilox4_32_10_t *m_RngStates;

  bool m_ConstantSurfaceTensionFlag;
  CudaContainer<double> m_SurfaceTension;

  int m_MaxPredictorCorrectorIterations;

  int m_AverageWindowSize;
  CudaContainer<double> m_KineticEnergy;
  CudaContainer<double> m_AverageTemperature;
  CudaContainer<double> m_AveragePressureTensor;
  CudaContainer<double> m_AveragePressureScalar;

  bool m_UsingOldTemperature;
};
