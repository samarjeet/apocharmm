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
#include <random>

/**
 * @brief Uses the BBK algorithm to update crystal dimensions for pressure
 * control
 *
 * @warning To properly use this integrator, one HAS TO call:
 *  - setPistonFriction
 *  - setBathTemperature
 *
 */
class CudaLangevinPistonIntegrator : public CudaIntegrator {
public:
  CudaLangevinPistonIntegrator(const double timeStep);
  CudaLangevinPistonIntegrator(const double timeStep,
                               const CRYSTAL crystalType);
  ~CudaLangevinPistonIntegrator(void);

public: // Setters
  /**
   * @brief Set the pressure along the XX, YY and ZZ axes
   * TODO : modify this to include the full triclinic one
   *
   * @param _referencePressure
   */
  void setPressure(const std::vector<double> &referencePressure);

  /** @brief Sets mass of the Langevin Piston using a vector, allowing for
   * anisotropic barostat
   */
  void setPistonMass(const std::vector<double> &pistonMass);

  /**
   * @brief Set the mass of the NoseHoover piston
   */
  void setNoseHooverPistonMass(const double nhMass);

  void setCrystalType(const CRYSTAL crystalType);

  /**
   * @brief Constant surface tension. Units should be dyne/cm. Since we only
   * have orthorombic box, only Z perpendicular to X-Y are apt. When called,
   * sets constantSurfaceTensionFlag to true.
   *
   * @param st
   */
  void setSurfaceTension(const double st);

  // void setBoxDimensions(const std::vector<double> &boxDimensions);

  /**
   * @brief Set the friction coefficient for the piston degree of freedom
   *
   *  Required ! to run properly, as it sets some of the intermediate values
   *
   * @todo Put computation of pgam, palpha, pbfact, pvfact elsewhere
   *
   * @param _friction
   */
  void setPistonFriction(const double friction);

  /**
   * @brief Set the Bath Temperature of the thermostat. Default: 300K.
   *
   * @param temp
   */
  void setBathTemperature(const double bathTemperature);

  void setNoseHooverFlag(const bool noseHooverFlag);

  void setOnStepPistonVelocity(
      const CudaContainer<double> &onStepPistonVelocity) override;

  void setOnStepPistonVelocity(
      const std::vector<double> &onStepPistonVelocity) override;

  void setHalfStepPistonVelocity(
      const CudaContainer<double> &halfStepPistonVelocity) override;

  void setHalfStepPistonVelocity(
      const std::vector<double> &halfStepPistonVelocity) override;

  void setOnStepPistonPosition(
      const CudaContainer<double> &onStepPistonPosition) override;

  void setOnStepPistonPosition(
      const std::vector<double> &onStepPistonPosition) override;

  void setHalfStepPistonPosition(
      const CudaContainer<double> &halfStepPistonPosition) override;

  void setHalfStepPistonPosition(
      const std::vector<double> &halfStepPistonPosition) override;

  /** @brief Sets coordsDeltaPrevious container, notably used by
   * RestartSubscriber to restart a simulation
   */
  void setCoordsDeltaPrevious(
      const std::vector<std::vector<double>> &coordsDelta) override;

  void setNoseHooverPistonVelocity(const double noseHooverPistonVelocity);

  void setNoseHooverPistonVelocityPrevious(
      const double noseHooverPistonVelocityPrevious);

  void setNoseHooverPistonForce(const double noseHooverPistonForce);

  void
  setNoseHooverPistonForcePrevious(const double noseHooverPistonForcePrevious);

  void setNoseHooverPistonPosition(const double noseHooverPistonPosition);

  /** @brief Set the number of predictor-corrector steps for the LP integrator.
   * Default value is 3
   */
  void setMaxPredictorCorrectorSteps(const int maxPredictorCorrectorSteps);

  /** @brief Sets seed value used to generate the friction. Initializes RNG
   */
  void setSeedForPistonFriction(const uint64_t seed);

public: // Getters
  double getPressureScalar(void) const;

  const std::vector<double> &getPressureTensor(void) const;
  std::vector<double> &getPressureTensor(void);

  double getInstantaneousPressureScalar(void) const;

  const std::vector<double> &getInstantaneousPressureTensor(void) const;
  std::vector<double> &getInstantaneousPressureTensor(void);

  double getPistonMass(void) const;

  CRYSTAL getCrystalType(void) const;

  /**
   * @brief Get the value of the refernece pressure set by the user
   *
   * @return const CudaContainer<double> &
   */
  const CudaContainer<double> &getReferencePressure(void) const;

  /**
   * @brief Get the value of the refernece pressure set by the user
   *
   * @return CudaContainer<double> &
   */
  CudaContainer<double> &getReferencePressure(void);

  int getPistonDegreesOfFreedom(void) const;

  /**
   * @brief Get the Bath Temperature of the thermostat
   *
   * @return double
   */
  double getBathTemperature(void) const;

  double getNoseHooverPistonMass(void) const;
  double getNoseHooverPistonPosition(void) const;
  double getNoseHooverPistonVelocity(void) const;
  double getNoseHooverPistonVelocityPrevious(void) const;
  double getNoseHooverPistonForce(void) const;
  double getNoseHooverPistonForcePrevious(void) const;

  const CudaContainer<double4> &getCoordsDelta(void) const override;
  CudaContainer<double4> &getCoordsDelta(void) override;

  const CudaContainer<double4> &getCoordsDeltaPrevious(void) const override;
  CudaContainer<double4> &getCoordsDeltaPrevious(void) override;

  const CudaContainer<double> &getOnStepPistonVelocity(void) const;
  CudaContainer<double> &getOnStepPistonVelocity(void);

  /** @brief return box dimensions
   */
  // const std::vector<double> &
  // getBoxDimensions(void) const; // { return boxDimensions; }
  // std::vector<double> &getBoxDimensions(void);

  const CudaContainer<double> &getHalfStepPistonVelocity(void) const;
  CudaContainer<double> &getHalfStepPistonVelocity(void);

  const CudaContainer<double> &getOnStepPistonPosition(void) const;
  CudaContainer<double> &getOnStepPistonPosition(void);

  const CudaContainer<double> &getHalfStepPistonPosition(void) const;
  CudaContainer<double> &getHalfStepPistonPosition(void);

  bool hasPistonFrictionSet(void) const;

  uint64_t getSeedForPistonFriction(void) const;

public:
  void initialize(void);

  void propagateOneStep(void) override;

  std::map<std::string, std::string> getIntegratorDescriptors(void) override;

  /** @brief Following CHARMM-GUI heuristics, set the Nose-Hoover dof mass as
   * 2\% of the mass of the system
   */
  double computeNoseHooverPistonMass(void);

private:
  void allocatePistonVariables(void);

  /**
   * @brief Project any quantity from the crystal dimensions  onto the Langevin
   * piston degree(s) of freedom
   */
  // void projectCrystalDimensionsToPistonDof();

  void removeCenterOfMassMotion(void);

  void removeCenterOfMassAverageNetForce(void);

private:
  curandStatePhilox4_32_10_t *m_DevPHILOXStates;
  CudaContainer<double4> m_CoordsDeltaPredicted;

  CudaContainer<double4> m_HolonomicConstraintForces;

  int m_StepsSinceLastReport;
  double m_BathTemperature;

  /** @brief Mass of the Nose-Hoover degree of freedom. Default: 2\% of the
   * system mass
   */
  bool m_NoseHooverFlag;
  double m_NoseHooverPistonMass;
  double m_NoseHooverPistonPosition;
  double m_NoseHooverPistonVelocity;
  double m_NoseHooverPistonVelocityPrevious;
  double m_NoseHooverPistonForce;
  double m_NoseHooverPistonForcePrevious;

  /** @brief Flag. If true, the simulation is run at constant surface tension
   */
  bool m_ConstantSurfaceTensionFlag;

  /**
   * @brief 6 element crystal dimensions.
   * This should go to CharmmContext
   */
  CudaContainer<double> m_CrystalDimensions;
  CudaContainer<double> m_InverseCrystalDimensions;
  CudaContainer<double> m_CrystalDimensionsPrevious;

  /** @brief deltaPressure is the difference between the reference pressure and
   * the (virial + kinetic) pressure. Size 9 (full matrix stored)
   */
  CudaContainer<double> m_DeltaPressure;

  /** @brief deltaPressureNonChanging = deltaPressure - (predicted) next
   * half-step velocity contribution. Size 9 (full matrix stored).
   */
  CudaContainer<double> m_DeltaPressureNonChanging;

  /** @brief (predicted) next half-step velocity contribution to the pressure.
   * Size 9 (full matrix stored)
   */
  CudaContainer<double> m_DeltaPressureHalfStepKinetic;

  /** @brief Mass of the piston degree of freedom. Default: 50 AU
   */
  CudaContainer<double> m_PistonMass;

  /** @brief Inverse of PistonMass (distributed onto piston dofs)
   *
   * @todo Does not have a setter yet ! Only default value now (=50.)
   */
  CudaContainer<double> m_InversePistonMass;

  /** @brief Crystal type. Default: Orthorhombic
   */
  CRYSTAL m_CrystalType; // = CRYSTAL::ORTHORHOMBIC;

  /** @brief
   */
  int m_CrystalDegreesOfFreedom = 3;

  /** @brief Number of degrees of freedom for the piston.
   * @todo Initialize depending on the crystal type.
   */
  int m_PistonDegreesOfFreedom;

  /**
   * @brief deaults to 1 atm along XX, YY, ZZ and 0 otherwise
   * TODO make a setter for this
   */
  CudaContainer<double> m_ReferencePressure;

  /** @brief Friction constant for the piston degree of freedom. Default: 0.0
   */
  double m_Pgamma;

  double m_Pbfact;
  double m_Pvfact; // vars to propagate crystal dof

  /**          1 - .5 * gamma * dt^2
   * palpha =  --------------------
   *           1 + .5 * gamma * dt^2
   */
  double m_Palpha;
  CudaContainer<double> m_Prfwd;

  std::mt19937 m_Rng;

  int m_StepId;

  CudaContainer<double> m_KineticEnergyPressureTensor;
  CudaContainer<double> m_HolonomicVirial;
  CudaContainer<double> m_PressureTensor;

  /** @brief Distance the piston moves in one step or
   * equivalently half step piston velocity * timeStep
   */
  CudaContainer<double> m_PressurePistonPositionDelta;

  /** @brief Distance the piston moved at previous step
   */
  CudaContainer<double> m_PressurePistonPositionDeltaPrevious;

  /** @brief Saved distance the piston moved at previous step
   */
  CudaContainer<double> m_PressurePistonPositionDeltaStored;

  CudaContainer<double> m_OnStepPistonVelocity;
  CudaContainer<double> m_HalfStepPistonVelocity;
  CudaContainer<double> m_OnStepPistonPosition;
  CudaContainer<double> m_HalfStepPistonPosition;

  CudaContainer<double> m_HalfStepKineticEnergy;
  CudaContainer<double> m_HalfStepKineticEnergy1StepPrevious;
  CudaContainer<double> m_HalfStepKineticEnergy2StepsPrevious;
  CudaContainer<double> m_PotentialEnergyPrevious;
  CudaContainer<double> m_HfctenTerm;

  /** @brief crystal_vel / crystal_dim for the onStep and halfStep
   */
  CudaContainer<double> m_OnStepCrystalFactor;
  CudaContainer<double> m_HalfStepCrystalFactor;

  /** @brief deltaPressure projected onto Langevin Piston dof(s)
   */
  CudaContainer<double> m_PistonDeltaPressure;

  /** @brief Number of predictor-corrector steps to predict the LP dof
   * evolution
   */
  int m_MaxPredictorCorrectorSteps; // = 3;

  /**
   * @brief targeted surface tension value. Set by user.
   *
   */
  double m_SurfaceTension;

  CudaContainer<double> m_PressureScalar;

  bool m_PistonFrictionSetFlag;

  /** @brief Seed value for the RNG used for the piston friction
   */
  uint64_t m_Seed;

  CudaContainer<double> m_AveragePressureScalar;
  CudaContainer<double> m_AveragePressureTensor;
};
