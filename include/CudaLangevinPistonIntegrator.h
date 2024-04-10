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
private:
  // double forceScale;
  // double velScale;
  // double noiseScale;

  //** @brief Friction used for the crystal dofs */
  // double friction;

  std::vector<double> boxDimensions;

  curandStatePhilox4_32_10_t *devPHILOXStates;
  // CudaContainer<double4> coordsDeltaPrevious;
  CudaContainer<double4> coordsDeltaPredicted;

  CudaContainer<double4> holonomicConstraintForces;

  int stepsSinceLastReport;
  double bathTemperature;

  void allocatePistonVariables();

  /**
   * @brief 6 element crystal dimensions.
   * This should go to CharmmContext
   */
  CudaContainer<double> crystalDimensions, inverseCrystalDimensions;

  CudaContainer<double> crystalDimensionsPrevious;

  /** @brief deltaPressure is the difference between the reference pressure and
   * the (virial + kinetic) pressure. Size 9 (full matrix stored)
   */
  CudaContainer<double> deltaPressure;
  /** @brief deltaPressureNonChanging = deltaPressure - (predicted) next
   * half-step velocity contribution. Size 9 (full matrix stored).
   */
  CudaContainer<double> deltaPressureNonChanging;
  /** @brief (predicted) next half-step velocity contribution to the pressure.
   * Size 9 (full matrix stored)
   */
  CudaContainer<double> deltaPressureHalfStepKinetic;

  /** @brief Inverse of PistonMass (distributed onto piston dofs)
   *
   * @todo Does not have a setter yet ! Only default value now (=50.)
   */
  CudaContainer<double> inversePistonMass;

  // CudaContainer<double> deltaPressureVelocity;
  // CudaContainer<double> randomForceScaleFactor;

  /** @brief Crystal type. Default: Orthorhombic */
  CRYSTAL crystalType = CRYSTAL::ORTHORHOMBIC;

  /* @brief
   */
  int crystalDegreesOfFreedom = 3;

  /** @brief Number of degrees of freedom for the piston.
   * @todo Initialize depending on the crystal type.
   */
  int pistonDegreesOfFreedom;

  /**
   * @brief deaults to 1 atm along XX, YY, ZZ and 0 otherwise
   * TODO make a setter for this
   */
  CudaContainer<double> referencePressure;

  /** @brief Mass of the piston degree of freedom. Default: 50 AU  */
  std::vector<double> pistonMass;

  /** @brief Mass of the Nose-Hoover degree of freedom. Default: 2\% of the
   * system mass
   */
  double noseHooverPistonMass;

  double noseHooverPistonPosition, noseHooverPistonVelocity,
      noseHooverPistonVelocityPrevious, noseHooverPistonForce,
      noseHooverPistonForcePrevious;

  /** @brief Friction constant for the piston degree of freedom. Default: 0.0 */
  double pgamma;

  double pbfact, pvfact; // vars to propagate crystal dof
  /**          1 - .5 * gamma * dt^2
   * palpha =  --------------------
   *           1 + .5 * gamma * dt^2
   */
  double palpha;
  std::vector<double> prfwd;

  std::mt19937 rng;

  bool noseHooverFlag;

  /** @brief Flag. If true, the simulation is run at constant surface tension */
  bool constantSurfaceTensionFlag;

  CudaContainer<double> kineticEnergyPressureTensor, holonomicVirial,
      pressureTensor;

  /** @brief Distance the piston moves in one step or
   * equivalently half step piston velocity * timeStep
   */
  CudaContainer<double> pressurePistonPositionDelta;
  /** @brief Distance the piston moved at previous step */
  CudaContainer<double> pressurePistonPositionDeltaPrevious;
  /** @brief Saved distance the piston moved at previous step */
  CudaContainer<double> pressurePistonPositionDeltaStored;

  CudaContainer<double> onStepPistonVelocity, halfStepPistonVelocity,
      onStepPistonPosition, halfStepPistonPosition;

  CudaContainer<double> halfStepKineticEnergy,
      halfStepKineticEnergy1StepPrevious, halfStepKineticEnergy2StepsPrevious,
      potentialEnergyPrevious;
  CudaContainer<double> hfctenTerm;

  /** @brief crystal_vel / crystal_dim for the onStep and halfStep */
  CudaContainer<double> onStepCrystalFactor, halfStepCrystalFactor;

  /** @brief deltaPressure projected onto Langevin Piston dof(s) */
  CudaContainer<double> pistonDeltaPressure;

  /** @brief Number of predictor-corrector steps to predict the LP dof
   * evolution */
  int maxPredictorCorrectorSteps = 3;

  /**
   * @brief Project any quantity from the crystal dimensions  onto the Langevin
   * piston degree(s) of freedom
   */
  // void projectCrystalDimensionsToPistonDof();
  int stepId;

  /**
   * @brief targeted surface tension value. Set by user.
   *
   */
  double surfaceTension;

  CudaContainer<double> pressureScalar;

  void removeCenterOfMassMotion();

  bool pistonFrictionSetFlag;

  /** @brief Seed value for the RNG used for the piston friction */
  uint64_t seed;

  void removeCenterOfMassAverageNetForce();

  ////////////
  // PUBLIC //
  ////////////
public:
  CudaLangevinPistonIntegrator(ts_t timeStep);
  CudaLangevinPistonIntegrator(ts_t timeStep, CRYSTAL _crystalType);
  ~CudaLangevinPistonIntegrator();

  /**
   * @brief Set the pressure along the XX, YY and ZZ axes
   * TODO : modify this to include the full triclinic one
   *
   * @param _referencePressure
   */
  void setPressure(std::vector<double> _referencePressure);

  double getPressureScalar();

  std::vector<double> getPressureTensor();

  double getInstantaneousPressureScalar();

  std::vector<double> getInstantaneousPressureTensor();

  /**
   * @brief Set the mass of the piston
   * I'm forcefully deprecating this.
   */
  // void setPistonMass(double _pistonMass);

  double getPistonMass() { return pistonMass[0]; }

  /** @brief Sets mass of the Langevin Piston using a vector, allowing for
   * anisotropic barostat */
  void setPistonMass(std::vector<double> _pistonMass);

  /**
   * @brief Set the mass of the NoseHoover piston
   */
  void setNoseHooverPistonMass(double _nhMass);

  void setCrystalType(CRYSTAL _crystalType);

  CRYSTAL getCrystalType(void) const;

  /**
   * @brief Constant surface tension. Units should be dyne/cm. Since we only
   * have orthorombic box, only Z perpendicular to X-Y are apt. When called,
   * sets constantSurfaceTensionFlag to true.
   *
   * @param st
   */
  void setSurfaceTension(double st);

  /**
   * @brief Get the value of the refernece pressure set by the user
   *
   * @return std::vector<double>
   */
  std::vector<double> getReferencePressure();

  int getPistonDegreesOfFreedom() { return pistonDegreesOfFreedom; }

  // Put these in the base class
  // void setContext();
  void initialize();

  void setBoxDimensions(std::vector<double> boxDimensionsOriginal);

  /**
   * @brief Set the friction coefficient for the piston degree of freedom
   *
   *  Required ! to run properly, as it sets some of the intermediate values
   *
   * @todo Put computation of pgam, palpha, pbfact, pvfact elsewhere
   *
   * @param _friction
   */
  void setPistonFriction(double _friction);
  /**
   * @brief Set the Bath Temperature of the thermostat. Default: 300K.
   *
   * @param temp
   */
  void setBathTemperature(double temp) { bathTemperature = temp; }

  /**
   * @brief Get the Bath Temperature of the thermostat
   *
   * @return double
   */
  // double getBathTemperature() const { return bathTemperature; }

  void propagateOneStep() override;

  double getNoseHooverPistonMass() { return noseHooverPistonMass; }

  void setNoseHooverFlag(bool _noseHooverFlag) {
    noseHooverFlag = _noseHooverFlag;
  }

  double getNoseHooverPistonPosition() { return noseHooverPistonPosition; }
  double getNoseHooverPistonVelocity() { return noseHooverPistonVelocity; }
  double getNoseHooverPistonForce() { return noseHooverPistonForce; }

  CudaContainer<double4> getCoordsDelta() override;
  CudaContainer<double4> getCoordsDeltaPrevious() override;
  // std::vector<std::vector<double>> getCoordsDeltaPrevious() override;

  CudaContainer<double> averagePressureScalar;
  CudaContainer<double> averagePressureTensor;

  CudaContainer<double> getOnStepPistonVelocity() {
    return onStepPistonVelocity;
  }

  /** @brief return box dimensions */
  std::vector<double> getBoxDimensions() { return boxDimensions; }

  void setOnStepPistonVelocity(
      CudaContainer<double> _onStepPistonVelocity) override {
    onStepPistonVelocity.set(_onStepPistonVelocity.getHostArray());
  }

  void setOnStepPistonVelocity(
      const std::vector<double> _onStepPistonVelocity) override {
    onStepPistonVelocity.set(_onStepPistonVelocity);
  }

  void setHalfStepPistonVelocity(
      CudaContainer<double> _halfStepPistonVelocity) override {
    halfStepPistonVelocity.set(_halfStepPistonVelocity.getHostArray());
  }

  void setHalfStepPistonVelocity(
      const std::vector<double> _halfStepPistonVelocity) override {
    halfStepPistonVelocity.set(_halfStepPistonVelocity);
  }

  CudaContainer<double> getHalfStepPistonVelocity() {
    return halfStepPistonVelocity;
  }

  CudaContainer<double> getOnStepPistonPosition() {
    return onStepPistonPosition;
  }

  void setOnStepPistonPosition(
      CudaContainer<double> _onStepPistonPosition) override {
    onStepPistonPosition.set(_onStepPistonPosition.getHostArray());
  }

  void setOnStepPistonPosition(
      const std::vector<double> _onStepPistonPosition) override {
    onStepPistonPosition.set(_onStepPistonPosition);
  }

  CudaContainer<double> getHalfStepPistonPosition() {
    return halfStepPistonPosition;
  }

  void setHalfStepPistonPosition(
      CudaContainer<double> _halfStepPistonPosition) override {
    halfStepPistonPosition.set(_halfStepPistonPosition.getHostArray());
  }

  void setHalfStepPistonPosition(
      const std::vector<double> _halfStepPistonPosition) override {
    halfStepPistonPosition.set(_halfStepPistonPosition);
  }

  /** @brief Sets coordsDeltaPrevious container, notably used by
   * RestartSubscriber to restart a simulation */
  void setCoordsDeltaPrevious(
      std::vector<std::vector<double>> _coordsDelta) override;

  double getNoseHooverPistonVelocityPrevious() const {
    return noseHooverPistonVelocityPrevious;
  }
  double getNoseHooverPistonForcePrevious() const {
    return noseHooverPistonForcePrevious;
  }
  double getNoseHooverPistonPosition() const {
    return noseHooverPistonPosition;
  }
  void setNoseHooverPistonVelocity(double _noseHooverPistonVelocity) {
    noseHooverPistonVelocity = _noseHooverPistonVelocity;
  }
  void setNoseHooverPistonVelocityPrevious(
      double _noseHooverPistonVelocityPrevious) {
    noseHooverPistonVelocityPrevious = _noseHooverPistonVelocityPrevious;
  }
  void setNoseHooverPistonForce(double _noseHooverPistonForce) {
    noseHooverPistonForce = _noseHooverPistonForce;
  }
  void setNoseHooverPistonForcePrevious(double _noseHooverPistonForcePrevious) {
    noseHooverPistonForcePrevious = _noseHooverPistonForcePrevious;
  }
  void setNoseHooverPistonPosition(double _noseHooverPistonPosition) {
    noseHooverPistonPosition = _noseHooverPistonPosition;
  }

  /** @brief Set the number of predictor-corrector steps for the LP integrator.
   * Default value is 3 */
  void setMaxPredictorCorrectorSteps(int _maxPredictorCorrectorSteps) {
    maxPredictorCorrectorSteps = _maxPredictorCorrectorSteps;
  }

  std::map<std::string, std::string> getIntegratorDescriptors() override;

  bool hasPistonFrictionSet() { return pistonFrictionSetFlag; }

  /** @brief Sets seed value used to generate the friction. Initializes RNG */
  void setSeedForPistonFriction(uint64_t _seed) {
    seed = _seed;
    rng.seed(_seed);
  }
  uint64_t getSeedForPistonFriction() { return seed; }

  /** @brief Following CHARMM-GUI heuristics, set the Nose-Hoover dof mass as
   * 2\% of the mass of the system */
  double computeNoseHooverPistonMass();
};
