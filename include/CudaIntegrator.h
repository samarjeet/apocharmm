// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include "CharmmContext.h"
#include "CudaHolonomicConstraint.h"
#include "Subscriber.h"
#include "XYZQ.h"
#include <map>
#include <memory>

typedef double ts_t;

/**
 * @brief Base class for integrators
 * @todo Add CudaHolonomicConstraints as a member here
 * @todo should timeStep be float/double?
 *
 * Base class to build integrators.While propagating, CudaIntegrator will
 * notify the right Subscriber object (through the CharmmContext) using this
 * reportFreqList as an index.
 *
 */
class CudaIntegrator : public std::enable_shared_from_this<CudaIntegrator> {

  ////////////
  // PUBLIC //
  ////////////
public:
  /** @brief Class constructor. Takes timeStep (in ps) as argument.
   *
   * @param[in] timeStep Time step, in ps
   */
  CudaIntegrator(ts_t timeStep);
  /** @brief Class constructor. Takes timeStep (ps) as argument, as well as
   * debugPrintFrequency (default 0).
   * @param[in] timeStep Time step, in ps
   * @param[in] debugPrintFrequency Frequency (number of timestep) at which to
   * print integrator infos
   */
  CudaIntegrator(ts_t timeStep, int debugPrintFrequency);
  /** @brief Returns integrator timestep (in ps)
   */
  ts_t getTimeStep() const;

  /**
   * @brief Set integrator timestep (in ps)
   *
   * Sets timeStep variable to dt * unit time factor.
   * If the Subscriber list is not empty, then calls setTimeStepFromIntegrator
   * for each member.
   */
  void setTimeStep(const ts_t dt);

  /**
   * @brief Link integrator to  CharmmContext
   *
   * Sets context member variable, allocates coordsRef (?), sets
   * holonomicConstraint. If a CharmmContext was already set, throws an
   * exception.
   *
   * @param[in] ctx CharmmContext to be linked
   */
  virtual void setSimulationContext(std::shared_ptr<CharmmContext> ctx);

  std::shared_ptr<CharmmContext> getSimulationContext();

  // remove this
  // void initializeOldNewCoords(int numAtoms);

  // void setReportSteps(int num);
  virtual void initialize();

  /**
   * @brief Propagate a single time step
   */
  virtual void propagateOneStep();
  /**
   * @brief Propagate given number of steps
   *
   * After each step, checks if one subscriber should be updated by computing
   * modulos with report frequencies of each Subscriber of the attached
   * CharmmContext.
   *
   * @param[in] numSteps Number of steps to propagate
   */
  void propagate(int numSteps);
  // void useHolonomicConstraints(bool set);

  /**
   * @brief Get the Number Of Atoms
   *
   * @return int
   */
  int getNumberOfAtoms();

  void setDebugPrintFrequency(int freq) { debugPrintFrequency = freq; }

  void setNonbondedListUpdateFrequency(int _nfreq);

  // SUBSCRIBER FUNCTIONS
  //======================
  /**
   * @brief Add a Subscriber
   * @param[in] sub Subscriber
   *
   * Appends a Subscriber to the subscribers list, appends its report frequency
   * to reportFreqList.
   */
  void subscribe(std::shared_ptr<Subscriber> sub);
  /**
   * @brief Add a list of Subscribers
   * @param[in] sublist Subscriber vector (list of subscribers)
   *
   * Appends a vector of Subscribers to the subscribers list, appends their
   * respective report frequency to reportFreqList.
   */
  void subscribe(std::vector<std::shared_ptr<Subscriber>> sublist);
  /**
   * @brief Remove a Subscriber
   *
   * @param[in] sub Subscriber to be removed from the subscribers list
   */
  void unsubscribe(std::shared_ptr<Subscriber> sub);
  /**
   * @brief Remove a list of Subscribers
   * @param[in] sublist Subscriber vector (list of subscribers) to be removed
   */
  void unsubscribe(std::vector<std::shared_ptr<Subscriber>> sublist);

  /**
   * @brief Return the list of subscribers attached
   */
  std::vector<std::shared_ptr<Subscriber>> getSubscribers() {
    return subscribers;
  }

  /**
   * @brief Return list of all Subscriber frequencies
   */
  std::vector<int> getReportFreqList() { return reportFreqList; }

  /**
   * @brief Set the Remove Center Of Mass Frequency value
   *
   * @param freq
   */
  void setRemoveCenterOfMassFrequency(int freq) {
    removeCenterOfMassFrequency = freq;
  }

  virtual CudaContainer<double4> getCoordsDelta() {
    std::cerr << "CudaIntegrator::getCoordsDelta() : override me!\n";
    exit(1);
    return CudaContainer<double4>();
  }
  virtual CudaContainer<double4> getCoordsDeltaPrevious() {
    std::cerr << "CudaIntegrator::getCoordsDeltaPrevious() : override me!\n";
    exit(1);
    return CudaContainer<double4>();
  }

  virtual void
  setCoordsDeltaPrevious(const std::vector<std::vector<double>> _coordsDelta) {
    std::cerr << "CudaIntegrator::setCoordsDeltaPrevious() : override me!\n";
    exit(1);
  }
  virtual void
  setOnStepPistonVelocity(CudaContainer<double> _onStepPistonVelocity) {
    std::cerr << "CudaIntegrator::setOnStepPistonVelocity() : override me!\n";
    exit(1);
  }
  virtual void
  setOnStepPistonVelocity(const std::vector<double> _onStepPistonVelocity) {
    std::cerr << "CudaIntegrator::setOnStepPistonVelocity() : override me!\n";
    exit(1);
  }
  virtual void
  setOnStepPistonPosition(CudaContainer<double> _onStepPistonPosition) {
    std::cerr << "CudaIntegrator::setOnStepPistonPosition() : override me!\n";
    exit(1);
  }
  virtual void
  setOnStepPistonPosition(const std::vector<double> _onStepPistonPosition) {
    std::cerr << "CudaIntegrator::setOnStepPistonPosition() : override me!\n";
    exit(1);
  }
  virtual void
  setHalfStepPistonPosition(CudaContainer<double> _halfStepPistonPosition) {
    std::cerr << "CudaIntegrator::setHalfStepPistonPosition() : override me!\n";
    exit(1);
  }
  virtual void
  setHalfStepPistonPosition(const std::vector<double> _halfStepPistonPosition) {
    std::cerr << "CudaIntegrator::setHalfStepPistonPosition() : override me!\n";
    exit(1);
  }

  /** @brief Returns a map of the integrator descriptor. Should be overriden by
   * child classes.*/
  virtual std::map<std::string, std::string> getIntegratorDescriptors();

  /** @brief Returns the current step which the integrator is on.
   */
  int getCurrentPropagatedStep(void) const;

  ///////////////
  // PROTECTED //
  ///////////////
protected:
  // double timeStep;
  /** @brief Integrator time step, in AKMA units (ps / timfac) */
  ts_t timeStep;
  /** @brief Time unit conversion factor (t (AKMA) = t (psf) / timfac ) */
  double timfac;

  /** @brief If not 0, frequency at which integrator should print debug infos */
  int debugPrintFrequency = 0;

  /** @brief CharmmContext to which Integrator is attached */
  std::shared_ptr<CharmmContext> context = nullptr;

  /** @todo Pick a better name : this counts the number of steps and is used to
   * compare with nblupdate frequency, rather than counting how many steps have
   * happened since last NBLupdate */
  int stepsSinceNeighborListUpdate;

  /** @brief Allows subscribers to have knowledge of which step the integrator
   * has propgated*/
  int currentPropagatedStep;

  std::shared_ptr<CudaHolonomicConstraint> holonomicConstraint;

  /**
   * @todo  document this
   */
  CudaContainer<double4> coordsRef, coordsDelta;

  std::shared_ptr<cudaStream_t> integratorStream, integratorMemcpyStream;
  bool usingHolonomicConstraints;

  /**
   * @brief Returns indices of Subscriber needing update
   *
   * @param[in] istep current timestep number
   *
   * Computes modulo(i,reportFreq) for each member of reportFreqList. Returns
   * list of all indices for which the modulo is 0.
   *
   * @return List of indices of Subscriber to be updated (can be empty)
   */
  void reportIfNeeded(int istep);

  /**
   * @brief Subscribers linked
   *
   * List of all Subscriber objects linked to the Integrator
   */
  std::vector<std::shared_ptr<Subscriber>> subscribers;

  /**
   * @brief Report frequencies
   *
   * List of all Subscriber report frequencies. Entry #i corresponds to
   * reportFreq of Subscriber #i.
   */
  std::vector<int> reportFreqList;

  /**
   * @brief Flag reporting if CharmmContext object has been set
   */
  bool isCharmmContextSet = false;

  int nonbondedListUpdateFrequency;

  /** @brief Checks that the kinetic and potential energy are not nans. Throws
   * an error otherwise. Called every min(10^7, min(reportFreqList)) steps.
   * @todo Unittest this  */
  void checkForNanEnergy();
  int removeCenterOfMassFrequency;

  /** @brief Describe the integrator type. Useful to discriminate methods to
   * use, e.g. for restart subscribers. Could/should be a trait */
  std::string integratorTypeName = "BaseClass integrator";
};
