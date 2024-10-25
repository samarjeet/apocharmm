// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once
#include "CharmmContext.h"
#include "CudaHolonomicConstraint.h"
#include "Subscriber.h"
#include "XYZQ.h"
#include <map>
#include <memory>

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
  CudaIntegrator(void);

  /** @brief Class constructor. Takes timeStep (in ps) as argument.
   *
   * @param[in] timeStep Time step, in ps
   */
  CudaIntegrator(const double timeStep);

  /** @brief Class constructor. Takes timeStep (ps) as argument, as well as
   * debugPrintFrequency (default 0).
   * @param[in] timeStep Time step, in ps
   * @param[in] debugPrintFrequency Frequency (number of timestep) at which to
   * print integrator infos
   */
  CudaIntegrator(const double timeStep, const int debugPrintFrequency);

  /** @brief Returns integrator timestep (in ps)
   */
  double getTimeStep(void) const;

  /**
   * @brief Set integrator timestep (in ps)
   *
   * Sets timeStep variable to dt * unit time factor.
   * If the Subscriber list is not empty, then calls setTimeStepFromIntegrator
   * for each member.
   */
  void setTimeStep(const double dt);

  /**
   * @brief Link integrator to  CharmmContext
   *
   * Sets context member variable, allocates coordsRef (?), sets
   * holonomicConstraint. If a CharmmContext was already set, throws an
   * exception.
   *
   * @param[in] ctx CharmmContext to be linked
   */
  virtual void setCharmmContext(std::shared_ptr<CharmmContext> ctx);

  const std::shared_ptr<CharmmContext> getCharmmContext(void) const;

  std::shared_ptr<CharmmContext> getCharmmContext(void);

  virtual void initialize(void);

  /**
   * @brief Propagate a single time step
   */
  virtual void propagateOneStep(void);

  /**
   * @brief Propagate given number of steps
   *
   * After each step, checks if one subscriber should be updated by computing
   * modulos with report frequencies of each Subscriber of the attached
   * CharmmContext.
   *
   * @param[in] numSteps Number of steps to propagate
   */
  void propagate(const int numSteps);

  /**
   * @brief Get the Number Of Atoms
   *
   * @return int
   */
  int getNumberOfAtoms(void) const;

  const std::vector<double> &getBoxDimensions(void) const;
  std::vector<double> &getBoxDimensions(void);

  void setDebugPrintFrequency(const int freq);

  void setNonbondedListUpdateFrequency(const int nfreq);

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
  void subscribe(const std::vector<std::shared_ptr<Subscriber>> &sublist);

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
  void unsubscribe(const std::vector<std::shared_ptr<Subscriber>> &sublist);

  /**
   * @brief Return the list of subscribers attached
   */
  const std::vector<std::shared_ptr<Subscriber>> &getSubscribers(void) const;

  /**
   * @brief Return the list of subscribers attached
   */
  std::vector<std::shared_ptr<Subscriber>> &getSubscribers(void);

  /**
   * @brief Return list of all Subscriber frequencies
   */
  const std::vector<int> &getReportFreqList(void) const;

  /**
   * @brief Return list of all Subscriber frequencies
   */
  std::vector<int> &getReportFreqList(void);

  /**
   * @brief Set the Remove Center Of Mass Frequency value
   *
   * @param freq
   */
  void setRemoveCenterOfMassFrequency(const int freq);

  virtual const CudaContainer<double4> &getCoordsDelta(void) const;

  virtual CudaContainer<double4> &getCoordsDelta(void);

  virtual const CudaContainer<double4> &getCoordsDeltaPrevious(void) const;

  virtual CudaContainer<double4> &getCoordsDeltaPrevious(void);

  virtual void
  setCoordsDeltaPrevious(const std::vector<std::vector<double>> &coordsDelta);

  virtual void
  setOnStepPistonVelocity(const CudaContainer<double> &onStepPistonVelocity);

  virtual void
  setOnStepPistonVelocity(const std::vector<double> &onStepPistonVelocity);

  virtual void setHalfStepPistonVelocity(
      const CudaContainer<double> &halfStepPistonVelocity);

  virtual void
  setHalfStepPistonVelocity(const std::vector<double> &halfStepPistonVelocity);

  virtual void
  setOnStepPistonPosition(const CudaContainer<double> &onStepPistonPosition);

  virtual void
  setOnStepPistonPosition(const std::vector<double> &onStepPistonPosition);

  virtual void setHalfStepPistonPosition(
      const CudaContainer<double> &halfStepPistonPosition);

  virtual void
  setHalfStepPistonPosition(const std::vector<double> &halfStepPistonPosition);

  /** @brief Returns a map of the integrator descriptor. Should be overriden by
   * child classes.*/
  virtual std::map<std::string, std::string> getIntegratorDescriptors(void);

  /** @brief Returns the current step which the integrator is on.
   */
  int getCurrentPropagatedStep(void) const;

  ///////////////
  // PROTECTED //
  ///////////////
protected:
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
  void reportIfNeeded(const int istep);

  /** @brief Checks that the kinetic and potential energy are not nans. Throws
   * an error otherwise. Called every min(10^7, min(reportFreqList)) steps.
   * @todo Unittest this  */
  void checkForNanEnergy(void);

protected:
  // double timeStep;
  /** @brief Integrator time step, in AKMA units (ps / timfac) */
  double m_TimeStep;

  /** @brief Time unit conversion factor (t (AKMA) = t (psf) / timfac ) */
  double m_Timfac;

  /** @brief If not 0, frequency at which integrator should print debug infos */
  int m_DebugPrintFrequency;

  /** @brief CharmmContext to which Integrator is attached */
  std::shared_ptr<CharmmContext> m_Context;

  /** @todo Pick a better name : this counts the number of steps and is used to
   * compare with nblupdate frequency, rather than counting how many steps have
   * happened since last NBLupdate */
  int m_StepsSinceNeighborListUpdate;

  /** @brief Allows subscribers to have knowledge of which step the integrator
   * has propgated*/
  int m_CurrentPropagatedStep;

  std::shared_ptr<CudaHolonomicConstraint> m_HolonomicConstraint;

  /**
   * @todo  document this
   */
  CudaContainer<double4> m_CoordsRef;
  CudaContainer<double4> m_CoordsDelta;
  CudaContainer<double4> m_CoordsDeltaPrevious;

  std::shared_ptr<cudaStream_t> m_IntegratorStream;
  std::shared_ptr<cudaStream_t> m_IntegratorMemcpyStream;
  bool m_UsingHolonomicConstraints;

  /**
   * @brief Subscribers linked
   *
   * List of all Subscriber objects linked to the Integrator
   */
  std::vector<std::shared_ptr<Subscriber>> m_Subscribers;

  /**
   * @brief Report frequencies
   *
   * List of all Subscriber report frequencies. Entry #i corresponds to
   * reportFreq of Subscriber #i.
   */
  std::vector<int> m_ReportFreqList;

  /**
   * @brief Flag reporting if CharmmContext object has been set
   */
  bool m_IsCharmmContextSet;

  int m_NonbondedListUpdateFrequency;

  int m_RemoveCenterOfMassFrequency;

  /** @brief Describe the integrator type. Useful to discriminate methods to
   * use, e.g. for restart subscribers. Could/should be a trait */
  std::string m_IntegratorTypeName;
};
