// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CudaIntegrator.h"
#include "Subscriber.h"
// #include "pybind11/pybind11.h"
#include <chrono>
#include <climits>
// #include <experimental/source_location> // C++20
#include <cpp_utils.h>
#include <iomanip>
#include <iostream>
#include <source_location> // C++20
#include <sstream>

// namespace py = pybind11;

CudaIntegrator::CudaIntegrator(void)
    : m_TimeStep(0.0), m_Timfac(0.0488882129), m_DebugPrintFrequency(0),
      m_Context(nullptr), m_StepsSinceNeighborListUpdate(-1),
      m_CurrentPropagatedStep(0), m_HolonomicConstraint(nullptr), m_CoordsRef(),
      m_CoordsDelta(), m_CoordsDeltaPrevious(), m_IntegratorStream(nullptr),
      m_IntegratorMemcpyStream(nullptr), m_UsingHolonomicConstraints(false),
      m_Subscribers(), m_ReportFreqList(), m_IsCharmmContextSet(false),
      m_NonbondedListUpdateFrequency(20), m_RemoveCenterOfMassFrequency(1000),
      m_IntegratorTypeName("BaseClass integrator") {
  m_IntegratorStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_IntegratorStream.get()));

  m_IntegratorMemcpyStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_IntegratorMemcpyStream.get()));
}

CudaIntegrator::CudaIntegrator(const double timeStep) : CudaIntegrator() {
  m_TimeStep = timeStep / 0.0488882129;
}

CudaIntegrator::CudaIntegrator(const double timeStep,
                               const int debugPrintFrequency)
    : CudaIntegrator(timeStep) {
  m_DebugPrintFrequency = debugPrintFrequency;
}

double CudaIntegrator::getTimeStep(void) const {
  return (m_TimeStep * m_Timfac);
}

void CudaIntegrator::setTimeStep(const double timeStep) {
  // Converting from ps to AKMA units ltm/consta_ltm
  m_TimeStep = timeStep / 0.0488882129;

  // If a new time step is set, and there are Subscribers linked to the current
  // integrator, the new timestep should be communicated to the subscribers.
  for (std::size_t i = 0; i < m_Subscribers.size(); i++)
    m_Subscribers[i]->setTimeStepFromIntegrator(timeStep);

  return;
}

void CudaIntegrator::setCharmmContext(std::shared_ptr<CharmmContext> ctx) {
  if (m_IsCharmmContextSet) {
    throw std::invalid_argument(
        "A CharmmContext object was already set for this CudaIntegrator.");
  }
  m_Context = ctx;
  m_IsCharmmContextSet = true;
  if (m_Context->getNumAtoms() < 0) {
    throw std::invalid_argument("CudaIntegrator: Number of atoms is " +
                                std::to_string(m_Context->getNumAtoms()) +
                                ".\nCan't allocate memory with such a size.\n "
                                "-> No configuration (crd, pdb) was given?\n");
  }

  m_CoordsRef.resize(m_Context->getNumAtoms());
  m_CoordsDelta.resize(m_Context->getNumAtoms());
  m_CoordsDeltaPrevious.resize(m_Context->getNumAtoms());
  m_UsingHolonomicConstraints = m_Context->isUsingHolonomicConstraints();
  if (m_UsingHolonomicConstraints) {
    m_HolonomicConstraint = std::make_shared<CudaHolonomicConstraint>();
    m_HolonomicConstraint->setCharmmContext(ctx);
    m_HolonomicConstraint->setup(m_TimeStep);
    m_HolonomicConstraint->setStream(m_IntegratorStream);
    m_HolonomicConstraint->setMemcpyStream(m_IntegratorMemcpyStream);
  }
  this->initialize();

  return;
}

const std::shared_ptr<CharmmContext>
CudaIntegrator::getCharmmContext(void) const {
  return m_Context;
}

std::shared_ptr<CharmmContext> CudaIntegrator::getCharmmContext(void) {
  return m_Context;
}

void CudaIntegrator::initialize(void) {
  std::cerr << "CudaIntegrator::initialize() : override me!" << std::endl;
  exit(1);
  return;
}

void CudaIntegrator::propagateOneStep() {
  std::cout << "CudaIntegrator::propagateOneStep() : override me!" << std::endl;
  exit(1);
  return;
}

void CudaIntegrator::propagate(const int numSteps) {
  // Before starting the propagation, check if ForceManager is initialized.
  if (m_Context == nullptr) {
    throw std::invalid_argument(
        "CudaIntegrator::setSimulationContext\nNo CharmmContext object was "
        "set for this CudaIntegrator.\n");
  }
  if (not m_Context->getForceManager()->isInitialized()) {
    throw std::invalid_argument(
        "CudaIntegrator::setSimulationContext\nForceManager is not "
        "initialized. Please call "
        "ForceManager::initialize() before setting the integrator.\n");
  }

  // Logging
  if (false) {
    if (not m_Context->hasLoggerSet())
      m_Context->setLogger();
    auto testptr = this->shared_from_this();
    std::shared_ptr<Logger> currentLogger = m_Context->getLogger();
    currentLogger->updateLog(this->shared_from_this(), numSteps);
  }

  m_Context->resetNeighborList();

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  m_StepsSinceNeighborListUpdate = 1;

  for (int step = 1; step <= numSteps; step++) {
    m_CurrentPropagatedStep = step;
    // std::cout << "---\nStep " << step << " of " << numSteps << "\n";

    // Capture Ctrl-C SIGINT when running with the python interface
    // if (PyErr_CheckSignals() != 0){
    //  throw py::error_already_set();
    //}

    if (step % 10000 == 0) {
      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      std::chrono::steady_clock::duration duration = end - start;

      std::cout
          << "Step = " << step << " "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 //<< std::chrono::duration_cast<std::chrono::microseconds>(
                 duration)
                 .count()
          << "ms" << std::endl;
      start = std::chrono::steady_clock::now();
    }

    // if (step % removeCenterOfMassFrequency == 0) {
    //   context->removeCenterOfMassMotion();
    // }
    this->propagateOneStep();

    m_StepsSinceNeighborListUpdate++;

    int minReportFreq = 100000;

    // if there are subscribers, find the smallest report freq instead
    for (std::size_t i = 0; i < m_Subscribers.size(); i++) {
      if (m_ReportFreqList[i] < minReportFreq)
        minReportFreq = m_ReportFreqList[i];
    }
    // Check if we have nan-esque energy.
    if (step % minReportFreq == 0)
      this->checkForNanEnergy();

    // Check if report is needed for one or more of the subscribers
    this->reportIfNeeded(step);
  }

  return;
}

int CudaIntegrator::getNumberOfAtoms(void) const {
  assert(m_Context != nullptr);
  return m_Context->getNumAtoms();
}

const std::vector<double> &CudaIntegrator::getBoxDimensions(void) const {
  assert(m_Context != nullptr);
  return m_Context->getBoxDimensions();
}

std::vector<double> &CudaIntegrator::getBoxDimensions(void) {
  assert(m_Context != nullptr);
  return m_Context->getBoxDimensions();
}

void CudaIntegrator::setDebugPrintFrequency(const int freq) {
  m_DebugPrintFrequency = freq;
  return;
}

void CudaIntegrator::setNonbondedListUpdateFrequency(const int nfreq) {
  m_NonbondedListUpdateFrequency = nfreq;
  return;
}

void CudaIntegrator::subscribe(std::shared_ptr<Subscriber> sub) {
  m_Subscribers.push_back(sub);
  m_ReportFreqList.push_back(sub->getReportFreq());
  sub->setCharmmContext(m_Context);
  sub->setTimeStepFromIntegrator(m_TimeStep * m_Timfac);

  try {
    sub->setIntegrator(this->shared_from_this());
  } catch (const std::exception &e) {
    std::cout << "Error : " << e.what() << '\n';
  }
}

void CudaIntegrator::subscribe(
    const std::vector<std::shared_ptr<Subscriber>> &sublist) {
  for (std::size_t i = 0; i < sublist.size(); i++) {
    this->subscribe(sublist[i]);
  }
}

void CudaIntegrator::unsubscribe(std::shared_ptr<Subscriber> sub) {
  auto subIterator = std::find(m_Subscribers.begin(), m_Subscribers.end(), sub);
  if (subIterator != m_Subscribers.end())
    m_Subscribers.erase(subIterator);
  else {
    // std::stringstream tmpexc;
    // tmpexc << "Subscriber not found (file " << sub->getFileName() << ")"
    //        << std::endl;
    // throw std::invalid_argument(tmpexc.str());
    throw std::invalid_argument("Subscriber not found (file \"" +
                                sub->getFileName() + "\")");
  }
  // if you unsubscribe, you should also remove the corresponding freq
  auto freqIterator = std::find(m_ReportFreqList.begin(),
                                m_ReportFreqList.end(), sub->getReportFreq());
  m_ReportFreqList.erase(freqIterator);

  return;
}

void CudaIntegrator::unsubscribe(
    const std::vector<std::shared_ptr<Subscriber>> &sublist) {
  for (std::size_t i = 0; i < sublist.size(); i++)
    this->unsubscribe(sublist[i]);
  return;
}

const std::vector<std::shared_ptr<Subscriber>> &
CudaIntegrator::getSubscribers(void) const {
  return m_Subscribers;
}

std::vector<std::shared_ptr<Subscriber>> &CudaIntegrator::getSubscribers(void) {
  return m_Subscribers;
}

const std::vector<int> &CudaIntegrator::getReportFreqList(void) const {
  return m_ReportFreqList;
}

std::vector<int> &CudaIntegrator::getReportFreqList(void) {
  return m_ReportFreqList;
}

void CudaIntegrator::setRemoveCenterOfMassFrequency(const int freq) {
  m_RemoveCenterOfMassFrequency = freq;
  return;
}

const CudaContainer<double4> &CudaIntegrator::getCoordsDelta(void) const {
  std::cerr << "CudaIntegrator::getCoordsDelta() : override me!" << std::endl;
  exit(1);
  return m_CoordsDelta;
}

CudaContainer<double4> &CudaIntegrator::getCoordsDelta(void) {
  std::cerr << "CudaIntegrator::getCoordsDelta() : override me!" << std::endl;
  exit(1);
  return m_CoordsDelta;
}

const CudaContainer<double4> &
CudaIntegrator::getCoordsDeltaPrevious(void) const {
  std::cerr << "CudaIntegrator::getCoordsDeltaPrevious() : override me!"
            << std::endl;
  exit(1);
  return m_CoordsDeltaPrevious;
}

CudaContainer<double4> &CudaIntegrator::getCoordsDeltaPrevious(void) {
  std::cerr << "CudaIntegrator::getCoordsDeltaPrevious() : override me!"
            << std::endl;
  exit(1);
  return m_CoordsDeltaPrevious;
}

void CudaIntegrator::setCoordsDeltaPrevious(
    const std::vector<std::vector<double>> &coordsDelta) {
  std::cerr << "CudaIntegrator::setCoordsDeltaPrevious() : override me!"
            << std::endl;
  exit(1);
  return;
}

void CudaIntegrator::setOnStepPistonVelocity(
    const CudaContainer<double> &onStepPistonVelocity) {
  std::cerr << "CudaIntegrator::setOnStepPistonVelocity() : override me!"
            << std::endl;
  exit(1);
  return;
}

void CudaIntegrator::setOnStepPistonVelocity(
    const std::vector<double> &onStepPistonVelocity) {
  std::cerr << "CudaIntegrator::setOnStepPistonVelocity() : override me!"
            << std::endl;
  exit(1);
  return;
}

void CudaIntegrator::setHalfStepPistonVelocity(
    const CudaContainer<double> &halfStepPistonVelocity) {
  std::cerr << "CudaIntegrator::setHalfStepPistonVelocity() : override me!"
            << std::endl;
  exit(1);
  return;
}

void CudaIntegrator::setHalfStepPistonVelocity(
    const std::vector<double> &halfStepPistonVelocity) {
  std::cerr << "CudaIntegrator::setHalfStepPistonVelocity() : override me!"
            << std::endl;
  exit(1);
  return;
}

void CudaIntegrator::setOnStepPistonPosition(
    const CudaContainer<double> &onStepPistonPosition) {
  std::cerr << "CudaIntegrator::setOnStepPistonPosition() : override me!"
            << std::endl;
  exit(1);
  return;
}

void CudaIntegrator::setOnStepPistonPosition(
    const std::vector<double> &onStepPistonPosition) {
  std::cerr << "CudaIntegrator::setOnStepPistonPosition() : override me!"
            << std::endl;
  exit(1);
  return;
}

void CudaIntegrator::setHalfStepPistonPosition(
    const CudaContainer<double> &halfStepPistonPosition) {
  std::cerr << "CudaIntegrator::setHalfStepPistonPosition() : override me!"
            << std::endl;
  exit(1);
  return;
}

void CudaIntegrator::setHalfStepPistonPosition(
    const std::vector<double> &halfStepPistonPosition) {
  std::cerr << "CudaIntegrator::setHalfStepPistonPosition() : override me!"
            << std::endl;
  exit(1);
  return;
}

std::map<std::string, std::string>
CudaIntegrator::getIntegratorDescriptors(void) {
  std::cerr << "CudaIntegrator::getIntegratorDescriptors() : override me!"
            << std::endl;
  exit(1);
  return {{"IntegratorDescriptor", "CudaIntegrator Baseclass"}};
}

int CudaIntegrator::getCurrentPropagatedStep(void) const {
  return m_CurrentPropagatedStep;
}

void CudaIntegrator::reportIfNeeded(const int istep) {
  // Loop over each report frequency. If modulo is 0, then update the
  // corresponding Subscriber
  for (std::size_t i = 0; i < m_ReportFreqList.size(); i++) {
    if ((istep + 1) % m_ReportFreqList[i] == 0)
      m_Subscribers[i]->update();
  }
  return;
}

void CudaIntegrator::checkForNanEnergy(void) {
  // Check if we have nan-esque energy
  CudaContainer<double> &potEnergyCC = m_Context->getPotentialEnergy();
  potEnergyCC.transferFromDevice();
  double potEnergy = potEnergyCC.getHostArray()[0];
  double kinEnergy = m_Context->getKineticEnergy();

  if (std::isnan(kinEnergy))
    throw std::runtime_error("Kinetic energy is NaN");
  if (std::isnan(potEnergy))
    throw std::runtime_error("Potential energy is NaN");

  return;
}
