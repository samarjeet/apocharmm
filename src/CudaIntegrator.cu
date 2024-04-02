// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "CudaIntegrator.h"
#include "Subscriber.h"
#include "pybind11/pybind11.h"
#include <chrono>
#include <climits>
// #include <experimental/source_location> // C++20
#include <cpp_utils.h>
#include <iomanip>
#include <iostream>
#include <source_location> // C++20
#include <sstream>

namespace py = pybind11;

CudaIntegrator::CudaIntegrator(ts_t timeStep)
    : timeStep{timeStep / 0.0488882129}, timfac{0.0488882129} {

  stepsSinceNeighborListUpdate = -1;

  integratorStream = std::make_shared<cudaStream_t>();
  cudaStreamCreate(integratorStream.get());
  integratorMemcpyStream = std::make_shared<cudaStream_t>();
  cudaStreamCreate(integratorMemcpyStream.get());

  context = nullptr;
  setNonbondedListUpdateFrequency(20);
  setRemoveCenterOfMassFrequency(1000);
}

CudaIntegrator::CudaIntegrator(ts_t timeStep, int debugPrintFrequency = 0)
    : timeStep{timeStep / 0.0488882129}, timfac{0.0488882129} {

  stepsSinceNeighborListUpdate = -1;

  integratorStream = std::make_shared<cudaStream_t>();
  cudaStreamCreate(integratorStream.get());
  integratorMemcpyStream = std::make_shared<cudaStream_t>();
  cudaStreamCreate(integratorMemcpyStream.get());

  context = nullptr;
  debugPrintFrequency = debugPrintFrequency;
  setNonbondedListUpdateFrequency(20);
  setRemoveCenterOfMassFrequency(1000);
}

ts_t CudaIntegrator::getTimeStep() const { return timeStep * timfac; }

void CudaIntegrator::setTimeStep(const ts_t dt) {
  // Converting from ps to AKMA units ltm/consta_ltm
  timeStep = dt / 0.0488882129;
  // If a new time step is set, and there are Subscribers linked to the current
  // integrator, the new timestep should be communicated to the subscribers.
  if (subscribers.size() != 0) {
    for (int i = 0; i < subscribers.size(); i++) {
      subscribers[i]->setTimeStepFromIntegrator(timeStep * timfac);
    }
  }
}

std::shared_ptr<CharmmContext> CudaIntegrator::getCharmmContext(void) {
  return context;
}

void CudaIntegrator::initialize() {
  std::cout << "Should not have been called!!\n";
}

void CudaIntegrator::setCharmmContext(std::shared_ptr<CharmmContext> ctx) {
  if (isCharmmContextSet) {
    throw std::invalid_argument(
        "A CharmmContext object was already set for this CudaIntegrator.\n");
  }
  context = ctx;
  isCharmmContextSet = true;
  if (context->getNumAtoms() < 0) {
    std::stringstream mes;
    mes << "CudaIntegrator: number of atoms is " << context->getNumAtoms()
        << ".\n"
        << "Can't allocate coordsRef with such a size.\n"
        << "  -> No configuration (crd,pdb) was given ?\n";
    throw std::invalid_argument(mes.str());
  }

  coordsRef.allocate(context->getNumAtoms());
  usingHolonomicConstraints = context->isUsingHolonomicConstraints();
  if (usingHolonomicConstraints) {
    holonomicConstraint = std::make_shared<CudaHolonomicConstraint>();
    holonomicConstraint->setCharmmContext(ctx);
    holonomicConstraint->setup(timeStep);
    holonomicConstraint->setStream(integratorStream);
    holonomicConstraint->setMemcpyStream(integratorMemcpyStream);
  }
  initialize();
}

// void CudaIntegrator::setReportSteps(int num) { reportSteps = num; }

void CudaIntegrator::propagate(int numSteps) {
  // Before starting the propagation, check if ForceManager is initialized.
  if (context == nullptr) {
    throw std::invalid_argument(
        "CudaIntegrator::setSimulationContext\nNo CharmmContext object was "
        "set for this CudaIntegrator.\n");
  }
  if (not context->getForceManager()->isInitialized()) {
    throw std::invalid_argument(
        "CudaIntegrator::setSimulationContext\nForceManager is not "
        "initialized. Please call "
        "ForceManager::initialize() before setting the integrator.\n");
  }

  // Logging
  if (false) {
    if (not context->hasLoggerSet()) {
      context->setLogger();
    }
    auto testptr = shared_from_this();
    std::shared_ptr<Logger> currentLogger = context->getLogger();
    currentLogger->updateLog(shared_from_this(), numSteps);
  }

  context->resetNeighborList();

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  for (int i = 0; i < numSteps; ++i) {
    currentPropagatedStep = i;
    // std::cout << "---\nStep " << i << " of " << numSteps << "\n";

    // Capture Ctrl-C SIGINT when running with the python interface
    // if (PyErr_CheckSignals() != 0){
    //  throw py::error_already_set();
    //}

    if (i % 10000 == 0 && i != 0) {
      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      std::chrono::steady_clock::duration duration = end - start;

      std::cout
          << "Step = " << i << " "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 //<< std::chrono::duration_cast<std::chrono::microseconds>(
                 duration)
                 .count()
          << "ms\n";
      start = std::chrono::steady_clock::now();
    }

    if (i % removeCenterOfMassFrequency == 0) {
      // context->removeCenterOfMassMotion();
    }
    propagateOneStep();

    ++stepsSinceNeighborListUpdate;

    int minReportFreq = 100000;

    if (subscribers.size() > 0) {

      // if there are subscribers, find the smallest report freq instead
      for (int j = 0; j < subscribers.size(); j++) {
        if (reportFreqList[j] < minReportFreq) {
          minReportFreq = reportFreqList[j];
        }
      }
    }
    // Check if we have nan-esque energy.
    if (i % minReportFreq == 0) {
      checkForNanEnergy();
    }
    // Check if report is needed for one or more of the subscribers
    reportIfNeeded(i);
  }
}

int CudaIntegrator::getNumberOfAtoms() {
  assert(context != nullptr);
  return context->getNumAtoms();
}

void CudaIntegrator::setNonbondedListUpdateFrequency(int _nfreq) {
  nonbondedListUpdateFrequency = _nfreq;
}

void CudaIntegrator::propagateOneStep() {
  std::cout << "Integrator : override me!\n";
}

// SUBSCRIBER FUNCTIONS

void CudaIntegrator::subscribe(
    const std::vector<std::shared_ptr<Subscriber>> sublist) {
  for (int i = 0; i < sublist.size(); i++) {
    this->subscribe(sublist[i]);
  }
}

void CudaIntegrator::subscribe(std::shared_ptr<Subscriber> sub) {
  subscribers.push_back(sub);
  reportFreqList.push_back(sub->getReportFreq());
  sub->setCharmmContext(context);
  sub->setTimeStepFromIntegrator(timeStep * timfac);

  try {
    sub->setIntegrator(shared_from_this());
  } catch (const std::exception &e) {
    std::cout << "Error : " << e.what() << '\n';
  }
}

void CudaIntegrator::unsubscribe(
    const std::vector<std::shared_ptr<Subscriber>> sublist) {
  for (int i = 0; i < sublist.size(); i++) {
    this->unsubscribe(sublist[i]);
  }
}

void CudaIntegrator::unsubscribe(std::shared_ptr<Subscriber> sub) {
  auto subIterator = std::find(subscribers.begin(), subscribers.end(), sub);
  if (subIterator != subscribers.end()) {
    subscribers.erase(subIterator);
  } else {
    std::stringstream tmpexc;
    tmpexc << "Subscriber not found (file " << sub->getFileName() << ")"
           << std::endl;
    throw std::invalid_argument(tmpexc.str());
  }
  // if you unsubscribe, you should also remove the corresponding freq
  auto freqIterator = std::find(reportFreqList.begin(), reportFreqList.end(),
                                sub->getReportFreq());
  reportFreqList.erase(freqIterator);
}

void CudaIntegrator::reportIfNeeded(int istep) {
  // Loop over each report frequency. If modulo is 0, then update the
  // corresponding Subscriber
  for (int i = 0; i < reportFreqList.size(); i++) {
    if ((istep + 1) % reportFreqList[i] == 0) {
      subscribers[i]->update();
    }
  }
}

void CudaIntegrator::checkForNanEnergy() {
  // Check if we have nan-esque energy
  CudaContainer<double> potEnergyCC = context->getPotentialEnergy();
  potEnergyCC.transferFromDevice();
  double potEnergy = potEnergyCC.getHostArray()[0];
  double kinEnergy = context->getKineticEnergy();

  if (std::isnan(kinEnergy)) {
    throw std::runtime_error("Kinetic energy is NaN");
  }
  if (std::isnan(potEnergy)) {
    throw std::runtime_error("Potential energy is NaN");
  }
}

std::map<std::string, std::string> CudaIntegrator::getIntegratorDescriptors() {
  std::cout
      << "CudaIntegrator::getIntegratorDescriptors() : override me! Returning "
         "base class.\n";
  return {{"IntegratorDescriptor", "CudaIntegrator Baseclass"}};
}

int CudaIntegrator::getCurrentPropagatedStep(void) const {
  return currentPropagatedStep;
}
