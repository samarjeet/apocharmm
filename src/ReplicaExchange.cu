// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "ReplicaExchange.h"
#include <chrono>
#include <climits>
#include <cpp_utils.h>
#include <iomanip>
#include <iostream>

//
//
ReplicaExchange::ReplicaExchange(
    std::vector<std::shared_ptr<CudaIntegrator>> integrators,
    int _stepsPerExchange, int _numExchanges, std::string _logFileName)
    : integrators(integrators), stepsPerExchange(_stepsPerExchange),
      numExchanges(_numExchanges) {}

void ReplicaExchange::initialize() {
  // for (auto integrator : integrators) {
  //   integrator->initialize();
  // }

  // fill the left exchange vector
  leftExchanges.push_back(integrators.size() - 1);
  for (int i = 1; i < integrators.size(); i++) {
    leftExchanges.push_back(i - 1);
  }

  // fill the right exchange vector
  for (int i = 0; i < integrators.size() - 1; i++) {
    rightExchanges.push_back(i + 1);
  }
  rightExchanges.push_back(0);

  isInitialized = true;
}

void ReplicaExchange::propagate() {
  if (not isInitialized) {
    initialize();
  }

  std::vector<std::shared_ptr<CharmmContext>> contexts;
  for (auto integrator : integrators) {
    contexts.push_back(integrator->getCharmmContext());
  }

  std::vector<float> potentialEnergies(contexts.size());

  for (int i = 0; i < numExchanges; i++) {
    std::cout << "Exchange " << i << std::endl;

    // Choose a exchange scheme
    // choosing left for now
    auto exchange = leftExchanges;

    for (auto integrator : integrators) {
      integrator->propagate(stepsPerExchange);
    }

    int index = 0;
    for (auto context : contexts) {

      context->calculatePotentialEnergy(true, true);
      auto pe = context->getPotentialEnergy();
      pe.transferFromDevice();
      potentialEnergies[index] = pe[0];

      std::cout << "Potential energy " << index << " "
                << potentialEnergies[index] << std::endl;
      index++;
    }

    /*
    auto pe00 = ctx_0->getPotentialEnergy();
    pe00.transferFromDevice();
    std::cout << "Potential energy 0:" << pe00[0] << std::endl;

    auto pe11 = ctx_1->getPotentialEnergy();
    pe11.transferFromDevice();
    std::cout << "Potential energy 1:" << pe11[0] << std::endl;

    auto temp_crd0 = ctx_0->getCoordinates();
    auto temp_crd1 = ctx_1->getCoordinates();

    ctx_0->setCoordinates(temp_crd1);
    ctx_1->setCoordinates(temp_crd0);
    // std::cout << "\n\nSwapped" << std::endl;

    // Now calculate the energies
    ctx_0->calculatePotentialEnergy(true, true);
    ctx_1->calculatePotentialEnergy(true, true);

    auto pe01 = ctx_0->getPotentialEnergy(); // psf 0 and coords 1
    pe01.transferFromDevice();
    std::cout << "Potential energy 0:" << pe01[0] << std::endl;

    auto pe10 = ctx_1->getPotentialEnergy(); // psf 1 and coords 0
    pe10.transferFromDevice();
    std::cout << "Potential energy 1:" << pe10[0] << std::endl;

    // double T = 300;
    // double kT = charmm::constants::kBoltz * T;
    double beta = 1 / kT;
    auto delta = beta * (pe01[0] + pe10[0] - pe00[0] - pe11[0]);
    std::cout << "Delta: " << delta << std::endl;

    double prob = delta <= 0 ? 1 : exp(-delta);
    std::cout << "Probability: " << prob << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    double r = dis(gen);
    std::cout << "Random number: " << r << std::endl;

    if (r < prob) {
      std::cout << "Accepted" << std::endl;
    } else {
      std::cout << "Rejected" << std::endl;
      ctx_0->setCoordinates(temp_crd0);
      ctx_1->setCoordinates(temp_crd1);
    }
    */
  }
}
