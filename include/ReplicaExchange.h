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
#include "CudaIntegrator.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

class ReplicaExchange {
public:
  // Constructor takes a vector of integrators.
  //

  ReplicaExchange(std::vector<std::shared_ptr<CudaIntegrator>> integrators,
                  int _stepsPerExchange = 1000, int _numExchanges = 1000,
                  std::string _logFileName = "rex.log");

  void initialize();

  // Run the replica exchange simulation.
  void propagate();

private:
  // Integrators for each replica.
  std::vector<std::shared_ptr<CudaIntegrator>> integrators;

  // Number of steps per exchange.
  int stepsPerExchange;

  // Number of exchanges.
  int numExchanges;

  std::string logFileName;

  bool isInitialized = false;

  std::vector<int> leftExchanges, rightExchanges;
};
