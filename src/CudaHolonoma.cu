// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include <iostream>

#include "CudaHolonoma.h"

CudaHolonoma::CudaHolonoma() {}

void CudaHolonoma::setCharmmContext(std::shared_ptr<CharmmContext> context) {
  simulationContext = context;
}

void CudaHolonoma::setup() {
  // just doing it for water molecules for now
  // will get this information from charmmContext

  numSettleMolecules = simulationContext->getNumAtoms() / 3;
  std::cout << "there are " << numSettleMolecules << " settle molecules.\n";
}

void CudaHolonoma::constrainWaterMolecules() {

  std::cout << "Constraining water molecules.\n";
}
