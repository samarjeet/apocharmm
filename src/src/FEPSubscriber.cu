// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, Felix Aviat
//
// ENDLICENSE

#include "BEDSForceManager.h"
#include "CharmmContext.h"
#include "FEPEIForceManager.h"
#include "FEPSubscriber.h"
#include <fstream>
#include <iostream>

FEPSubscriber::FEPSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  initialize();
  numFramesWritten = 0;
}
FEPSubscriber::FEPSubscriber(const std::string &fileName, int reportFreq)
    : Subscriber(fileName, reportFreq) {
  initialize();
  numFramesWritten = 0;
}

FEPSubscriber::~FEPSubscriber() { fout.close(); }

void FEPSubscriber::initialize() { fout.open(fileName); }

void FEPSubscriber::update() {
  auto fm = charmmContext->getForceManager();
  auto bridgeEDSForceManager = std::dynamic_pointer_cast<FEPEIForceManager>(fm);

  if (fm->isComposite()) {
    auto children = fm->getChildren();

    auto lambdaPotentialEnergies =
        bridgeEDSForceManager->getLambdaPotentialEnergies();
    for (int i = 0; i < lambdaPotentialEnergies.size(); i++) {
      fout << lambdaPotentialEnergies[i] << "\t";
    }
    fout << std::endl;

    ++numFramesWritten;
  } else {
    std::cout << "WARNING -- You should not be using a FEPSubscriber with a "
                 "non-composite ForceManager.\n";
  }
}
