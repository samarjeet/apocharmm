// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "BEDSForceManager.h"
#include "BEDSSubscriber.h"
#include "CharmmContext.h"
#include <fstream>
#include <iostream>

BEDSSubscriber::BEDSSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  m_NumFramesWritten = 0;
}

BEDSSubscriber::BEDSSubscriber(const std::string &fileName, int reportFrequency)
    : Subscriber(fileName, reportFrequency) {
  m_NumFramesWritten = 0;
}

BEDSSubscriber::~BEDSSubscriber(void) {
  if (m_FileStream.is_open())m_FileStream.close(); return;}

void BEDSSubscriber::update(void) {
  auto fm = m_CharmmContext->getForceManager();
  auto bridgeEDSForceManager = std::dynamic_pointer_cast<BEDSForceManager>(fm);

  if (fm->isComposite()) {
    auto children = fm->getChildren();

    auto lambdaPotentialEnergies =
        bridgeEDSForceManager->getLambdaPotentialEnergies();
    for (int i = 0; i < lambdaPotentialEnergies.size(); i++)
      m_FileStream << lambdaPotentialEnergies[i] << "\t";
    m_FileStream << std::endl;

    m_NumFramesWritten++;
  } else {
    std::cout << "WARNING -- You should not be using a BEDSSubscriber with a "
                 "non-composite ForceManager.\n";
  }

  return;
}
