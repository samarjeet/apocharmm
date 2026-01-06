// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "EDSSubscriber.h"

#include "CharmmContext.h"
#include "EDSForceManager.h"
#include <fstream>
#include <iostream>

EDSSubscriber::EDSSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  m_NumFramesWritten = 0;
}

EDSSubscriber::EDSSubscriber(const std::string &fileName, int reportFrequency)
    : Subscriber(fileName, reportFrequency) {
  m_NumFramesWritten = 0;
}

EDSSubscriber::~EDSSubscriber(void) {
  if (m_FileStream.is_open())
    m_FileStream.close();
}

void EDSSubscriber::update(void) {
  // We need to access the potential energy for all members of the CompositeFM
  auto fm = m_CharmmContext->getForceManager();

  if (fm->isComposite()) {
    auto children = fm->getChildren();
    for (int i = 0; i < children.size(); i++) {
      auto childPotentialEnergy = children[i]->getPotentialEnergy();
      childPotentialEnergy.transferFromDevice();
      m_FileStream << childPotentialEnergy[0] << "\t";
    }
    auto pe = fm->getPotentialEnergy();
    pe.transferFromDevice();
    m_FileStream << pe[0] << "\t";
    m_FileStream << std::endl;

    m_NumFramesWritten++;
  } else {
    std::cout << "WARNING -- You should not be using a EDSSubscriber with a "
                 "non-composite ForceManager.\n";
  }

  return;
}
