// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "DualTopologySubscriber.h"
#include <fstream>
#include <iostream>

DualTopologySubscriber::DualTopologySubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  m_NumFramesWritten = 0;
}

DualTopologySubscriber::DualTopologySubscriber(const std::string &fileName,
                                               int reportFrequency)
    : Subscriber(fileName, reportFrequency) {
  m_NumFramesWritten = 0;
}

DualTopologySubscriber::~DualTopologySubscriber(void) {
  if (m_FileStream.is_open())
    m_FileStream.close();
}

void DualTopologySubscriber::update(void) {
  m_CharmmContext->calculateForces(false, true, true);

  // the previous command has calculated the energies as well
  // here we are just calling them. No force or energy is being calculated here
  m_FileStream << m_CharmmContext->getPotentialEnergies() << '\t';

  m_NumFramesWritten++;

  return;
}
