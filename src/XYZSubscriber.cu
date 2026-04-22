// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "XYZSubscriber.h"

#include "CharmmContext.h"
#include <iostream>

XYZSubscriber::XYZSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  m_NumFramesWritten = 0;
}

XYZSubscriber::XYZSubscriber(const std::string &fileName,
                             const int reportFrequency)
    : Subscriber(fileName, reportFrequency) {
  m_NumFramesWritten = 0;
}

XYZSubscriber::~XYZSubscriber(void) {
  if (m_FileStream.is_open())
    m_FileStream.close();
}

void XYZSubscriber::update(void) {
  if (m_CharmmContext == nullptr)
    throw std::runtime_error("ERROR: XYZSubscriber has no CHARMM context.");

  // // vector in the shared ptr
  // std::vector<float4> xyzq = *(m_CharmmContext->getXYZQ()->getHostXYZQ());

  // for (int i = 0; i < m_CharmmContext->getNumAtoms(); i++) {
  //   m_FileStream << i << "\t" << xyzq[i].x << "\t" << xyzq[i].y << "\t"
  //                << xyzq[i].z << std::endl;
  // }
  m_CharmmContext->getXYZQ().transferToHost();
  for (int i = 0; i < m_CharmmContext->getNumAtoms(); i++) {
    m_FileStream << i << "\t" << m_CharmmContext->getXYZQ()[i].x << "\t"
                 << m_CharmmContext->getXYZQ()[i].y << "\t"
                 << m_CharmmContext->getXYZQ()[i].z << std::endl;
  }

  m_NumFramesWritten++;

  return;
}
