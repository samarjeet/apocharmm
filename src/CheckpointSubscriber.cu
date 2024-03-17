// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CheckpointSubscriber.h"

CheckpointSubscriber::CheckpointSubscriber(
    std::shared_ptr<Checkpoint> checkpoint) {
  m_Checkpoint = checkpoint;
}

CheckpointSubscriber::CheckpointSubscriber(const std::string &fileName)
    : m_Checkpoint(std::make_shared<Checkpoint>(fileName)), m_ReportFreq(1000) {
}

CheckpointSubscriber::CheckpointSubscriber(const std::string &fileName,
                                           const int reportFreq)
    : CheckpointSubscriber(fileName) {
  m_ReportFreq = reportFreq;
}

void CheckpointSubscriber::setReportFrequency(const int reportFreq) {
  m_ReportFreq = reportFreq;
  return;
}

int CheckpointSubscriber::getReportFrequency(void) const {
  return m_ReportFreq;
}