// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "Checkpoint.h"
#include "Subscriber.h"
#include <fstream>

class CheckpointSubscriber : public Subscriber {
public:
  CheckpointSubscriber(std::shared_ptr<Checkpoint> checkpoint);
  CheckpointSubscriber(const std::string &fileName);
  CheckpointSubscriber(const std::string &fileName, const int reportFreq);

  void setReportFrequency(const int reportFreq);

  int getReportFrequency(void) const;

private:
  std::shared_ptr<Checkpoint> m_Checkpoint;
  int m_ReportFreq;
};