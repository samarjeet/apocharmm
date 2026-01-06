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

#include "Subscriber.h"
#include <fstream>

class DualTopologySubscriber : public Subscriber {
public:
  DualTopologySubscriber(const std::string &fileName);
  DualTopologySubscriber(const std::string &fileName, int reportFrequency);
  ~DualTopologySubscriber(void);

public:
  void update(void) override;

private:
  int m_NumFramesWritten;
};
