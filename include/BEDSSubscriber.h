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

class BEDSSubscriber : public Subscriber {
public:
  BEDSSubscriber(const std::string &fileName);
  BEDSSubscriber(const std::string &fileName, int reportFreq);
  void update() override;
  ~BEDSSubscriber();

private:
  void initialize();
  std::ofstream fout;

  int numFramesWritten;
};
