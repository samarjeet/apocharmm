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

class FEPSubscriber : public Subscriber {
public:
  FEPSubscriber(const std::string &fileName);
  FEPSubscriber(const std::string &fileName, int reportFreq);
  void update() override;
  ~FEPSubscriber();

private:
  void initialize();
  std::ofstream fout;

  int numFramesWritten;
};
