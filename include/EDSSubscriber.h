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

#include "ForceManager.h"
#include "Subscriber.h"
#include <fstream>

/** @brief Linked to several ForceManager objects. When reporting, asks each FM
 * to compute energy and reports it along with the actual CharmmContext FM. */

/** @brief Reports the enegy of every single ForceManager child */
class EDSSubscriber : public Subscriber {
public:
  EDSSubscriber(const std::string &fileName);
  EDSSubscriber(const std::string &fileName, int reportFreq);
  void update() override;
  ~EDSSubscriber();

private:
  void initialize();
  int numFramesWritten;
  std::vector<std::shared_ptr<ForceManager>> fmlist;
};
