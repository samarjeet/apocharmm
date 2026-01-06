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

#include "ForceManager.h"
#include <fstream>

/** @brief Linked to several ForceManager objects. When reporting, asks each FM
 * to compute energy and reports it along with the actual CharmmContext FM. */

/** @brief Reports the enegy of every single ForceManager child */
class EDSSubscriber : public Subscriber {
public:
  EDSSubscriber(const std::string &fileName);
  EDSSubscriber(const std::string &fileName, int reportFrequency);
  ~EDSSubscriber(void);

public:
  void update(void) override;

private:
  int m_NumFramesWritten;
};
