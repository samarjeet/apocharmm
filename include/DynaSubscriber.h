// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "Subscriber.h"

/** @brief Reports detailed information regarding dynamics properties
 * of the current simulation.
 */
class DynaSubscriber : public Subscriber {
public:
  DynaSubscriber(const std::string &fileName);
  DynaSubscriber(const std::string &fileName, const int reportFrequency);
  ~DynaSubscriber(void);

public:
  void update(void) override;

private:
  void writeHeader(void);

private:
  bool m_HasWrittenHeader;
};
