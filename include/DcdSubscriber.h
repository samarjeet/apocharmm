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

#include "Subscriber.h"

/**
 * @brief CHARMM DCD output Subscriber
 *
 * Should be initialized with a report frequency and a context.
 */
class DcdSubscriber : public Subscriber {
public:
  DcdSubscriber(const std::string &fileName);
  DcdSubscriber(const std::string &fileName, const int reportFrequency);
  ~DcdSubscriber(void);

public:
  void update(void) override;

private:
  void writeHeader(void);
  void writeXtalData(void);
  void writeCoordData(void);

  /**
   * @brief Number of frames written in total (NFILE)
   */
  int m_NumFramesWritten;

  bool m_IsHeaderWritten;
};
