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
#include <fstream>

/**
 * @brief CHARMM DCD output Subscriber
 *
 * Should be initialized with a report frequency and a context.
 */
class DcdSubscriber : public Subscriber {
public:
  DcdSubscriber(const std::string &fileName);
  DcdSubscriber(const std::string &fileName, int reportFrequency);
  /** @deprecated Constructor. CharmmContext dependency should be initialized
   * upon subscription */
  DcdSubscriber(const std::string &fileName, int reportFrequency,
                std::shared_ptr<CharmmContext> ctx);
  ~DcdSubscriber(void);

public:
  void update(void) override;

private:
  void writeHeader(void);
  void writeXtalData(void);
  void writeCoordData(void);

  // void initialize(void);

  /**
   * @brief Number of frames written in total (NFILE)
   */
  int m_NumFramesWritten;

  bool m_IsHeaderWritten;
};
