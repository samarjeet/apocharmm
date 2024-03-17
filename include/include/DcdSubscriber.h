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

/**
 * @brief CHARMM DCD output Subscriber
 * 
 * Should be initialized with a report frequency and a context.
 */
class DcdSubscriber : public Subscriber {
public:
  DcdSubscriber(const std::string &fileName);
  DcdSubscriber(const std::string &fileName, int reportFreq);
  /** @deprecated Constructor. CharmmContext dependency should be initialized upon subscription */
  DcdSubscriber(const std::string &fileName, int reportFreq,
                std::shared_ptr<CharmmContext> ctx);
  ~DcdSubscriber();
  void update() override;

private:
  void initialize();
  /**
   * @brief Number of frames written in total
   */
  int numFramesWritten;
  int numAtoms;

  /** @brief Monitor if the subscriber has been initialized, e.g. if header has
   * been written 
   */
  bool isInitialized;
};
