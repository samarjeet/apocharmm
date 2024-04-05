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

/** @brief Reports atomic positions 
*/
class XYZSubscriber : public Subscriber {
public:
  /** @brief Basic constructor, writes output to fileName and uses default report
   * frequency of 1/1000 steps.
   */
  XYZSubscriber(const std::string &fileName);
  /** @brief XYZSubscriber constructor, takes output file name and report
   * frequency as parameters
   * @param[in] fileName Name (and possibly path) of the output file
   * @param[in] reportFreq Number of steps between two reports
   */
  XYZSubscriber(const std::string &fileName, int reportFreq);
  void update() override;
  ~XYZSubscriber();

private:
  void initialize();

  int numFramesWritten;
  //int numAtoms;

  /** @brief Tracks if subscriber has been initialized (=numAtoms has been set) */
  bool isInitialized;
};
