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
  /** @brief Basic constructor, take output to fileName and uses default
   * report frequency of 1/1000 steps.
   * @param[in] _fileName Name (and possibly path) of the output file
   */
  DynaSubscriber(const std::string &_fileName);
  /** @brief Basic constructor, takes output file name and report
   * frequency as parameters
   * report frequency of 1/1000 steps.
   * @param[in] _fileName Name (and possibly path) of the output file
   * @param[in] _reportFreq Number of steps between reports
   */
  DynaSubscriber(const std::string &_fileName,
                 const int _reportFreq);
  /** @brief Destructor, flushes output and closes file.
   */
  ~DynaSubscriber(void);

public:
  void update(void) override;

private:
  void writeHeader(void);

private:
  bool m_HasWrittenHeader;
};
