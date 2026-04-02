// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include "Subscriber.h"
#include <CudaContainer.h>

/** @brief Reports positions and velocities necessary to resume simulation. The
 * coordinates saved
 *
 *
 *
 * For now, we'll assume there's only ONE file written as output, and it gets
 * rewritten at each update.
 */
class RestartSubscriber : public Subscriber {
public:
  RestartSubscriber(void);

  /** @brief Basic constructor, writes output to fileName and uses default
   * report frequency of 1/1000 steps.
   */
  RestartSubscriber(const std::string &fileName);

  /** @brief XYZSubscriber constructor, takes output file name and report
   * frequency as parameters
   * @param[in] fileName Name (and possibly path) of the output file
   * @param[in] reportFrequency Number of steps between two reports
   */
  RestartSubscriber(const std::string &fileName, const int reportFrequency);

  ~RestartSubscriber(void);

  /** @brief Rewrites the restart file to the latest configuration.
   *
   * File contains three sections : positions, velocities, and box dimensions.
   */
  void update(void) override;
};
