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
#include <CudaContainer.h>
#include <fstream>

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
  /** @brief Basic constructor, writes output to fileName and uses default
   * report frequency of 1/1000 steps.
   */
  RestartSubscriber(const std::string &fileName);
  /** @brief XYZSubscriber constructor, takes output file name and report
   * frequency as parameters
   * @param[in] fileName Name (and possibly path) of the output file
   * @param[in] reportFreq Number of steps between two reports
   */
  RestartSubscriber(const std::string &fileName, int reportFreq);
  /** @brief Rewrites the restart file to the latest configuration.
   *
   * File contains three sections : positions, velocities, and box dimensions.
   */

  void update() override;
  ~RestartSubscriber();

  std::vector<double> readBoxDimensions() const;
  /** @brief Parses content of the subscriber's output file, extracts on-step
   * piston position (needed to restart a Langevin Piston integration)
   */
  std::vector<double> readOnStepPistonPosition() const;
  std::vector<double> readHalfStepPistonPosition() const;
  std::vector<double> readOnStepPistonVelocity() const;
  std::vector<double> readHalfStepPistonVelocity() const;

  std::vector<std::vector<double>> readPositions() const;
  std::vector<std::vector<double>> readVelocities() const;
  std::vector<std::vector<double>> readCoordsDeltaPrevious() const;

  /** @brief Parser functions for Nose-Hoover variables */
  double readNoseHooverPistonPosition() const;
  double readNoseHooverPistonVelocity() const;
  double readNoseHooverPistonForce() const;

  /** @brief generic function. Not actually used. */
  void getRestartContent(std::string fileName, std::string sectionName);

  /** @brief Reads a restart file and sets up the context and integrator to be
   * able to resume simulation
   * @note This was implemented for the Langevin Piston integrator initially,
   * and might require some work to be adapted to the other (simplification) */
  void readRestart();

  /** @brief Redefined here to read in read/write mode (as we can't erase the
   * file contents)
   */
  void openFile();

private:
  int numFramesWritten;
  int numAtoms;

  /** @brief Tracks if subscriber has been initialized (=numAtoms has been set)
   */
  bool isInitialized;

  /** @brief Checks that no coordinate or velocity (or box dim! ) is NaN. Throws
   * an error otherwise, and does not proceed with the restart file printing.
   * @todo Unittest this
   */
  void checkForNanValues(CudaContainer<double4> coords,
                         CudaContainer<double4> velocities,
                         std::vector<double> boxDims);
};