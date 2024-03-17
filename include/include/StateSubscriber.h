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
#include <map>
#include <vector>

/**
 * @brief Infos over time
 *
 *  Kinetic, Potential and Total energy reporter. Depends on a CharmmContext.
 *  Outputs to a file.
 * @todo Add HFCTE. Add a way to select which quantities to output
 */
class StateSubscriber : public Subscriber {
public:
  StateSubscriber(const std::string &fileNameIn, int reportFreq);
  StateSubscriber(const std::string &fileName);
  void update() override;
  ~StateSubscriber();

  void setReportPotentialEnergy(bool reportPotentialEnergyIn = true) { reportFlags["potentialenergy"] = reportPotentialEnergyIn; }
  void setReportKineticEnergy(bool reportKineticEnergyIn = true) { reportFlags["kineticenergy"] = reportKineticEnergyIn; }
  void setReportTotalEnergy(bool reportTotalEnergyIn = true) { reportFlags["totalenergy"] = reportTotalEnergyIn; }
  void setReportTemperature(bool reportTemperatureIn = true) { reportFlags["temperature"] = reportTemperatureIn; }
  void setReportPressureComponents(bool reportPressureComponentsIn = true) { reportFlags["pressurecomponents"] = reportPressureComponentsIn; }
  void setReportPressureScalar(bool reportPressureScalarIn = true) { reportFlags["pressurescalar"] = reportPressureScalarIn; }
  void setReportBoxSizeComponents(bool reportBoxSizeComponentsIn = true) { reportFlags["boxsizecomponents"] = reportBoxSizeComponentsIn; }
  void setReportVolume(bool reportVolumeIn = true) { reportFlags["volume"] = reportVolumeIn; }

  std::map<std::string, bool> getReportFlags() { return reportFlags; }

  /** @brief Prepares a vector of strings to be used as flags, then sets the
   * flags */
  void readReportFlags(std::vector<std::string> reportStrings);
  /** @brief Splits a string into vector of strings, trimmed, then sets the flags */
  void readReportFlags(std::string reportStringIn);

private:
  /** @brief Sets up default report quantities, writes header.  */
  void initialize();

  /** @brief Write header depending on the reporting flags */
  void writeHeader();

  /**
   * @brief Compute time propagated 
   *
   * Computes the total propagated time based on the number of frames written
   * (numFramesWritten), the report frequency and the integrator timestep.
   */
  float computeTime();

  /**
   * @brief Number of updates done
   *
   * Counter-like int variable, incremented everytime the update() function is
   * called.
   */
  int numFramesWritten;

  /** @brief Map of all the possible report flags. keys are quantities names,
   * values are pointers to the corresponding bools. */
  std::map<std::string, bool> reportFlags = {
    {"potentialenergy",    false},
    {"kineticenergy",      false},
    {"totalenergy",        false},
    {"temperature",        false},
    {"pressurecomponents", false},
    {"pressurescalar",     false},
    {"boxsizecomponents",  false},
    {"density",  false},
    {"volume",             false}
  };

  // formatting options
  int outwidth = 15 ;
  std::string spacing;

  bool headerWritten = false;

  /** @brief Given a list of strings, sets all report flags accordingly */
  void setReportFlags(std::vector<std::string> reportStrings);

};
