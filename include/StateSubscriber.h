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
  StateSubscriber(const std::string &fileName);
  StateSubscriber(const std::string &fileNameIn, const int reportFrequency);
  ~StateSubscriber();

public:
  void update(void) override;

  void setReportPotentialEnergy(const bool reportPotentialEnergy = true);
  void setReportKineticEnergy(const bool reportKineticEnergy = true);
  void setReportTotalEnergy(const bool reportTotalEnergy = true);
  void setReportTemperature(const bool reportTemperature = true);
  void setReportPressureComponents(const bool reportPressureComponents = true);
  void setReportPressureScalar(const bool reportPressureScalar = true);
  void setReportBoxSizeComponents(const bool reportBoxSizeComponents = true);
  void setReportVolume(const bool reportVolume = true);

  const std::map<std::string, bool> &getReportFlags(void) const;
  std::map<std::string, bool> &getReportFlags(void);

  /** @brief Prepares a vector of strings to be used as flags, then sets the
   * flags */
  void readReportFlags(const std::vector<std::string> &reportStrings);
  /** @brief Splits a string into vector of strings, trimmed, then sets the
   * flags */
  void readReportFlags(const std::string &reportString);

private:
  /** @brief Sets up default report quantities, writes header.  */
  void initialize(void);

  /**
   * @brief Compute time propagated
   *
   * Computes the total propagated time based on the number of frames written
   * (numFramesWritten), the report frequency and the integrator timestep.
   */
  float computeTime(void);

  /** @brief Given a list of strings, sets all report flags accordingly */
  void setReportFlags(const std::vector<std::string> &reportFlags);

  /** @brief Write header depending on the reporting flags */
  void writeHeader(void);

private:
  /**
   * @brief Number of updates done
   *
   * Counter-like int variable, incremented everytime the update() function is
   * called.
   */
  int m_NumFramesWritten;

  /** @brief Map of all the possible report flags. keys are quantities names,
   * values are pointers to the corresponding bools. */
  std::map<std::string, bool> m_ReportFlags;

  // formatting options
  int m_OutWidth;
  std::string m_Spacing;

  bool m_HeaderWritten;
};
