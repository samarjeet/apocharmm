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

#include <fstream>
#include <memory>
#include <string>

class CharmmContext;
class CudaIntegrator;

class Subscriber {
public:
  Subscriber(void);
  Subscriber(const std::string &fileName);

  /**
   * @param[in] fileName Name of the output file to which updates will be
   * printed.
   * @param[in] reportFrequency (opt) Report frequency (number of timestep
   * between two updates). Optional.
   */
  Subscriber(const std::string &fileName, const int reportFrequency);

public:
  /**
   * @brief Set frequency of reports
   *
   * Set number of steps between two reports of the Subscriber
   */
  void setReportFrequency(const int reportFrequency);

  /**
   * @brief Sets name of output file
   *
   * Sets the name of the ouptut file to fileNameIn, after checking
   * that it is in an already existing path
   */
  void setFileName(const std::string &fileName);

  // /**
  //  * @brief Sets integratorTimeStep from CudaIntegrator.
  //  *
  //  * Should be called upon subscription.
  //  */
  // void setTimeStepFromIntegrator(const double ts);

  /**
   * @brief Attaches Subscriber to a CharmmContext
   */
  void setCharmmContext(std::shared_ptr<CharmmContext> ctx);

  /** @brief Attaches integrator to Subscriber. Should be done by the Integrator
   * itself upon calling ::subscribe function. */
  void setIntegrator(std::shared_ptr<CudaIntegrator> integrator);

public:
  /**
   * @brief Returns report frequency
   * @ return number of steps between two reports
   */
  int getReportFrequency(void) const;

  const std::string &getFileName(void) const;

  std::string &getFileName(void);

public:
  /**
   * @brief Add an update to the output
   *
   * Query the system (CharmmContext) to update its target information, adds to
   * the output file
   */
  virtual void update(void) = 0;

  /**
   * @brief Check existence of output file dir
   *
   * Throws an error if the path to the output file given (fileNameIn) is
   * incorrect/non-existing
   */
  void checkPath(const std::string &fileName);

  /** @brief Opens output file stream (checks path)
   */
  void openFile(void);

  /** @brief Add a comment section to the output file. Assumes the input string
   * is formatted.
   * @param[in] commentLines String to add.
   *
   * Checks that the final character of the given input is a line break.
   */
  void addCommentSection(const std::string &commentLines);

protected:
  /**
   * @brief Number of timestep between reports
   *
   * Subscriber's output will be updated every reportFreq steps [default:1000].
   */
  int m_ReportFrequency;

  std::string m_FileName;

  std::fstream m_FileStream;

  std::shared_ptr<CharmmContext> m_CharmmContext;

  /** @brief Integrator linked to this subscriber*/
  std::shared_ptr<CudaIntegrator> m_Integrator;
};
