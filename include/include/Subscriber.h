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

#include <fstream>
#include <memory>
#include <string>

class CharmmContext;
class CudaIntegrator;

class Subscriber {
  ////////////
  // PUBLIC //
  ////////////
public:
  /**
   * @param[in] fileName Name of the output file to which updates will be
   * printed.
   * @param[in] reportFreq (opt) Report frequency (number of timestep between
   * two updates). Optional.
   */
  Subscriber(const std::string &fileName, int reportFreq);
  Subscriber(const std::string &fileName);
  Subscriber() = default;

  /**
   * @brief Add an update to the output
   *
   * Query the system (CharmmContext) to update its target information, adds to
   * the output file
   */
  virtual void update() = 0;

  /**
   * @brief Sets name of output file
   *
   * Sets the name of the ouptut file to fileNameIn, after checking
   * that it is in an already existing path
   */
  void setFileName(std::string fileNameIn);

  /**
   * @brief Check existence of output file dir
   *
   * Throws an error if the path to the output file given (fileNameIn) is
   * incorrect/non-existing
   */
  void checkPath(std::string fileNameIn);

  /**
   * @brief Set frequency of reports
   *
   * Set number of steps between two reports of the Subscriber
   */
  void setReportFreq(int n) { reportFreq = n; }

  /**
   * @brief Returns report frequency
   * @ return number of steps between two reports
   */
  int getReportFreq();

  std::string getFileName() { return fileName; }

  /**
   * @brief Sets integratorTimeStep from CudaIntegrator.
   *
   * Should be called upon subscription.
   */
  void setTimeStepFromIntegrator(double ts);

  /** @brief Add a comment section to the output file. Assumes the input string
   * is formatted.
   * @param[in] commentLines String to add.
   *
   * Checks that the final character of the given input is a line break.
   */
  void addCommentSection(std::string commentLines);

  /** @brief Opens output file stream (checks path)
   */
  void openFile();

  /**
   * @brief Attaches Subscriber to a CharmmContext
   */
  void setCharmmContext(std::shared_ptr<CharmmContext> ctx);

  /** @brief Attaches integrator to Subscriber. Should be done by the Integrator
   * itself upon calling ::subscribe function. */
  void setIntegrator(std::shared_ptr<CudaIntegrator> integratorIn);

  ///////////////
  // PROTECTED //
  ///////////////
protected:
  std::string fileName;
  std::shared_ptr<CharmmContext> charmmContext;
  // std::ofstream fout;
  std::fstream fout;

  /**
   * @brief Number of timestep between reports
   *
   * Subscriber's output will be updated every reportFreq steps [default:1000].
   */
  int reportFreq;

  /**
   * @brief Tracks if CharmmContext has been set
   */
  bool hasCharmmContext = false;

  /**
   * @brief Timestep extracted from the CudaIntegrator
   */
  float integratorTimeStep;

  /** @brief Integrator linked to this subscriber*/
  std::shared_ptr<CudaIntegrator> integrator;

  /** @brief Tracks if integrator has been set */
  bool hasIntegrator = false;
};
