// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Félix Aviat, Samarjeet Prasad
//
// ENDLICENSE

/* Idea : have a logger class to create log files

The logger is attached to a CharmmContext, and should report:
 - the CharmmContext params/variables (outside of the large arrays),
 - the ForceManager attached to it,
 - upon integration, the Integrator params, and the integration description
(nsteps...)
 - the name of input files used ?

The logger would be called by the Integrator upon propagating many steps (not
through PropagateOneStep).

Should be created upon initialization of a CharmmContext ?


needs an output file name (obv), with good default if nothing provided.

Should report variables / parameters from :
- the integrator : this means that every child class needs its own logger
specific function ?
- the CharmmContext
- the ForceManager: long term, if we want to handle Composite, we'll need to
think about that too.

Every main class could have a "log" function, to enable the logger output to be
adapted to every class (ex: params for Langevin Piston are not the same as
Langevin Thermostat)

TODO:
- add version number to the log file,
- possibly commit hash ?
- xml (or yaml ? or json?) output format  */

#pragma once

#include "CharmmContext.h"
#include "CudaIntegrator.h"
#include "tinyxml2.h"
#include <fstream>
#include <memory>
#include <string>

class CudaIntegrator;
/** @brief Outputs a log file describing the Context, the ForceManager(s), the
 * Integrator(s) and their integrations undertaken.
 *
 * File is written down/updated upon calling propagate()  */
class Logger : public std::enable_shared_from_this<Logger> {
public:
  /** @brief Constructor class. Uses optionally filename as argument. Attached
   * to a context, necessarily, and is thus created with one. */
  Logger(std::shared_ptr<CharmmContext> context);
  Logger(std::shared_ptr<CharmmContext> context, std::string filename);

  /** @brief Destructor class, closes fout. */
  ~Logger();

  void setFilename(std::string filename);

  /** @brief Update log file with information that might have changed. Logs
   * everything upon first call, then only the integrators. Called by
   * integrator, should feed a pointer.*/
  void updateLog(std::shared_ptr<CudaIntegrator> integrator, int nsteps = 0);

protected:
  /** @brief Opens log file for writing. Adds most general information (code
   * version ?)
   * @todo add code version, commit hash, etc.
   */
  void initialize();

  /** @brief Outputs the information relative to the CharmmContext to the output
   * file */
  void logContext();

  /** @brief Outputs the information relative to the ForceManager to the output
   * file */
  void logForceManager();

  /** @brief Outputs the information relative to the Integrator to the output
   * file */
  void logIntegrator(std::shared_ptr<CudaIntegrator> integrator);

  /** @brief Logs number of steps to be taken by the next integration call. */
  void logIntegration(int nSteps);

  /** @brief returns a good default file name for the output log file
   * @todo Use the jobID instead of the random number at the end
   */
  std::string getDefaultFilename();

  /** @brief Output file name.*/
  std::string filename;
  /** @brief CharmmContext to be reported. */
  std::shared_ptr<CharmmContext> context;

  std::ofstream fout;

  bool isInitialized = false;
  bool isContextLogged = false;
  bool isForceManagerLogged = false;

  tinyxml2::XMLDocument xmldoc;
  tinyxml2::XMLElement *root;
};
