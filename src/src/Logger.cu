// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Félix Aviat, Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CudaIntegrator.h"
#include "ForceManager.h"
#include "Logger.h"
#include "PBC.h"
#include "tinyxml2.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>

/* Extras :
 - the code version
 - commit hash ?
 */

//// Constructors
Logger::Logger(std::shared_ptr<CharmmContext> contextIn) {
  context = contextIn;
  setFilename(getDefaultFilename());
  initialize();
}

Logger::Logger(std::shared_ptr<CharmmContext> contextIn,
               std::string logFileNameIn) {
  context = contextIn;
  setFilename(logFileNameIn);
  initialize();
}

Logger::~Logger() { // fout.close();
}

//// Utilities

// TODO : use the jobID instead of a random number
std::string Logger::getDefaultFilename() {
  // Get current date and time in manipulable format
  auto date = std::chrono::system_clock::now();
  std::time_t timet = std::chrono::system_clock::to_time_t(date);

  // Random tools to hopefully generate unique name if several logs are created
  // at same second
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> randomstamp(0,
                                                                       100000);

  // Format
  std::stringstream ss;
  ss << std::put_time(std::localtime(&timet), "%Y%m%d%H%M%S") << "_"
     << randomstamp(rng) << ".log";

  return ss.str();
}

void Logger::setFilename(std::string logFileNameIn) {
  filename = logFileNameIn;
}

void Logger::initialize() {
  // TODO : open file, write header
  // We need the code version extractable from somewhere !
  // fout.open(filename);

  root = xmldoc.NewElement("ApoCharmm Log");
  root->SetAttribute("Version", "");
  root->SetAttribute("Platform", "");
  xmldoc.InsertFirstChild(root);
}

// TODO : This is the same as the subscriber function (copy-paste). Should be
// put somewhere else (in a utils file ?)
// TODO : we also need to check that the file does not exist already
void checkPath(std::string fileName) {
  std::string pathandfile = fileName;
  // If no "/" character in pathandfile, nothing to check
  if (pathandfile.find('/') == std::string::npos) {
    return;
  }
  struct stat sb;
  std::size_t botDirPos = pathandfile.find_last_of('/');
  std::string dirname = pathandfile.substr(0, botDirPos);
  if (dirname != "") {
    // Check that the directory actually exists
    // if (stat(dirname, &sb) != 0) {
    if (stat(dirname.c_str(), &sb) != 0) {
      std::cout << "Error (path: " << dirname << " )\n";
      throw std::invalid_argument("Output directory does not exist\n");
      exit(1);
    }
  }
}

//// Logging functions

void Logger::updateLog(std::shared_ptr<CudaIntegrator> integrator, int nsteps) {
  if (not isContextLogged) {
    logContext();
  }
  if (not isForceManagerLogged) {
    logForceManager();
  }
  logIntegrator(integrator);
  if (nsteps != 0) {
    logIntegration(nsteps);
  }
}

/* CharmmContext contained params/things that we might want to log:
- forceManager -> WHAT DOES THAT MEAN ? What am I writing ?
- numAtoms
- numDegreesOfFreedom
- pbc
*/
void Logger::logContext() {
  tinyxml2::XMLElement *xmlContext = xmldoc.NewElement("CharmmContext");
  xmlContext->SetAttribute("dof", context->getNumDegreesOfFreedom());
  root->InsertEndChild(xmlContext);
  isContextLogged = true;
}

void Logger::logForceManager() {
  std::shared_ptr<ForceManager> fm = context->getForceManager();
  if (fm->isComposite()) { return ; } // if composite, don't log (not implemented yet)
  tinyxml2::XMLElement *xmlForceManager = xmldoc.NewElement("ForceManager");
  xmlForceManager->SetAttribute("NumAtoms", fm->getNumAtoms());
  xmlForceManager->SetAttribute("Cutoff", fm->getCutoff());
  xmlForceManager->SetAttribute("CTONNB", fm->getCtonnb());
  xmlForceManager->SetAttribute("CTOFNB", fm->getCtofnb());
  xmlForceManager->SetAttribute("Kappa", fm->getKappa());

  tinyxml2::XMLElement *xmlBox = xmldoc.NewElement("Box");
  xmlBox->SetAttribute("X", fm->getBoxDimensions()[0]);
  xmlBox->SetAttribute("Y", fm->getBoxDimensions()[1]);
  xmlBox->SetAttribute("Z", fm->getBoxDimensions()[2]);

  // stringify PBC ?
  std::map<PBC, std::string> pbcMap;
  pbcMap[PBC::P1] = "P1";
  pbcMap[PBC::P21] = "P21";
  std::string tempstr = pbcMap[fm->getPeriodicBoundaryCondition()].c_str();
  xmlForceManager->SetAttribute("PBC", tempstr.c_str());

  tinyxml2::XMLElement *xmlPSF = xmldoc.NewElement("PSF");
  xmlPSF->SetText(fm->getPSF()->getOriginalPSFFileName().c_str());
  tinyxml2::XMLElement *xmlPRM = xmldoc.NewElement("PRM");
  std::vector<std::string> prmFileNames =
      fm->getPrm()->getOriginalPrmFileNames();
  std::string tempstr2 = std::accumulate(prmFileNames.begin(),
                                         prmFileNames.end(), std::string(" "));
  xmlPRM->SetText(tempstr2.c_str());

  xmlForceManager->InsertEndChild(xmlBox);
  xmlForceManager->InsertEndChild(xmlPSF);
  xmlForceManager->InsertEndChild(xmlPRM);
  root->InsertEndChild(xmlForceManager);

  isForceManagerLogged = true;
}

/* Integrators : it's a slightly bigger problems.
Each integrator has a different set of variables/params of interest. So we'd
have to call a function that distinguishes which child class of integrator is
in use. Instead, we'll need to put a function in EACH child integrator class,
returning a dict (data type tbd) that can then be logged by the Logger itself.
What if... we then create difft types of FM, Contexts, ... ? Seems unlikely,
given the design. These are true mediators. Integrator isn't. */

void Logger::logIntegrator(std::shared_ptr<CudaIntegrator> integrator) {
  tinyxml2::XMLElement *xmlIntegrator = xmldoc.NewElement("Integrator");
  auto descriptors = integrator->getIntegratorDescriptors();
  for (auto &descriptor : descriptors) {
    xmlIntegrator->SetAttribute(descriptor.first.c_str(),
                                descriptor.second.c_str());
  }

  root->InsertEndChild(xmlIntegrator);
}

/* Integration : Log number of steps called by propagate() function. Just to
 * know how long things are running. Might be useless.*/
void Logger::logIntegration(int nSteps) {
  tinyxml2::XMLElement *xmlPropagation = xmldoc.NewElement("Propagation");
  xmlPropagation->SetText(std::to_string(nSteps).c_str());
  root->InsertEndChild(xmlPropagation);
  xmldoc.SaveFile(filename.c_str());
}
