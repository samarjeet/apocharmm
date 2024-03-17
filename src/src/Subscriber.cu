// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "CudaIntegrator.h"
#include "Subscriber.h"
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <typeinfo>

// Two constructors : one with a specified reportFreq, one using default
// (filename only)

Subscriber::Subscriber(const std::string &fileName) : fileName(fileName) {
  setReportFreq(1000);
  openFile();
}

Subscriber::Subscriber(const std::string &fileNameIn, int reportFreq)
    : fileName(fileNameIn) {
  setReportFreq(reportFreq);
  openFile();
}

void Subscriber::setCharmmContext(std::shared_ptr<CharmmContext> ctx) {
  if (hasCharmmContext) {
    throw std::invalid_argument("Subscriber already has a CharmmContext.\n");
  }
  charmmContext = ctx;
  hasCharmmContext = true;
}

void Subscriber::checkPath(std::string fileName) {
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
    if (stat(dirname.c_str(), &sb) != 0) {
      std::cout << "Error (path: " << dirname << " )\n";
      throw std::invalid_argument("Output directory does not exist\n");
      exit(1);
    }
  }
}

int Subscriber::getReportFreq() { return reportFreq; }

void Subscriber::setFileName(std::string fileNameIn) {
  checkPath(fileNameIn);
  fileName = fileNameIn;
}

void Subscriber::setTimeStepFromIntegrator(double ts) {
  integratorTimeStep = ts;
}

void Subscriber::addCommentSection(std::string commentLines) {
  if (commentLines[commentLines.length() - 1] != '\n') {
    // Make sure that last char is a line break
    commentLines += "\n";
  }
  fout << commentLines;
}

void Subscriber::openFile() {
  checkPath(fileName);
  // fout.open(fileName, std::ios::in | std::ios::out);
  fout.open(fileName, std::ios::out);
}

void Subscriber::setIntegrator(std::shared_ptr<CudaIntegrator> integratorIn) {
  if (hasIntegrator) {
    throw std::invalid_argument("Subscriber already has an Integrator.\n");
  }
  integrator = integratorIn;
  hasIntegrator = true;
}
