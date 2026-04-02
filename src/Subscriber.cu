// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CudaIntegrator.h"
#include "Subscriber.h"
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <typeinfo>

Subscriber::Subscriber(void)
    : m_ReportFrequency(1000), m_FileName(""), m_FileStream(),
      m_CharmmContext(nullptr), m_Integrator(nullptr) {}

Subscriber::Subscriber(const std::string &fileName) : Subscriber() {
  this->setFileName(fileName);
  this->openFile();
}

Subscriber::Subscriber(const std::string &fileName, const int reportFrequency)
    : Subscriber(fileName) {
  m_ReportFrequency = reportFrequency;
}

void Subscriber::setReportFrequency(const int reportFrequency) {
  m_ReportFrequency = reportFrequency;
  return;
}

void Subscriber::setFileName(const std::string &fileName) {
  this->checkPath(fileName);
  m_FileName = fileName;
  return;
}

void Subscriber::setCharmmContext(std::shared_ptr<CharmmContext> ctx) {
  if (m_CharmmContext != nullptr)
    throw std::invalid_argument("Subscriber already has a CharmmContext.\n");
  m_CharmmContext = ctx;
  return;
}

void Subscriber::setIntegrator(std::shared_ptr<CudaIntegrator> integrator) {
  if (m_Integrator != nullptr)
    throw std::invalid_argument("Subscriber already has an Integrator.\n");
  m_Integrator = integrator;
  return;
}

int Subscriber::getReportFrequency(void) const { return m_ReportFrequency; }

const std::string &Subscriber::getFileName(void) const { return m_FileName; }

std::string &Subscriber::getFileName(void) { return m_FileName; }

void Subscriber::checkPath(const std::string &fileName) {
  std::string totalFilePath = fileName;

  // If no "/" character in pathandfile, nothing to check
  if (totalFilePath.find('/') == std::string::npos)
    return;

  struct stat sb;
  std::size_t botDirPos = totalFilePath.find_last_of('/');
  std::string dirName = totalFilePath.substr(0, botDirPos);
  if (dirName != "") {
    // Check that the directory actually exists
    if (stat(dirName.c_str(), &sb) != 0) {
      throw std::invalid_argument("FATAL ERROR: directory \"" + dirName +
                                  "\" does not exist\n");
    }
  }

  return;
}

void Subscriber::openFile(void) {
  this->checkPath(m_FileName);
  m_FileStream.open(m_FileName, std::ios::out);
  return;
}

void Subscriber::addCommentSection(const std::string &commentLines) {
  // Make sure that last char is a line break
  std::string sdum = commentLines;
  if (commentLines[commentLines.length() - 1] != '\n')
    sdum += "\n";
  m_FileStream << sdum;
  return;
}
