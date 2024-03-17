// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CudaLangevinPistonIntegrator.h"
#include "StateSubscriber.h"
#include "cpp_utils.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

StateSubscriber::StateSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  initialize();
  numFramesWritten = 0;
}
StateSubscriber::StateSubscriber(const std::string &fileName, int reportFreq)
    : Subscriber(fileName, reportFreq) {
  initialize();
  numFramesWritten = 0;
}

StateSubscriber::~StateSubscriber() { fout.close(); }

void StateSubscriber::initialize() {
  // Default values to be reported
  reportFlags["potentialenergy"] = true;
  reportFlags["kineticenergy"] = true;
  reportFlags["totalenergy"] = true;
  reportFlags["temperature"] = true;
  reportFlags["pressurescalar"] = false;
  for (int i = 0; i < outwidth; ++i) {
    spacing += " ";
  }
  // writeHeader();
}

void StateSubscriber::update() {
  // TODO : add HFCTE
  if (!headerWritten) {
    writeHeader();
    headerWritten = true;
  }

  // Update energies
  this->charmmContext->calculatePotentialEnergy(true, false);
  auto peContainer = this->charmmContext->getPotentialEnergy();
  peContainer.transferFromDevice();
  double pe = peContainer.getHostArray()[0];

  auto ke = this->charmmContext->getKineticEnergy();

  auto boxDim = this->charmmContext->getBoxDimensions();

  // Compute the density
  /* Take the mass
  Convert into g
  Get the volume
  Convert it into cm^3
  Dive the converted values by each other m/v
  */
  CudaContainer<double4> velmassCC = charmmContext->getVelocityMass();
  velmassCC.transferFromDevice();
  std::vector<double4> velmass = velmassCC.getHostArray();
  double densityMass = 0.0;
  for (int i = 0; i < velmass.size(); i++) {
    densityMass = densityMass + 1. / velmass[i].w;
  }
  double convertedDensityMass = densityMass * 1.660540199e-24;

  double densityVolume = charmmContext->getVolume();
  double convertedDensityVolume = densityVolume * 1e-24;

  double density = convertedDensityMass / convertedDensityVolume;

  //  Compute time spent
  float time = computeTime();
  float temperature = this->charmmContext->computeTemperature();

  fout << std::left << std::setw(outwidth) << std::to_string(time);
  if (reportFlags["potentialenergy"]) {
    fout << std::setw(outwidth) << std::to_string(pe);
  }
  if (reportFlags["kineticenergy"]) {
    fout << std::setw(outwidth) << std::to_string(ke);
  }
  if (reportFlags["totalenergy"]) {
    fout << std::setw(outwidth) << std::to_string(ke + pe);
  }
  if (reportFlags["temperature"]) {
    fout << std::setw(outwidth) << std::to_string(temperature);
  }
  if (reportFlags["pressurescalar"]) {
    try {
      auto langevinPistonIntegrator =
          std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(
              this->integrator);
      auto pressure = langevinPistonIntegrator->getPressureScalar();
      fout << std::setw(outwidth) << std::to_string(pressure);
    } catch (const std::exception &e) {
      throw std::invalid_argument(
          "Current integrator does not support pressure calculation");
    }
  }
  if (reportFlags["pressurecomponents"]) {
    try {
      auto langevinPistonIntegrator =
          std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(
              this->integrator);
      auto pressureTensor = langevinPistonIntegrator->getPressureTensor();
      fout << std::setw(outwidth) << std::to_string(pressureTensor[0])
           << std::setw(outwidth) << std::to_string(pressureTensor[4])
           << std::setw(outwidth) << std::to_string(pressureTensor[8]);
    } catch (const std::exception &e) {
      throw std::invalid_argument(
          "Current integrator does not support pressure calculation");
    }
  }
  if (reportFlags["boxdimensions"]) {
    fout << std::setw(outwidth) << std::to_string(boxDim[0])
         << std::setw(outwidth) << std::to_string(boxDim[1])
         << std::setw(outwidth) << std::to_string(boxDim[2]);
  }
  if (reportFlags["density"]) {
    fout << std::setw(outwidth) << std::to_string(density);
  }

  fout << std::endl;
  ++numFramesWritten;
}

float StateSubscriber::computeTime() {
  float t = (numFramesWritten + 1) * reportFreq * integratorTimeStep;
  return t;
}

// Splits a string into words following comma as delimiter.
// No other modification
std::vector<std::string> splitStringToWords(std::string reportStringIn) {
  std::string flag;
  std::vector<std::string> foundWords;
  std::stringstream ss(reportStringIn);
  while (getline(ss, flag, ',')) {
    foundWords.push_back(flag);
  }
  return foundWords;
}

// Trims (eliminates space) + sets everything to lower case
std::vector<std::string> trimWords(std::vector<std::string> reportStrings) {
  std::vector<std::string> trimmedWords;
  for (auto &c : reportStrings) {
    c = trim(c);
    trimmedWords.push_back(toLower(c));
  }
  return trimmedWords;
}

void StateSubscriber::readReportFlags(std::vector<std::string> reportStrings) {
  std::vector<std::string> foundFlags = trimWords(reportStrings);
  setReportFlags(foundFlags);
}

void StateSubscriber::readReportFlags(std::string reportStringIn) {
  std::vector<std::string> splittedWords = splitStringToWords(reportStringIn);
  std::vector<std::string> foundFlags = trimWords(splittedWords);
  setReportFlags(foundFlags);
}

void StateSubscriber::setReportFlags(std::vector<std::string> inputFlags) {
  // for (auto &c: inputFlags) c = tolower(c);
  for (int i = 0; i < inputFlags.size(); i++) {
    for (auto &c : inputFlags[i])
      c = tolower(c);
  }
  for (auto &c : reportFlags)
    c.second = false; // set all flags to false
  // if "all" is found, set all flags to true
  if (std::any_of(inputFlags.begin(), inputFlags.end(),
                  [](std::string &s) { return s == "all"; })) {
    for (auto &f : reportFlags) {
      f.second = true;
    }
  } else {
    // Otherwise, set only the flags found to true
    for (auto &f : inputFlags) {
      if (reportFlags.find(f) != reportFlags.end()) {
        reportFlags[f] = true;
      } else {
        std::cout << "Warning: " << f << " is not a valid flag" << std::endl;
        //                if (f.find("pressure") != std::string::npos) {
        //                    std::cout << "Did you mean pressurescalar or
        //                    pressurecomponents?" << std::endl;
      }
    }
  }
}

void StateSubscriber::writeHeader() {
  fout << std::left << std::setw(outwidth) << "# Time";
  if (reportFlags["potentialenergy"]) {
    fout << std::setw(outwidth) << "Pot. ene";
  }
  if (reportFlags["kineticenergy"]) {
    fout << std::setw(outwidth) << "Kin. ene";
  }
  if (reportFlags["totalenergy"]) {
    fout << std::setw(outwidth) << "Total ene";
  }
  if (reportFlags["temperature"]) {
    fout << std::setw(outwidth) << "Temperature";
  }
  if (reportFlags["pressurescalar"]) {
    fout << std::setw(outwidth) << "Pressure";
  }
  if (reportFlags["pressurecomponents"]) {
    fout << std::setw(3 * outwidth) << spacing + "Pressure comp.";
  }
  if (reportFlags["boxsizecomponents"]) {
    fout << std::setw(3 * outwidth) << spacing + "Box (AA)";
  }
  if (reportFlags["density"]) {
    fout << std::setw(outwidth) << "Density";
  }

  fout << std::endl;
  fout << std::left << std::setw(outwidth) << "# (ps)";
  if (reportFlags["potentialenergy"]) {
    fout << std::setw(outwidth) << "(kcal/mol)";
  }
  if (reportFlags["kineticenergy"]) {
    fout << std::setw(outwidth) << "(kcal/mol)";
  }
  if (reportFlags["totalenergy"]) {
    fout << std::setw(outwidth) << "(kcal/mol)";
  }
  if (reportFlags["temperature"]) {
    fout << std::setw(outwidth) << "(K)";
  }
  if (reportFlags["pressurescalar"]) {
    fout << std::setw(outwidth) << "(atm)";
  }
  if (reportFlags["pressurecomponents"]) {
    fout << std::setw(outwidth) << "pxx" << std::setw(outwidth) << "pyy"
         << std::setw(outwidth) << "pzz";
  }
  if (reportFlags["boxsizecomponents"]) {
    fout << std::setw(outwidth) << "x" << std::setw(outwidth) << "y"
         << std::setw(outwidth) << "z";
  }
  if (reportFlags["density"]) {
    fout << std::setw(outwidth) << "(g/cm^3)";
  }
  fout << std::endl;
}
