// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "RestartSubscriber.h"
#include <fstream>
#include <iomanip>
#include <iostream>

RestartSubscriber::RestartSubscriber(const std::string &fileName) {
  numFramesWritten = 0;
  setFileName(fileName);
  setReportFreq(1000);
  openFile();
}

RestartSubscriber::RestartSubscriber(const std::string &fileName,
                                     int reportFreq) {
  numFramesWritten = 0;
  setFileName(fileName);
  setReportFreq(reportFreq);
  openFile();
}

RestartSubscriber::~RestartSubscriber(void) { fout.close(); }

void RestartSubscriber::update(void) {
  const int rstWidth = 23;
  const int rstPrec = 16;

  // Get positions, get velocities, print !
  auto coords = this->charmmContext->getCoordinatesCharges();
  auto velmass = this->charmmContext->getVelocityMass();
  auto boxdim = this->charmmContext->getBoxDimensions();

  coords.transferFromDevice();
  velmass.transferFromDevice();

  checkForNanValues(coords, velmass, boxdim);

  fout.close();
  fout.open(fileName, std::ios::out);

  // position section
  fout << "!X, Y, Z\n";
  for (int i = 0; i < this->charmmContext->getNumAtoms(); ++i) {
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << coords[i].x << " " << std::setw(rstWidth) << std::scientific
         << std::setprecision(rstPrec) << coords[i].y << " "
         << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << coords[i].z << "\n";
  }
  fout << "\n";

  // velocities section
  fout << "!VX, VY, VZ\n";
  for (int i = 0; i < this->charmmContext->getNumAtoms(); ++i) {
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << velmass[i].x << " " << std::setw(rstWidth) << std::scientific
         << std::setprecision(rstPrec) << velmass[i].y << " "
         << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << velmass[i].z << "\n";
  }
  fout << "\n";

  fout << "!BOXX, BOXY, BOXZ\n";
  fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
       << boxdim[0] << " " << std::setw(rstWidth) << std::scientific
       << std::setprecision(rstPrec) << boxdim[1] << " " << std::setw(rstWidth)
       << std::scientific << std::setprecision(rstPrec) << boxdim[2] << "\n\n";

  // Try downcasting to LP, check if nullptr to continue
  auto langevinPistonIntegrator =
      std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(this->integrator);

  auto langevinThermostatIntegrator =
      std::dynamic_pointer_cast<CudaLangevinThermostatIntegrator>(
          this->integrator);

  if ((langevinPistonIntegrator != nullptr) ||
      (langevinThermostatIntegrator != nullptr)) {
    auto coordsDeltaPrevious = this->integrator->getCoordsDeltaPrevious();
    coordsDeltaPrevious.transferFromDevice();
    fout << "!XOLD, YOLD, ZOLD\n";
    for (int i = 0; i < this->charmmContext->getNumAtoms(); ++i) {
      fout << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << coordsDeltaPrevious[i].x << " "
           << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << coordsDeltaPrevious[i].y << " "
           << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << coordsDeltaPrevious[i].z << "\n";
    }
    fout << "\n";
  }

  if (langevinPistonIntegrator != nullptr) {
    // Let's also add langevin piston related information
    auto pistonDegreesOfFreedom =
        langevinPistonIntegrator->getPistonDegreesOfFreedom();

    auto onStepPistonPosition =
        langevinPistonIntegrator->getOnStepPistonPosition();
    // onStepPistonPosition.transferFromDevice();
    fout << "!onStepPistonPosition\n";
    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      fout << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << onStepPistonPosition[i];
      if (i < pistonDegreesOfFreedom - 1)
        fout << " ";
    }
    fout << "\n\n";

    auto halfStepPistonPosition =
        langevinPistonIntegrator->getHalfStepPistonPosition();
    // halfStepPistonPosition.transferFromDevice();
    fout << "!halfStepPistonPosition\n";
    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      fout << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << halfStepPistonPosition[i];
      if (i < pistonDegreesOfFreedom - 1)
        fout << " ";
    }
    fout << "\n\n";

    auto onStepPistonVelocity =
        langevinPistonIntegrator->getOnStepPistonVelocity();
    // onStepPistonVelocity.transferFromDevice();
    fout << "!onStepPistonVelocity\n";
    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      fout << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << onStepPistonVelocity[i];
      if (i < pistonDegreesOfFreedom - 1)
        fout << " ";
    }
    fout << "\n\n";

    auto halfStepPistonVelocity =
        langevinPistonIntegrator->getHalfStepPistonVelocity();
    // halfStepPistonVelocity.transferFromDevice();
    fout << "!halfStepPistonVelocity" << std::endl;
    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      fout << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << halfStepPistonVelocity[i];
      if (i < pistonDegreesOfFreedom - 1)
        fout << " ";
    }
    fout << "\n\n";

    auto noseHooverPistonMass =
        langevinPistonIntegrator->getNoseHooverPistonMass();
    fout << "!noseHooverPistonMass\n";
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << noseHooverPistonMass << "\n\n";

    auto noseHooverPistonPosition =
        langevinPistonIntegrator->getNoseHooverPistonPosition();
    fout << "!noseHooverPistonPosition\n";
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << noseHooverPistonPosition << "\n\n";

    auto noseHooverPistonVelocity =
        langevinPistonIntegrator->getNoseHooverPistonVelocity();
    fout << "!noseHooverPistonVelocity\n";
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << noseHooverPistonVelocity << "\n\n";

    auto noseHooverPistonVelocityPrevious =
        langevinPistonIntegrator->getNoseHooverPistonVelocityPrevious();
    fout << "!noseHooverPistonVelocityPrevious\n";
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << noseHooverPistonVelocityPrevious << "\n\n";

    auto noseHooverPistonForce =
        langevinPistonIntegrator->getNoseHooverPistonForce();
    fout << "!noseHooverPistonForce\n";
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << noseHooverPistonForce << "\n\n";

    auto noseHooverPistonForcePrevious =
        langevinPistonIntegrator->getNoseHooverPistonForcePrevious();
    fout << "!noseHooverPistonForcePrevious\n";
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << noseHooverPistonForcePrevious << "\n\n";
  }
  fout << std::flush;
  ++numFramesWritten;
}

/* *
std::vector<double> readRestartEntry(const std::string &entry) const {
  std::vector<double> data;

  std::ifstream fin(fileName);
  if (!fin.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open file \"" + fileName +
        "\"\nExiting\n");
  }

  // Find the line containing the section title
  bool found = false;
  while (!fin.eof()) {
    std::string line = "";
    std::getline(fin, line);
    if (line.find(entry) != std : string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::runtime_error(
        "ERROR(RestartSubscriber): Could not find entry \"" + entry +
        "\" in the file \"" + fileName + "\"\nExiting\n");
  }

  return data;
}
* */

std::vector<std::vector<double>> RestartSubscriber::readPositions(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file \"" + fileName +
        "\"\nExiting\n");
    exit(1);
  }

  // Find line containing the section title
  std::string line;
  std::string sectionName = "!X, Y, Z";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw std::invalid_argument("ERROR(RestartSubscriber): Cannot find the "
                                "positions section (!X, Y, Z) in the file " +
                                fileName + "\nExiting\n");
    exit(1);
  }
  // Read and store the section content
  std::vector<std::string> sectionContent;
  while (std::getline(restartFile, line)) {
    if (line.empty()) {
      break;
    }
    sectionContent.push_back(line);
  }
  std::vector<std::vector<double>> positions;
  std::vector<double> position;
  for (auto &line : sectionContent) {
    std::istringstream iss(line);
    double x, y, z;
    iss >> x >> y >> z;
    position = {x, y, z};
    positions.push_back(position);
  }
  return positions;
}

std::vector<std::vector<double>> RestartSubscriber::readVelocities(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }

  // Find line containing the section title
  std::string line;
  std::string sectionName = "!VX, VY, VZ";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the velocities section (!VX, "
        "VY, VZ) in the file " +
        fileName + "\nExiting\n");
    exit(1);
  }
  // Read and store the section content
  std::vector<std::string> sectionContent;
  while (std::getline(restartFile, line)) {
    if (line.empty()) {
      break;
    }
    sectionContent.push_back(line);
  }
  std::vector<std::vector<double>> velocities;
  std::vector<double> velocity;
  for (auto &line : sectionContent) {
    std::istringstream iss(line);
    double x, y, z;
    iss >> x >> y >> z;
    velocity = {x, y, z};
    velocities.push_back(velocity);
  }
  return velocities;
}

std::vector<double> RestartSubscriber::readBoxDimensions(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }

  // Find line containing the section title
  std::string line;
  std::string sectionName = "!BOXX, BOXY, BOXZ";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the box "
        "dimension section (!BOXX, BOXY, BOXZ) in the file " +
        fileName + "\nExiting\n");
    exit(1);
  }
  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  double x, y, z;
  std::istringstream iss(line);
  iss >> x >> y >> z;

  std::vector<double> boxdim = {x, y, z};
  return boxdim;
}

std::vector<std::vector<double>>
RestartSubscriber::readCoordsDeltaPrevious(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }

  // Find line containing the section title
  std::string line;
  std::string sectionName = "!XOLD, YOLD, ZOLD";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the coordsDelta section (!XOLD, "
        "YOLD, ZOLD) in the file " +
        fileName + "\nExiting\n");
    exit(1);
  }
  // Read and store the section content
  std::vector<std::string> sectionContent;
  while (std::getline(restartFile, line)) {
    if (line.empty()) {
      break;
    }
    sectionContent.push_back(line);
  }
  std::vector<std::vector<double>> coordsDeltaPrevious;
  std::vector<double> coordDeltaPrevious;
  for (auto &line : sectionContent) {
    std::istringstream iss(line);
    double x, y, z;
    iss >> x >> y >> z;
    coordDeltaPrevious = {x, y, z};
    coordsDeltaPrevious.push_back(coordDeltaPrevious);
  }
  return coordsDeltaPrevious;
}

std::vector<double> RestartSubscriber::readOnStepPistonPosition(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }

  // Find line containing the section title
  std::string line;
  std::string sectionName = "!onStepPistonPosition";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the "
        "on-step piston position (!onStepPistonPosition) in the file " +
        fileName + "\nExiting\n");
    exit(1);
  }
  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  std::istringstream iss(line);
  std::vector<double> onStepPistonPosition;
  double x;
  while (iss >> x) {
    onStepPistonPosition.push_back(x);
  }
  return onStepPistonPosition;
}

std::vector<double> RestartSubscriber::readHalfStepPistonPosition(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }

  // Find line containing the section title
  std::string line;
  std::string sectionName = "!halfStepPistonPosition";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the "
        "half-step piston position (!halfStepPistonPosition) in the file " +
        fileName + "\nExiting\n");
    exit(1);
  }
  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  std::istringstream iss(line);
  std::vector<double> halfStepPistonPosition;
  double x;
  while (iss >> x) {
    halfStepPistonPosition.push_back(x);
  }
  return halfStepPistonPosition;
}

std::vector<double> RestartSubscriber::readOnStepPistonVelocity(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }

  // Find line containing the section title
  std::string line;
  std::string sectionName = "!onStepPistonVelocity";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the "
        "on-step piston velocity (!onStepPistonVelocity) in the file " +
        fileName + "\nExiting\n");
    exit(1);
  }
  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  std::istringstream iss(line);
  std::vector<double> onStepPistonVelocity;
  double x;
  while (iss >> x) {
    onStepPistonVelocity.push_back(x);
  }
  return onStepPistonVelocity;
}

std::vector<double> RestartSubscriber::readHalfStepPistonVelocity(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }

  // Find line containing the section title
  std::string line;
  std::string sectionName = "!halfStepPistonVelocity";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the "
        "half-step piston velocity (!halfStepPistonVelocity) in the file " +
        fileName + "\nExiting\n");
    exit(1);
  }
  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  std::istringstream iss(line);
  std::vector<double> halfStepPistonVelocity;
  double x;
  while (iss >> x) {
    halfStepPistonVelocity.push_back(x);
  }
  return halfStepPistonVelocity;
}

double RestartSubscriber::readNoseHooverPistonMass(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }
  // Find line containing the section title
  std::string line;
  std::string sectionName = "!noseHooverPistonMass";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the Nose Hoover piston mass "
        "section (!noseHooverPistonMass) in the file " +
        fileName + "\nExiting\n");
    exit(1);
  }

  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  std::istringstream iss(line);
  double pistonPosition;
  iss >> pistonPosition;
  return pistonPosition;
}

double RestartSubscriber::readNoseHooverPistonPosition(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }
  // Find line containing the section title
  std::string line;
  std::string sectionName = "!noseHooverPistonPosition";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::invalid_argument("ERROR(RestartSubscriber): Cannot find the "
                                "Nose Hoover piston position section "
                                "(!noseHooverPistonPosition) in the file " +
                                fileName + "\nExiting\n");
    exit(1);
  }

  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  std::istringstream iss(line);
  double pistonPosition;
  iss >> pistonPosition;
  return pistonPosition;
}

double RestartSubscriber::readNoseHooverPistonVelocity(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }
  // Find line containing the section title
  std::string line;
  std::string sectionName = "!noseHooverPistonVelocity";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::invalid_argument("ERROR(RestartSubscriber): Cannot find the "
                                "Nose Hoover piston velocity section "
                                "(!noseHooverPistonVelocity) in the file " +
                                fileName + "\nExiting\n");
    exit(1);
  }

  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  std::istringstream iss(line);
  double pistonVelocity;
  iss >> pistonVelocity;
  return pistonVelocity;
}

double RestartSubscriber::readNoseHooverPistonVelocityPrevious(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }
  // Find line containing the section title
  std::string line;
  std::string sectionName = "!noseHooverPistonVelocityPrevious";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the "
        "Nose Hoover piston previous velocity section "
        "(!noseHooverPistonVelocityPrevious) in the file " +
        fileName + "\nExiting\n");
    exit(1);
  }

  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  std::istringstream iss(line);
  double pistonVelocity;
  iss >> pistonVelocity;
  return pistonVelocity;
}

double RestartSubscriber::readNoseHooverPistonForce(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }
  // Find line containing the section title
  std::string line;
  std::string sectionName = "!noseHooverPistonForce";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::invalid_argument("ERROR(RestartSubscriber): Cannot find the "
                                "Nose Hoover piston force section "
                                "(!noseHooverPistonForce) in the file " +
                                fileName + "\nExiting\n");
    exit(1);
  }

  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  std::istringstream iss(line);
  double pistonForce;
  iss >> pistonForce;
  return pistonForce;
}

double RestartSubscriber::readNoseHooverPistonForcePrevious(void) const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }
  // Find line containing the section title
  std::string line;
  std::string sectionName = "!noseHooverPistonForcePrevious";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the "
        "Nose Hoover piston previous force section "
        "(!noseHooverPistonForcePrevious) in the file " +
        fileName + "\nExiting\n");
    exit(1);
  }

  // Read and store the section content
  std::vector<std::string> sectionContent;
  std::getline(restartFile, line);
  std::istringstream iss(line);
  double pistonForce;
  iss >> pistonForce;
  return pistonForce;
}

/* *
void RestartSubscriber::getRestartContent(std::string fileName,
                                          std::string sectionName) {
  std::fstream restartFile(fileName, std::ios::in);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }

  // Find line containing the section title
  std::string line;
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot find the section " + sectionName +
        " in the file " + fileName + "\nExiting\n");
    exit(1);
  }
  // Read and store the section content
  std::vector<std::string> sectionContent;
  while (std::getline(restartFile, line)) {
    if (line.empty()) {
      break;
      sectionContent.push_back(line);
    }
  }
}
* */

void RestartSubscriber::readRestart(void) {
  if (!hasCharmmContext) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): To read a restart file, "
        "RestartSubscriber needs to be linked to a CharmmContext object\n");
    exit(1);
  }
  if (!hasIntegrator) {
    throw std::invalid_argument("ERROR(RestartSubscriber): To read a restart "
                                "file, RestartSubscriber needs to be linked "
                                "to an Integrator object\n");
    exit(1);
  }

  charmmContext->setCoordinates(readPositions());
  charmmContext->assignVelocities(readVelocities());
  charmmContext->setBoxDimensions(readBoxDimensions());

  // Try downcasting to Langevin thermostat, check if nullptr to know if
  // pertinent or not
  auto langevinThermostatIntegrator =
      std::dynamic_pointer_cast<CudaLangevinThermostatIntegrator>(
          this->integrator);

  // Try downcasting to LP, check if nullptr to continue
  auto langevinPistonIntegrator =
      std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(this->integrator);

  // If LP or LT, need coords deltaprevious
  if ((langevinPistonIntegrator != nullptr) ||
      (langevinThermostatIntegrator != nullptr)) {
    integrator->setCoordsDeltaPrevious(readCoordsDeltaPrevious());
  }

  if (langevinPistonIntegrator != nullptr) {
    // Langevin Piston : piston variables. Verify they are the right size.
    // Also, box dim
    langevinPistonIntegrator->setBoxDimensions(readBoxDimensions());
    int nPistonDofs = langevinPistonIntegrator->getPistonDegreesOfFreedom();
    std::vector<double> ospp = readOnStepPistonPosition(),
                        hspp = readHalfStepPistonPosition(),
                        ospv = readOnStepPistonVelocity(),
                        hspv = readHalfStepPistonVelocity();
    if (ospp.size() != nPistonDofs) {
      throw std::invalid_argument(
          "ERROR(RestartSubscriber): Langevin-piston variable sizes (for "
          "on-step piston position) read do not correspond to the Crystal type "
          "chosen. dof read: " +
          std::to_string(ospp.size()) +
          ", dof expected: " + std::to_string(nPistonDofs));
    }
    if (hspp.size() != nPistonDofs) {
      throw std::invalid_argument(
          "ERROR(RestartSubscriber): Langevin-piston variable sizes (for "
          "half-step piston position) read do not correspond to the Crystal "
          "type chosen. dof read: " +
          std::to_string(hspp.size()) +
          ", dof expected: " + std::to_string(nPistonDofs));
    }
    if (ospv.size() != nPistonDofs) {
      throw std::invalid_argument(
          "ERROR(RestartSubscriber): Langevin-piston variable sizes (for "
          "on-step piston velocity) read do not correspond to the Crystal "
          "type chosen. dof read: " +
          std::to_string(ospv.size()) +
          ", dof expected: " + std::to_string(nPistonDofs));
    }
    if (hspv.size() != nPistonDofs) {
      throw std::invalid_argument(
          "ERROR(RestartSubscriber): Langevin-piston variable sizes (for "
          "half-step piston velocity) read do not correspond to the Crystal "
          "type chosen. dof read: " +
          std::to_string(ospv.size()) +
          ", dof expected: " + std::to_string(nPistonDofs));
    }
    langevinPistonIntegrator->setOnStepPistonPosition(ospp);
    langevinPistonIntegrator->setHalfStepPistonPosition(hspp);
    langevinPistonIntegrator->setOnStepPistonVelocity(ospv);
    langevinPistonIntegrator->setHalfStepPistonVelocity(hspv);

    // Nose-Hoover thermostat piston variables
    langevinPistonIntegrator->setNoseHooverPistonMass(
        readNoseHooverPistonMass());
    langevinPistonIntegrator->setNoseHooverPistonPosition(
        readNoseHooverPistonPosition());
    langevinPistonIntegrator->setNoseHooverPistonVelocity(
        readNoseHooverPistonVelocity());
    langevinPistonIntegrator->setNoseHooverPistonVelocityPrevious(
        readNoseHooverPistonVelocityPrevious());
    langevinPistonIntegrator->setNoseHooverPistonForce(
        readNoseHooverPistonForce());
    langevinPistonIntegrator->setNoseHooverPistonForcePrevious(
        readNoseHooverPistonForcePrevious());
  }
}

void RestartSubscriber::openFile() {
  checkPath(fileName);
  // If the file already exists, open in read mode.
  // If not, open in write mode.
  std::fstream ifile(fileName);
  if (ifile) { // file opened -> exists
    fout.open(fileName, std::ios::in | std::ios::out);
  } else { // ifile not opened -> does not exist previously -> write mode
    fout.open(fileName, std::ios::out);
  }
}

void RestartSubscriber::checkForNanValues(CudaContainer<double4> coords,
                                          CudaContainer<double4> velmass,
                                          std::vector<double> boxdim) {
  if (std::isnan(boxdim[0]) || std::isnan(boxdim[1]) || std::isnan(boxdim[2])) {
    std::cout << "Nan value found in box dimensions" << std::endl;
    std::cout << "Exiting..." << std::endl;
    exit(1);
  }
  for (int i = 0; i < this->charmmContext->getNumAtoms(); i++) {
    if (std::isnan(coords[i].x) || std::isnan(coords[i].y) ||
        std::isnan(coords[i].z)) {
      std::cout << "Nan value found in coordinates at index " << i << std::endl;
      std::cout << "Exiting..." << std::endl;
      exit(1);
    }

    if (std::isnan(velmass[i].x) || std::isnan(velmass[i].y) ||
        std::isnan(velmass[i].z)) {
      std::cout << "Nan value found in velocities at index " << i << std::endl;
      std::cout << "Exiting..." << std::endl;
      exit(1);
    }
  }
}
