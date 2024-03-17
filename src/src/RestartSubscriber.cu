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

RestartSubscriber::~RestartSubscriber() {
  // std::cout << "Trying to close the state subscriber\n";
  fout.close();
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

void RestartSubscriber::update() {
  const int rstWidth = 32;
  const int rstPrec = 12;

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
    auto coordsDelta = this->integrator->getCoordsDeltaPrevious();
    coordsDelta.transferFromDevice();
    fout << "!XOLD, YOLD, ZOLD\n";
    for (int i = 0; i < this->charmmContext->getNumAtoms(); ++i) {
      fout << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << coordsDelta[i].x << " "
           << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << coordsDelta[i].y << " "
           << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << coordsDelta[i].z << "\n";
    }
    fout << "\n";
  }

  if (langevinPistonIntegrator != nullptr) {
    // Let's also add langevin piston related information
    auto onStepPistonVelocity =
        langevinPistonIntegrator->getOnStepPistonVelocity();
    // onStepPistonVelocity.transferFromDevice();

    fout << "\n!onStepPistonVelocity\n";
    auto pistonDegreesOfFreedom =
        langevinPistonIntegrator->getPistonDegreesOfFreedom();
    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      fout << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << onStepPistonVelocity[i];
      if (i < pistonDegreesOfFreedom - 1)
        fout << " ";
    }
    fout << "\n";

    auto onStepPistonPosition =
        langevinPistonIntegrator->getOnStepPistonPosition();
    // onStepPistonPosition.transferFromDevice();

    fout << "\n!onStepPistonPosition\n";
    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      fout << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << onStepPistonPosition[i];
      if (i < pistonDegreesOfFreedom - 1)
        fout << " ";
    }
    fout << "\n";

    // Not needed
    //    auto halfStepPistonVelocity =
    //        langevinPistonIntegrator->getHalfStepPistonVelocity();
    //    // halfStepPistonVelocity.transferFromDevice();
    //
    //    fout << "\n!Half step Piston Velocity" << std::endl;
    //    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
    //      fout << halfStepPistonVelocity[i] << " ";
    //    }
    //    fout << std::endl;

    auto halfStepPistonPosition =
        langevinPistonIntegrator->getHalfStepPistonPosition();
    // halfStepPistonPosition.transferFromDevice();

    fout << "\n!halfStepPistonPosition\n";
    for (int i = 0; i < pistonDegreesOfFreedom; ++i) {
      fout << std::setw(rstWidth) << std::scientific
           << std::setprecision(rstPrec) << halfStepPistonPosition[i];
      if (i < pistonDegreesOfFreedom - 1)
        fout << " ";
    }
    fout << "\n";

    auto pistonNoseHooverPosition =
        langevinPistonIntegrator->getPistonNoseHooverPosition();
    fout << "\n!pistonNoseHooverPosition\n";
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << pistonNoseHooverPosition << "\n";
    auto pistonNoseHooverVelocityPrevious =
        langevinPistonIntegrator->getPistonNoseHooverVelocityPrevious();
    fout << "\n!pistonNoseHooverVelocity\n";
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << pistonNoseHooverVelocityPrevious << "\n";
    auto pistonNoseHooverForcePrevious =
        langevinPistonIntegrator->getPistonNoseHooverForcePrevious();
    fout << "\n!pistonNoseHooverForce\n";
    fout << std::setw(rstWidth) << std::scientific << std::setprecision(rstPrec)
         << pistonNoseHooverForcePrevious << "\n";
  }
  fout << std::flush;
  ++numFramesWritten;
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

// this generic function is actually not used.
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

std::vector<double> RestartSubscriber::readBoxDimensions() const {
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

std::vector<double> RestartSubscriber::readOnStepPistonPosition() const {
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

std::vector<double> RestartSubscriber::readHalfStepPistonPosition() const {
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

std::vector<double> RestartSubscriber::readOnStepPistonVelocity() const {
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

std::vector<std::vector<double>> RestartSubscriber::readPositions() const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
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

std::vector<std::vector<double>>
RestartSubscriber::readCoordsDeltaPrevious() const {
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
  std::vector<std::vector<double>> coordsDelta;
  std::vector<double> coordDelta;
  for (auto &line : sectionContent) {
    std::istringstream iss(line);
    double x, y, z;
    iss >> x >> y >> z;
    coordDelta = {x, y, z};
    coordsDelta.push_back(coordDelta);
  }
  return coordsDelta;
}

std::vector<std::vector<double>> RestartSubscriber::readVelocities() const {
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
};

void RestartSubscriber::readRestart() {
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
    std::vector<double> ospv = readOnStepPistonVelocity(),
                        ospp = readOnStepPistonPosition(),
                        hspp = readHalfStepPistonPosition();
    if (ospv.size() != nPistonDofs) {
      throw std::invalid_argument(
          "ERROR(RestartSubscriber): Langevin-piston variable sizes (for "
          "on-step piston velocity) read do not correspond to the Crystal "
          "type chosen. dof read: " +
          std::to_string(ospv.size()) +
          ", dof expected: " + std::to_string(nPistonDofs));
    }
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
    langevinPistonIntegrator->setOnStepPistonVelocity(ospv);
    langevinPistonIntegrator->setOnStepPistonPosition(ospp);
    langevinPistonIntegrator->setHalfStepPistonPosition(hspp);

    // Nose-Hoover thermostat- piston variables
    langevinPistonIntegrator->setPistonNoseHooverPosition(
        readNoseHooverPistonPosition());
    langevinPistonIntegrator->setPistonNoseHooverVelocityPrevious(
        readNoseHooverPistonVelocity());
    langevinPistonIntegrator->setPistonNoseHooverForcePrevious(
        readNoseHooverPistonForce());
  }
};

double RestartSubscriber::readNoseHooverPistonPosition() const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }
  // Find line containing the section title
  std::string line;
  std::string sectionName = "!pistonNoseHooverPosition";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::invalid_argument("ERROR(RestartSubscriber): Cannot find the "
                                "Piston Nose Hoover Position section "
                                "(!pistonNoseHooverPosition) in the file " +
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
};

double RestartSubscriber::readNoseHooverPistonVelocity() const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }
  // Find line containing the section title
  std::string line;
  std::string sectionName = "!pistonNoseHooverVelocity";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::invalid_argument("ERROR(RestartSubscriber): Cannot find the "
                                "Nose Hoover Piston Velocity section "
                                "(!pistonNoseHooverVelocity) in the file " +
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
};

double RestartSubscriber::readNoseHooverPistonForce() const {
  std::ifstream restartFile(fileName);
  if (!restartFile.is_open()) {
    throw std::invalid_argument(
        "ERROR(RestartSubscriber): Cannot open the file " + fileName +
        "\nExiting\n");
    exit(1);
  }
  // Find line containing the section title
  std::string line;
  std::string sectionName = "!pistonNoseHooverForce";
  bool found = false;
  while (std::getline(restartFile, line)) {
    if (line.find(sectionName) != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::invalid_argument("ERROR(RestartSubscriber): Cannot find the "
                                "Nose Hoover Piston Force section "
                                "(!pistonNoseHooverForce) in the file " +
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
};
