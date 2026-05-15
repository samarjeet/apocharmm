// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "StateSubscriber.h"

#include "CharmmContext.h"
#include "CudaLangevinPistonIntegrator.h"
#include "str_utils.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

StateSubscriber::StateSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  this->initialize();
}

StateSubscriber::StateSubscriber(const std::string &fileName,
                                 const int reportFrequency)
    : Subscriber(fileName, reportFrequency) {
  this->initialize();
}

StateSubscriber::~StateSubscriber(void) { m_FileStream.close(); }

void StateSubscriber::update(void) {
  // TODO : add HFCTE
  if (!m_HeaderWritten)
    this->writeHeader();

  // Update energies
  m_CharmmContext->calculatePotentialEnergy(true, false);
  CudaContainer<double> peContainer = m_CharmmContext->getPotentialEnergy();
  peContainer.transferFromDevice();
  double pe = peContainer.getHostArray()[0];

  CudaContainer<double> keContainer = m_CharmmContext->getKineticEnergy();
  keContainer.transferFromDevice();
  double ke = keContainer.getHostArray()[0];

  std::vector<double> boxDim = m_CharmmContext->getBoxDimensions();

  // Compute the density
  /* Take the mass
  Convert into g
  Get the volume
  Convert it into cm^3
  Dive the converted values by each other m/v
  */
  CudaContainer<double4> velmassCC = m_CharmmContext->getVelocityMass();
  velmassCC.transferFromDevice();
  std::vector<double4> velmass = velmassCC.getHostArray();
  double densityMass = 0.0;
  for (int i = 0; i < velmass.size(); i++)
    densityMass += 1.0 / velmass[i].w;
  double convertedDensityMass = densityMass * 1.660540199e-24;

  double densityVolume = m_CharmmContext->getVolume();
  double convertedDensityVolume = densityVolume * 1e-24;

  double density = convertedDensityMass / convertedDensityVolume;

  //  Compute time spent
  float time = this->computeTime();
  float temperature = m_CharmmContext->computeTemperature();

  m_FileStream << std::left << std::setw(m_OutWidth) << time;
  if (m_ReportFlags["potentialenergy"])
    m_FileStream << std::setw(m_OutWidth) << pe;
  if (m_ReportFlags["kineticenergy"])
    m_FileStream << std::setw(m_OutWidth) << ke;
  if (m_ReportFlags["totalenergy"])
    m_FileStream << std::setw(m_OutWidth) << (ke + pe);
  if (m_ReportFlags["temperature"])
    m_FileStream << std::setw(m_OutWidth) << temperature;
  if (m_ReportFlags["pressurescalar"]) {
    try {
      auto langevinPistonIntegrator =
          std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(m_Integrator);
      // double pressure = langevinPistonIntegrator->getPressureScalar();
      double pressure = 0.0;
      m_FileStream << std::setw(m_OutWidth) << pressure;
    } catch (const std::exception &e) {
      throw std::invalid_argument(
          "Current integrator does not support pressure calculation");
    }
  }
  if (m_ReportFlags["pressurecomponents"]) {
    try {
      auto langevinPistonIntegrator =
          std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(m_Integrator);
      // std::vector<double> pressureTensor =
      //     langevinPistonIntegrator->getPressureTensor();
      std::vector<double> pressureTensor(9, 0.0);
      m_FileStream << std::setw(m_OutWidth) << pressureTensor[0]
                   << std::setw(m_OutWidth) << pressureTensor[4]
                   << std::setw(m_OutWidth) << pressureTensor[8];
    } catch (const std::exception &e) {
      throw std::invalid_argument(
          "Current integrator does not support pressure calculation");
    }
  }
  if (m_ReportFlags["boxdimensions"]) {
    m_FileStream << std::setw(m_OutWidth) << boxDim[0] << std::setw(m_OutWidth)
                 << boxDim[1] << std::setw(m_OutWidth) << boxDim[2];
  }
  if (m_ReportFlags["density"])
    m_FileStream << std::setw(m_OutWidth) << density;

  m_FileStream << std::endl;
  m_NumFramesWritten++;

  return;
}

void StateSubscriber::setReportPotentialEnergy(
    const bool reportPotentialEnergy) {
  m_ReportFlags["potentialenergy"] = reportPotentialEnergy;
  return;
}

void StateSubscriber::setReportKineticEnergy(const bool reportKineticEnergy) {
  m_ReportFlags["kineticenergy"] = reportKineticEnergy;
  return;
}

void StateSubscriber::setReportTotalEnergy(const bool reportTotalEnergy) {
  m_ReportFlags["totalenergy"] = reportTotalEnergy;
  return;
}

void StateSubscriber::setReportTemperature(const bool reportTemperature) {
  m_ReportFlags["temperature"] = reportTemperature;
  return;
}

void StateSubscriber::setReportPressureComponents(
    const bool reportPressureComponents) {
  m_ReportFlags["pressurecomponents"] = reportPressureComponents;
  return;
}

void StateSubscriber::setReportPressureScalar(const bool reportPressureScalar) {
  m_ReportFlags["pressurescalar"] = reportPressureScalar;
  return;
}

void StateSubscriber::setReportBoxSizeComponents(
    const bool reportBoxSizeComponents) {
  m_ReportFlags["boxsizecomponents"] = reportBoxSizeComponents;
  return;
}

void StateSubscriber::setReportVolume(const bool reportVolume) {
  m_ReportFlags["volume"] = reportVolume;
  return;
}

const std::map<std::string, bool> &StateSubscriber::getReportFlags(void) const {
  return m_ReportFlags;
}

std::map<std::string, bool> &StateSubscriber::getReportFlags(void) {
  return m_ReportFlags;
}

void StateSubscriber::readReportFlags(
    const std::vector<std::string> &reportStrings) {
  std::vector<std::string> tokens = reportStrings;
  for (std::string &token : tokens) {
    apo::trim_ip(token);
    apo::to_lower_ip(token);
  }
  this->setReportFlags(tokens);
  return;
}

void StateSubscriber::readReportFlags(const std::string &reportString) {
  std::vector<std::string> tokens = apo::split(reportString, ",");
  this->readReportFlags(tokens);
  return;
}

void StateSubscriber::initialize(void) {
  m_NumFramesWritten = 0;

  // Default values to be reported
  m_ReportFlags = {{"potentialenergy", true},
                   {"kineticenergy", true},
                   {"totalenergy", true},
                   {"temperature", true},
                   {"pressurecomponents", false},
                   {"pressurescalar", true},
                   {"boxsizecomponents", false},
                   {"density", false},
                   {"volume", false}};

  m_OutWidth = 15;

  for (int i = 0; i < m_OutWidth; i++)
    m_Spacing += " ";

  m_HeaderWritten = false;
}

float StateSubscriber::computeTime(void) {
  float dt = static_cast<float>(m_Integrator->getTimeStep());
  float t = (m_NumFramesWritten + 1) * m_ReportFrequency * dt;
  return t;
}

void StateSubscriber::setReportFlags(
    const std::vector<std::string> &reportFlags) {
  // Set all flags to false
  for (auto &[key, value] : m_ReportFlags)
    value = false;

  // Check if any of the input flags is "all"
  for (const std::string &reportFlag : reportFlags) {
    if (reportFlag == "all") {
      for (auto &[key, value] : m_ReportFlags)
        value = true;
      return;
    }
  }

  // Set only the found flags to true
  for (const std::string &reportFlag : reportFlags) {
    if (m_ReportFlags.find(reportFlag) != m_ReportFlags.end())
      m_ReportFlags[reportFlag] = true;
    else {
      std::cerr << "Warning: \"" << reportFlag << "\" is not a valid flag"
                << std::endl;
    }
  }

  return;
}

void StateSubscriber::writeHeader(void) {
  m_FileStream << std::left << std::setw(m_OutWidth) << "# Time";
  if (m_ReportFlags["potentialenergy"])
    m_FileStream << std::setw(m_OutWidth) << "Pot. ene";
  if (m_ReportFlags["kineticenergy"])
    m_FileStream << std::setw(m_OutWidth) << "Kin. ene";
  if (m_ReportFlags["totalenergy"])
    m_FileStream << std::setw(m_OutWidth) << "Total ene";
  if (m_ReportFlags["temperature"])
    m_FileStream << std::setw(m_OutWidth) << "Temperature";
  if (m_ReportFlags["pressurescalar"])
    m_FileStream << std::setw(m_OutWidth) << "Pressure";
  if (m_ReportFlags["pressurecomponents"])
    m_FileStream << std::setw(3 * m_OutWidth) << m_Spacing + "Pressure comp.";
  if (m_ReportFlags["boxsizecomponents"])
    m_FileStream << std::setw(3 * m_OutWidth) << m_Spacing + "Box (AA)";
  if (m_ReportFlags["density"])
    m_FileStream << std::setw(m_OutWidth) << "Density";

  m_FileStream << std::endl;
  m_FileStream << std::left << std::setw(m_OutWidth) << "# (ps)";
  if (m_ReportFlags["potentialenergy"])
    m_FileStream << std::setw(m_OutWidth) << "(kcal/mol)";
  if (m_ReportFlags["kineticenergy"])
    m_FileStream << std::setw(m_OutWidth) << "(kcal/mol)";
  if (m_ReportFlags["totalenergy"])
    m_FileStream << std::setw(m_OutWidth) << "(kcal/mol)";
  if (m_ReportFlags["temperature"])
    m_FileStream << std::setw(m_OutWidth) << "(K)";
  if (m_ReportFlags["pressurescalar"])
    m_FileStream << std::setw(m_OutWidth) << "(atm)";
  if (m_ReportFlags["pressurecomponents"]) {
    m_FileStream << std::setw(m_OutWidth) << "pxx" << std::setw(m_OutWidth)
                 << "pyy" << std::setw(m_OutWidth) << "pzz";
  }
  if (m_ReportFlags["boxsizecomponents"]) {
    m_FileStream << std::setw(m_OutWidth) << "x" << std::setw(m_OutWidth) << "y"
                 << std::setw(m_OutWidth) << "z";
  }
  if (m_ReportFlags["density"])
    m_FileStream << std::setw(m_OutWidth) << "(g/cm^3)";
  m_FileStream << std::endl;

  m_HeaderWritten = true;

  return;
}
