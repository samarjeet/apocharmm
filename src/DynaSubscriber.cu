// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II
//
// ENDLICENSE

#include "DynaSubscriber.h"

#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaLeapFrogIntegrator.h"
#include "CudaNoseHooverThermostatIntegrator.h"
#include "CudaVMMSVelocityVerletIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "CudaVerletIntegrator.h"

#include <iomanip>

#include <iostream> // TEMP

DynaSubscriber::DynaSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  m_HasWrittenHeader = false;
}

DynaSubscriber::DynaSubscriber(const std::string &fileName,
                               const int reportFrequency)
    : Subscriber(fileName, reportFrequency) {
  m_HasWrittenHeader = false;
}

DynaSubscriber::~DynaSubscriber(void) {
  m_FileStream.flush();
  m_FileStream.close();
}

void DynaSubscriber::update(void) {
  if (!m_HasWrittenHeader)
    this->writeHeader();
  if (m_Integrator == nullptr) {
    throw std::invalid_argument("DynaSubscriber has not been subscribed to"
                                " an integrator!");
  }

  auto lp =
      std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(m_Integrator);
  auto lt =
      std::dynamic_pointer_cast<CudaLangevinThermostatIntegrator>(m_Integrator);
  auto lf = std::dynamic_pointer_cast<CudaLeapFrogIntegrator>(m_Integrator);
  auto nht = std::dynamic_pointer_cast<CudaNoseHooverThermostatIntegrator>(
      m_Integrator);
  auto vv =
      std::dynamic_pointer_cast<CudaVelocityVerletIntegrator>(m_Integrator);
  auto v = std::dynamic_pointer_cast<CudaVerletIntegrator>(m_Integrator);
  auto vmmsvv =
      std::dynamic_pointer_cast<CudaVMMSVelocityVerletIntegrator>(m_Integrator);

  if (lp != nullptr) { // CudaLangevinPistonIntegrator
    int iStep = lp->getCurrentPropagatedStep() + 1;
    double iTime =
        static_cast<double>(iStep) * static_cast<double>(lp->getTimeStep());
    double totke = m_CharmmContext->getKineticEnergy();
    CudaContainer<double> totpecc = m_CharmmContext->getPotentialEnergy();
    totpecc.transferFromDevice();
    double totpe = totpecc.getHostArray()[0];
    float temp = m_CharmmContext->computeTemperature();
    double pressi = lp->getInstantaneousPressureScalar();

    // Calculate volume
    std::vector<double> boxDims = m_CharmmContext->getBoxDimensions();
    double volu = -9999.9999;
    switch (lp->getCrystalType()) {
    case CRYSTAL::CUBIC:
      volu = boxDims[0] * boxDims[0] * boxDims[0];
      break;
    case CRYSTAL::TETRAGONAL:
      volu = boxDims[0] * boxDims[0] * boxDims[1];
      break;
    case CRYSTAL::ORTHORHOMBIC:
      volu = boxDims[0] * boxDims[1] * boxDims[2];
      break;
    default:
      break;
    }

    std::shared_ptr<ForceManager> fm = m_CharmmContext->getForceManager();
    std::map<std::string, double> energyDecompositionMap =
        fm->getEnergyComponents();
    double bond = energyDecompositionMap["bond"];
    double angl = energyDecompositionMap["angle"];
    double urey = energyDecompositionMap["ureyb"];
    double dihe = energyDecompositionMap["dihe"];
    double impr = energyDecompositionMap["imdihe"];
    double vdwl = energyDecompositionMap["vdw"];
    double elec = energyDecompositionMap["elec"];
    double ewks = energyDecompositionMap["ewks"];
    double ewse = energyDecompositionMap["ewse"];
    double ewex = energyDecompositionMap["ewex"];
    CudaContainer<double> viricc = fm->getVirial();
    viricc.transferFromDevice();
    double viri = viricc.getHostArray()[0];

    m_FileStream << "DYNA>" << std::setw(9) << iStep << std::setw(13)
                 << std::fixed << std::setprecision(5) << iTime << std::setw(13)
                 << std::fixed << std::setprecision(5) << totpe + totke
                 << std::setw(13) << std::fixed << std::setprecision(5) << totke
                 << std::setw(13) << std::fixed << std::setprecision(5) << totpe
                 << std::setw(13) << std::fixed << std::setprecision(5) << temp
                 << "\n";
    m_FileStream << "DYNA PROP>    " << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << "\n";
    m_FileStream << "DYNA INTERN>  " << std::setw(13) << std::fixed
                 << std::setprecision(5) << bond << std::setw(13) << std::fixed
                 << std::setprecision(5) << angl << std::setw(13) << std::fixed
                 << std::setprecision(5) << urey << std::setw(13) << std::fixed
                 << std::setprecision(5) << dihe << std::setw(13) << std::fixed
                 << std::setprecision(5) << impr << "\n";
    m_FileStream << "DYNA CROSS>   " << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << "\n";
    m_FileStream << "DYNA EXTERN>  " << std::setw(13) << std::fixed
                 << std::setprecision(5) << vdwl << std::setw(13) << std::fixed
                 << std::setprecision(5) << elec << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << "\n";
    m_FileStream << "DYNA EWALD>   " << std::setw(13) << std::fixed
                 << std::setprecision(5) << ewks << std::setw(13) << std::fixed
                 << std::setprecision(5) << ewse << std::setw(13) << std::fixed
                 << std::setprecision(5) << ewex << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << "\n";
    m_FileStream << "DYNA PRESS>   " << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << viri << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << pressi << std::setw(13)
                 << std::fixed << std::setprecision(5) << volu << "\n";
    m_FileStream << "DYNA XTLE>                 " << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << std::setw(13) << std::fixed
                 << std::setprecision(5) << 0.0 << "\n";
  } else if (lt != nullptr) { // CudaLangevinThermostatIntegrator
    throw std::invalid_argument("DynaSubscriber::ERROR: Langevin Thermostat "
                                "Integrator is not supported!\n");
  } else if (lf != nullptr) { // CudaLeapFrogIntegrator
    throw std::invalid_argument(
        "DynaSubscriber::ERROR: Leap Frog Integrator is not supported!\n");
  } else if (nht != nullptr) { // CudaNoseHooverThermostatIntegrator
    throw std::invalid_argument("DynaSubscriber::ERROR: Nose-Hoover Thermostat "
                                "Integrator is not supported!\n");
  } else if (vv != nullptr) { // CudaVelocityVerletIntegrator
    throw std::invalid_argument("DynaSubscriber::ERROR: Velocity Verlet "
                                "Integrator is not supported!\n");
  } else if (v != nullptr) { // CudaVerletIntegrator
    throw std::invalid_argument(
        "DynaSubscriber::ERROR: Verlet Integrator is not supported!\n");
  } else if (vmmsvv != nullptr) { // CudaVMMSVelocityVerletIntegrator
    throw std::invalid_argument("DynaSubscriber::ERROR: VMMS Velocity Verlet "
                                "Integrator is not supported!\n");
  } else
    std::cout << "DynaSubscriber::WARNING: Unrecogonized integrator!"
              << std::endl;

  m_FileStream
      << " ----------       ---------    ---------    ---------    ---------"
      << "    ---------\n"
      << std::endl;

  return;
}

void DynaSubscriber::writeHeader(void) {
  m_FileStream
      << "DYNA DYN: Step         Time      TOTEner        TOTKe       ENERgy"
      << "  TEMPerature\n";
  m_FileStream
      << "DYNA PROP:             GRMS      HFCTote        HFCKe       EHFCor"
      << "        VIRKe\n";
  m_FileStream
      << "DYNA INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals"
      << "    IMPRopers\n";
  m_FileStream
      << "DYNA CROSS:           CMAPs        PMF1D        PMF2D        PRIMO"
      << "             \n";
  m_FileStream
      << "DYNA EXTERN:        VDWaals         ELEC       HBONds          ASP"
      << "         USER\n";
  m_FileStream
      << "DYNA EWALD:          EWKSum       EWSElf       EWEXcl       EWQCor"
      << "       EWUTil\n";
  m_FileStream
      << "DYNA PRESS:            VIRE         VIRI       PRESSE       PRESSI"
      << "       VOLUme\n";
  m_FileStream
      << "DYNA XTLE:                       XTLTe         SURFtension  XTLPe "
      << "       XTLtemp\n";
  m_FileStream
      << " ----------       ---------    ---------    ---------    ---------"
      << "    ---------\n";

  m_HasWrittenHeader = true;

  return;
}
