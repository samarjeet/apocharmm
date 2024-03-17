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
#include "CudaVelocityVerletIntegrator.h"
#include "CudaVerletIntegrator.h"
#include "CudaVMMSVelocityVerletIntegrator.h"

#include <iomanip>

#include <iostream> // TEMP

DynaSubscriber::DynaSubscriber(const std::string &_fileName) {
  setFileName(fileName); // Subscriber base class method
  setReportFreq(1000);   // Subscriber base class method
  openFile();            // Subscriber base class method
  m_HasWrittenHeader = false;
}

DynaSubscriber::DynaSubscriber(const std::string &_fileName,
                               const int _reportFreq) {
  setFileName(_fileName);     // Subscriber base class method
  setReportFreq(_reportFreq); // Subscriber base class method
  openFile();                 // Subscriber base class method
  m_HasWrittenHeader = false;
}

DynaSubscriber::~DynaSubscriber(void) {
  fout.flush();
  fout.close();
}

void DynaSubscriber::update(void) {
  if (not m_HasWrittenHeader)
    this->writeHeader();
  if (not hasIntegrator) {
    throw std::invalid_argument("DynaSubscriber has not been subscribed to"
                                " an integrator!");
  }

  auto lp =
    std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(integrator);
  auto lt =
    std::dynamic_pointer_cast<CudaLangevinThermostatIntegrator>(integrator);
  auto lf =
    std::dynamic_pointer_cast<CudaLeapFrogIntegrator>(integrator);
  auto nht =
    std::dynamic_pointer_cast<CudaNoseHooverThermostatIntegrator>(integrator);
  auto vv =
    std::dynamic_pointer_cast<CudaVelocityVerletIntegrator>(integrator);
  auto v =
    std::dynamic_pointer_cast<CudaVerletIntegrator>(integrator);
  auto vmmsvv =
    std::dynamic_pointer_cast<CudaVMMSVelocityVerletIntegrator>(integrator);

  if (lp != nullptr) { // CudaLangevinPistonIntegrator
    int iStep = lp->getCurrentPropagatedStep() + 1;
    double iTime = static_cast<double>(iStep) *
      static_cast<double>(lp->getTimeStep());
    double totke = charmmContext->getKineticEnergy();
    CudaContainer<double> totpecc = charmmContext->getPotentialEnergy();
    totpecc.transferFromDevice();
    double totpe = totpecc.getHostArray()[0];
    double temp = charmmContext->getTemperature();

    fout << "DYNA>" << std::setw(9) << iStep;
    fout << std::setw(13) << std::fixed << std::setprecision(5) << iTime;
    fout << std::setw(13) << std::fixed << std::setprecision(5)
         << totpe + totke;
    fout << std::setw(13) << std::fixed << std::setprecision(5)
         << totke;
    fout << std::setw(13) << std::fixed << std::setprecision(5)
         << totpe;
    fout << std::setw(13) << std::fixed << std::setprecision(5)
         << temp;
    fout << "\n";
  }
  else if (lt != nullptr) { // CudaLangevinThermostatIntegrator
    throw std::invalid_argument("DynaSubscriber::ERROR: Langevin Thermostat"
                                " Integrator is not supported!\n");
    exit(1);
  }
  else if (lf != nullptr) { // CudaLeapFrogIntegrator
    throw std::invalid_argument("DynaSubscriber::ERROR: Leap Frog Integrator"
                                " is not supported!\n");
    exit(1);
  }
  else if (nht != nullptr) { // CudaNoseHooverThermostatIntegrator
    throw std::invalid_argument("DynaSubscriber::ERROR: Nose-Hoover"
                                " Thermostat Integrator is not supported!\n");
    exit(1);
  }
  else if (vv != nullptr) { // CudaVelocityVerletIntegrator
    throw std::invalid_argument("DynaSubscriber::ERROR: Velocity Verlet"
                                " Integrator is not supported!\n");
    exit(1);
  }
  else if (v != nullptr) { // CudaVerletIntegrator
    throw std::invalid_argument("DynaSubscriber::ERROR: Verlet Integrator"
                                " is not supported!\n");
    exit(1);
  }
  else if (vmmsvv != nullptr) { // CudaVMMSVelocityVerletIntegrator
    throw std::invalid_argument("DynaSubscriber::ERROR: VMMS Velocity Verlet"
                          " Integrator is not supported!\n");
    exit(1);
  }
  else
    std::cout << "DynaSubscriber::WARNING: Unrecogonized integrator!\n";

  fout << " ----------       ---------    ---------    ---------    ---------"
       << "    ---------\n" << std::endl;

  return;
}

void DynaSubscriber::writeHeader(void) {
  fout << "DYNA DYN: Step         Time      TOTEner        TOTKe       ENERgy"
       << "  TEMPerature\n";
  fout << "DYNA PROP:             GRMS      HFCTote        HFCKe       EHFCor"
       << "        VIRKe\n";
  fout << "DYNA INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals"
       << "    IMPRopers\n";
  fout << "DYNA CROSS:           CMAPs        PMF1D        PMF2D        PRIMO"
       << "             \n";
  fout << "DYNA EXTERN:        VDWaals         ELEC       HBONds          ASP"
       << "         USER\n";
  fout << "DYNA EWALD:          EWKSum       EWSElf       EWEXcl       EWQCor"
       << "       EWUTil\n";
  fout << "DYNA PRESS:            VIRE         VIRI       PRESSE       PRESSI"
       << "       VOLUme\n";
  fout << "DYNA XTLE:                       XTLTe         SURFtension  XTLPe "
       << "       XTLtemp\n";
  fout << " ----------       ---------    ---------    ---------    ---------"
       << "    ---------\n";

  m_HasWrittenHeader = true;

  return;
}
