// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaNoseHooverThermostatIntegrator.h"
#include "PBC.h"
#include "RestartSubscriber.h"
#include "str_utils.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

RestartSubscriber::RestartSubscriber(void) : Subscriber() {}

RestartSubscriber::RestartSubscriber(const std::string &fileName)
    : Subscriber(fileName) {}

RestartSubscriber::RestartSubscriber(const std::string &fileName,
                                     const int reportFrequency)
    : Subscriber(fileName, reportFrequency) {}

RestartSubscriber::~RestartSubscriber(void) {
  if (m_FileStream.is_open())
    m_FileStream.close();
}

void RestartSubscriber::update(void) {
  constexpr int rstDoubleWidth = 22;
  constexpr int rstDoublePrecision = 15;
  constexpr int VERSION = 50;
  constexpr int LENENP = 60;
  constexpr int LENENT = 128;
  constexpr int LENENV = 50;

  if (m_FileStream.is_open())
    m_FileStream.close();
  m_FileStream.open(m_FileName, std::ios::out);

  // Attempt to cast to supported integrators to determine how we set some
  // values
  auto nh = std::dynamic_pointer_cast<CudaNoseHooverThermostatIntegrator>(
      m_Integrator);
  auto lp =
      std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(m_Integrator);
  auto lt =
      std::dynamic_pointer_cast<CudaLangevinThermostatIntegrator>(m_Integrator);

  // Ensure that the integrator is at least one of the supported types
  if ((nh == nullptr) && (lp == nullptr) && (lt == nullptr)) {
    std::string msg =
        "Attempted to write a restart file for an unsupported integrator type. "
        "Currently the only supported integrators are:\n";
    msg += "  1.) CudaNoseHooverThermostatIntegrator\n";
    msg += "  2.) CudaLangevinPistonIntegrator\n";
    msg += "  3.) CudaLangevinThermostatIntegrator";
    throw std::runtime_error(msg);
  }

  std::string crystalString = "NONE";
  const std::vector<double> boxDimensions = m_CharmmContext->getBoxDimensions();

  if ((nh != nullptr) || (lt != nullptr)) {
    // JEG260330: Get box dimensions and compare lengths to determine crystal
    // type. This should be fixed so that the CharmmContext always stores the
    // crystal type.
    if ((boxDimensions[0] == boxDimensions[1]) &&
        (boxDimensions[0] == boxDimensions[2]))
      crystalString = "CUBI";
    else if (boxDimensions[0] == boxDimensions[1])
      crystalString = "TETR";
    else
      crystalString = "ORTH";
  } else if (lp != nullptr) {
    if (lp->getCrystalType() == CRYSTAL::CUBIC)
      crystalString = "CUBI";
    else if (lp->getCrystalType() == CRYSTAL::TETRAGONAL)
      crystalString = "TETR";
    else if (lp->getCrystalType() == CRYSTAL::ORTHORHOMBIC)
      crystalString = "ORTH";
  }

  // Write formatted header
  m_FileStream << "REST" << std::setw(6) << VERSION << std::setw(6) << 1 << "  "
               << std::setw(4) << crystalString << "  "
               << "    "
               << "  "
               << "APO" << '\n';
  m_FileStream << '\n';

  // Write TITLE section
  // JEG260330: Dummy title for now. We should make this something.
  m_FileStream << std::setw(8) << 2 << " !NTITLE followed by title\n";
  m_FileStream << "* APOCHARMM RESTART FILE                                    "
                  "                    \n";
  m_FileStream << "* USED FOR CONTINUING MOLECULAR DYNAMICS TRAJECTORY         "
                  "                    \n";
  m_FileStream << '\n';

  // Write CRYSTAL PARAMETERS section
  std::vector<double> HDOT(6, 0.0);
  double PNH = 0.0, PNHV = 0.0, PNHF = 0.0;
  std::vector<double> UC1A(6, 0.0), UC2A(6, 0.0), UC1B(6, 0.0), UC2B(6, 0.0);
  constexpr double GRAD1A = 0.0;
  constexpr double GRAD1B = 0.0;
  constexpr double GRAD2A = 0.0;
  constexpr double GRAD2B = 0.0;

  if (nh != nullptr) {
    nh->getNoseHooverPistonVelocity().transferToHost();
    nh->getNoseHooverPistonForce().transferToHost();
    PNHV = nh->getNoseHooverPistonVelocity()[0];
    PNHF = nh->getNoseHooverPistonForce()[0];
  } else if (lp != nullptr) {
    lp->getNoseHooverPistonVelocity().transferToHost();
    lp->getNoseHooverPistonForce().transferToHost();
    PNHV = lp->getNoseHooverPistonVelocity()[0];
    PNHF = lp->getNoseHooverPistonForce()[0];
    lp->getLangevinPistonDeltaPosition().transferToHost();
    if (crystalString == "CUBI") {
      HDOT[0] = lp->getLangevinPistonDeltaPosition()[0];
    } else if (crystalString == "TETR") {
      HDOT[0] = lp->getLangevinPistonDeltaPosition()[0];
      HDOT[1] = lp->getLangevinPistonDeltaPosition()[1];
    } else if (crystalString == "ORTH") {
      HDOT[0] = lp->getLangevinPistonDeltaPosition()[0];
      HDOT[1] = lp->getLangevinPistonDeltaPosition()[1];
      HDOT[2] = lp->getLangevinPistonDeltaPosition()[2];
    }
  }

  m_FileStream << " !CRYSTAL PARAMETERS\n";
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(boxDimensions[0], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(0.0, rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(boxDimensions[1], rstDoublePrecision)
               << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(0.0, rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(0.0, rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(boxDimensions[2], rstDoublePrecision)
               << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(HDOT[0], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(HDOT[1], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(HDOT[2], rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(HDOT[3], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(HDOT[4], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(HDOT[5], rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(PNH, rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(PNHV, rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(PNHF, rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1A[0], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1A[1], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1A[2], rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1A[3], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1A[4], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1A[5], rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2A[0], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2A[1], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2A[2], rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2A[3], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2A[4], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2A[5], rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1B[0], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1B[1], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1B[2], rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1B[3], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1B[4], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC1B[5], rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2B[0], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2B[1], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2B[2], rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2B[3], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2B[4], rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(UC2B[5], rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(GRAD1A, rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(GRAD1B, rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(GRAD2A, rstDoublePrecision) << '\n';
  m_FileStream << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(GRAD2B, rstDoublePrecision) << '\n';
  m_FileStream << '\n';

  // Write integer section
  const int NATOM = m_CharmmContext->getNumAtoms();
  const unsigned long long int NPRIV = m_Integrator->getTotNumSteps();
  const int NSTEP = m_Integrator->getNumSteps();
  // JEG260330: This is technically wrong and should be the saving frequency for
  // coordinates being written to the DCD file.
  const int NSAVC = m_ReportFrequency;
  // JEG260330: This is technically wrong and should be the saving frequency for
  // velocities being written to the DCD file.
  const int NSAVV = 0;
  const int JHSTRT = 0;
  const int NDEGF = m_CharmmContext->getDegreesOfFreedom();
  std::uint64_t SEED = 0;
  std::string RNGSTATE = "";
  if (lp != nullptr) {
    SEED = lp->getLangevinPistonFrictionSeed();
    RNGSTATE = std::to_string(lp->getRngSequencePos());
  } else if (lt != nullptr) {
    SEED = lt->getThermostatRngSeed();
    RNGSTATE = std::to_string(lt->getRngSequencePos());
  }

  m_FileStream << " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n";
  m_FileStream << std::setw(12) << NATOM << std::setw(12) << NPRIV
               << std::setw(12) << NSTEP << std::setw(12) << NSAVC
               << std::setw(12) << NSAVV << std::setw(12) << JHSTRT
               << std::setw(12) << NDEGF << std::setw(22) << SEED << "  "
               << RNGSTATE << '\n';
  m_FileStream << '\n';

  // Write ENERGIES and STATISTICS section
  // JEG260330: As of now all of these are being set to 0. We don't calculate A
  // LOT of the energy terms that CHARMM does. They are set to 0 instead of
  // -999.999 (or something similar) to avoid causing any issues if they are
  // read and used by CHARMM at some point.
  const std::string QEPROP(LENENP, 'T');
  const std::string QETERM(LENENT, 'T');
  constexpr int ISTPSA = 0;
  constexpr double FITA = 0.0;
  constexpr double FITP = 0.0;
  double AVETEM = 0.0;
  std::vector<double> EPROP(LENENP, 0.0), EPRPP(LENENP, 0.0),
      EPRP2P(LENENP, 0.0);
  std::vector<double> EPRPA(LENENP, 0.0), EPRP2A(LENENP, 0.0);
  std::vector<double> ETERM(LENENT, 0.0), ETRMP(LENENT, 0.0),
      ETRM2P(LENENT, 0.0);
  std::vector<double> ETRMA(LENENT, 0.0), ETRM2A(LENENT, 0.0);
  std::vector<double> EPRESS(LENENV, 0.0), EPRSP(LENENV, 0.0),
      EPRS2P(LENENV, 0.0);
  std::vector<double> EPRSA(LENENV, 0.0), EPRS2A(LENENV, 0.0);

  if (nh != nullptr) {
    nh->getAverageTemperature().transferToHost();
    AVETEM = nh->getAverageTemperature()[0];
  } else if (lp != nullptr) {
    lp->getAverageTemperature().transferToHost();
    AVETEM = lp->getAverageTemperature()[0];
  } else if (lt != nullptr) {
    lt->getAverageTemperature().transferToHost();
    AVETEM = lt->getAverageTemperature()[0];
  }

  m_FileStream << " !ENERGIES and STATISTICS\n";
  m_FileStream << QEPROP << '\n';
  m_FileStream << QETERM << '\n';
  m_FileStream << std::setw(8) << ISTPSA << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(FITA, rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(FITP, rstDoublePrecision)
               << std::setw(rstDoubleWidth)
               << apo::cDoubleToFortSciStr(AVETEM, rstDoublePrecision) << '\n';
  for (int i = 0; i < LENENP; i++) {
    m_FileStream << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(EPROP[i], rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(EPRP2P[i], rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(EPRPP[i], rstDoublePrecision)
                 << '\n';
  }
  for (int i = 0; i < LENENP; i++) {
    m_FileStream << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(EPRPA[i], rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(EPRP2A[i], rstDoublePrecision)
                 << '\n';
  }
  for (int i = 0; i < LENENT; i++) {
    m_FileStream << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(ETERM[i], rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(ETRMP[i], rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(ETRM2P[i], rstDoublePrecision)
                 << '\n';
  }
  for (int i = 0; i < LENENT; i++) {
    m_FileStream << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(ETRMA[i], rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(ETRM2A[i], rstDoublePrecision)
                 << '\n';
  }
  for (int i = 0; i < LENENV; i++) {
    m_FileStream << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(EPRESS[i], rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(EPRSP[i], rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(EPRS2P[i], rstDoublePrecision)
                 << '\n';
  }
  for (int i = 0; i < LENENV; i++) {
    m_FileStream << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(EPRSA[i], rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(EPRS2A[i], rstDoublePrecision)
                 << '\n';
  }
  m_FileStream << '\n';

  // Write XOLD, YOLD, ZOLD section
  m_FileStream << " !XOLD, YOLD, ZOLD\n";
  m_Integrator->getCoordsDeltaPrevious().transferToHost();
  for (int i = 0; i < NATOM; i++) {
    m_FileStream
        << std::setw(rstDoubleWidth)
        << apo::cDoubleToFortSciStr(m_Integrator->getCoordsDeltaPrevious()[i].x,
                                    rstDoublePrecision)
        << std::setw(rstDoubleWidth)
        << apo::cDoubleToFortSciStr(m_Integrator->getCoordsDeltaPrevious()[i].y,
                                    rstDoublePrecision)
        << std::setw(rstDoubleWidth)
        << apo::cDoubleToFortSciStr(m_Integrator->getCoordsDeltaPrevious()[i].z,
                                    rstDoublePrecision)
        << '\n';
  }
  m_FileStream << '\n';

  // Write VX, VY, VZ section
  m_FileStream << " !VX, VY, VZ\n";
  m_CharmmContext->getVelocityMass().transferToHost();
  for (int i = 0; i < NATOM; i++) {
    m_FileStream
        << std::setw(rstDoubleWidth)
        << apo::cDoubleToFortSciStr(m_CharmmContext->getVelocityMass()[i].x,
                                    rstDoublePrecision)
        << std::setw(rstDoubleWidth)
        << apo::cDoubleToFortSciStr(m_CharmmContext->getVelocityMass()[i].y,
                                    rstDoublePrecision)
        << std::setw(rstDoubleWidth)
        << apo::cDoubleToFortSciStr(m_CharmmContext->getVelocityMass()[i].z,
                                    rstDoublePrecision)
        << '\n';
  }
  m_FileStream << '\n';

  // Write X, Y, Z section
  m_FileStream << " !X, Y, Z\n";
  m_CharmmContext->getCoordinatesCharges().transferToHost();
  for (int i = 0; i < NATOM; i++) {
    m_FileStream << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(
                        m_CharmmContext->getCoordinatesCharges()[i].x,
                        rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(
                        m_CharmmContext->getCoordinatesCharges()[i].y,
                        rstDoublePrecision)
                 << std::setw(rstDoubleWidth)
                 << apo::cDoubleToFortSciStr(
                        m_CharmmContext->getCoordinatesCharges()[i].z,
                        rstDoublePrecision)
                 << '\n';
  }

  m_FileStream.close();

  return;
}
