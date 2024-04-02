// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "Checkpoint.h"
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>

Checkpoint::Checkpoint(const std::string &fileName)
    : m_FileName(fileName), m_HeaderFields() {
  // Define the fields that are present in the header of the checkpoint file
  m_HeaderFields["numAtoms"] = 0;
  m_HeaderFields["coordsCharge"] = 0;
  m_HeaderFields["velocityMass"] = 0;
  m_HeaderFields["boxDimensions"] = 0;
  m_HeaderFields["coordsDelta"] = 0;
  m_HeaderFields["pistonDegreesOfFreedom"] = 0;
  m_HeaderFields["onStepPistonPosition"] = 0;
  m_HeaderFields["halfStepPistonPosition"] = 0;
  m_HeaderFields["onStepPistonVelocity"] = 0;
  m_HeaderFields["halfStepPistonVelocity"] = 0;
  m_HeaderFields["pistonNoseHooverPosition"] = 0;
  m_HeaderFields["pistonNoseHooverVelocity"] = 0;
  m_HeaderFields["pistonNoseHooverVelocityPrevious"] = 0;
  m_HeaderFields["pistonNoseHooverForce"] = 0;
  m_HeaderFields["pistonNoseHooverForcePrevious"] = 0;
}

template <typename T> T Checkpoint::get(const std::string &field) {
  std::ifstream fin(m_FileName, std::ios::binary);

  this->readHeader(fin);

  // Add check to make sure m_HeaderFields[field] > 0
  fin.seekg(m_HeaderFields[field], std::ios_base::beg);
  T val;

  fin.read(reinterpret_cast<char *>(&val), sizeof(T));
  fin.close();

  return val;
}

template int Checkpoint::get<int>(const std::string &field); // { return -1; }

void Checkpoint::writeCheckpoint(std::shared_ptr<CudaIntegrator> integrator) {
  std::ofstream fout;
  auto lp = std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(integrator);
  int pistonDegreesOfFreedom =
      (lp != nullptr) ? lp->getPistonDegreesOfFreedom() : -1;

  if (this->doesFileExist()) {
    // Append to existing checkpoint file
    // fout.open(m_FileName, std::ios::app | std::ios::binary);
    std::cout << "WARNING: Appending to checkpoint file is not yet supported, "
                 "overwriting existing file \""
              << m_FileName << "\"" << std::endl;
    fout.open(m_FileName, std::ios::binary);
  } else {
    // Create new checkpoint file
    fout.open(m_FileName, std::ios::binary);
    m_HeaderFields["numAtoms"] = 1;
    m_HeaderFields["coordsCharge"] = 1;
    m_HeaderFields["velocityMass"] = 1;
    m_HeaderFields["boxDimensions"] = 1;
    m_HeaderFields["coordaDelta"] = 1;
    if (lp != nullptr) {
      m_HeaderFields["pistonDegreesOfFreedom"] = 1;
      m_HeaderFields["onStepPistonPosition"] = 1;
      m_HeaderFields["halfStepPistonPosition"] = 1;
      m_HeaderFields["onStepPistonVelocity"] = 1;
      m_HeaderFields["halfStepPistonVelocity"] = 1;
      m_HeaderFields["pistonNoseHooverVelocity"] = 1;
      m_HeaderFields["pistonNoseHooverVelocityPrevious"] = 1;
      m_HeaderFields["pistonNoseHooverForce"] = 1;
      m_HeaderFields["pistonNoseHooverForcePrevious"] = 1;
    }
  }

  m_HeaderFields["numAtoms"] = 1;
  m_HeaderFields["coordsCharge"] = 1;
  m_HeaderFields["velocityMass"] = 1;
  m_HeaderFields["boxDimensions"] = 1;
  m_HeaderFields["coordsDelta"] = 1;
  if (lp != nullptr) {
    m_HeaderFields["pistonDegreesOfFreedom"] = 1;
    m_HeaderFields["onStepPistonPosition"] = 1;
    m_HeaderFields["halfStepPistonPosition"] = 1;
    m_HeaderFields["onStepPistonVelocity"] = 1;
    m_HeaderFields["halfStepPistonVelocity"] = 1;
    m_HeaderFields["pistonNoseHooverVelocity"] = 1;
    m_HeaderFields["pistonNoseHooverVelocityPrevious"] = 1;
    m_HeaderFields["pistonNoseHooverForce"] = 1;
    m_HeaderFields["pistonNoseHooverForcePrevious"] = 1;
  }

  std::shared_ptr<CharmmContext> ctx = integrator->getCharmmContext();

  int numAtoms = ctx->getNumAtoms();
  std::cout << "numAtoms = " << numAtoms << std::endl;

  this->writeHeader(fout, numAtoms, pistonDegreesOfFreedom);

  // Write numAtoms
  fout.write(reinterpret_cast<char *>(&numAtoms), sizeof(int));

  // Write coordsCharge
  CudaContainer<double4> coordsCharge = ctx->getCoordinatesCharges();
  coordsCharge.transferFromDevice();
  fout.write(reinterpret_cast<const char *>(coordsCharge.getHostArray().data()),
             4 * numAtoms * sizeof(double));

  // Write velocityMass
  CudaContainer<double4> velocityMass = ctx->getVelocityMass();
  velocityMass.transferFromDevice();
  fout.write(reinterpret_cast<const char *>(velocityMass.getHostArray().data()),
             4 * numAtoms * sizeof(double));

  // Write boxDimensions
  std::vector<double> boxDimensions = ctx->getBoxDimensions();
  fout.write(reinterpret_cast<const char *>(boxDimensions.data()),
             3 * sizeof(double));

  // Write coordsDelta
  CudaContainer<double4> coordsDelta = integrator->getCoordsDelta();
  coordsDelta.transferFromDevice();
  fout.write(reinterpret_cast<const char *>(coordsDelta.getHostArray().data()),
             4 * numAtoms * sizeof(double));

  if (lp != nullptr) {
    // Write pistonDegreesOfFreedom
    fout.write(reinterpret_cast<const char *>(&pistonDegreesOfFreedom),
               sizeof(int));

    // Write onStepPistonPosition
    CudaContainer<double> onStepPistonPosition = lp->getOnStepPistonPosition();
    onStepPistonPosition.transferFromDevice();
    fout.write(reinterpret_cast<const char *>(
                   onStepPistonPosition.getHostArray().data()),
               pistonDegreesOfFreedom * sizeof(double));

    // Write halfStepPistonPosition
    CudaContainer<double> halfStepPistonPosition =
        lp->getHalfStepPistonPosition();
    halfStepPistonPosition.transferFromDevice();
    fout.write(reinterpret_cast<const char *>(
                   halfStepPistonPosition.getHostArray().data()),
               pistonDegreesOfFreedom * sizeof(double));

    // Write onStepPistonVelocity
    CudaContainer<double> onStepPistonVelocity = lp->getOnStepPistonVelocity();
    onStepPistonVelocity.transferFromDevice();
    fout.write(reinterpret_cast<const char *>(
                   onStepPistonVelocity.getHostArray().data()),
               pistonDegreesOfFreedom * sizeof(double));

    // Write halfStepPistonVelocity
    CudaContainer<double> halfStepPistonVelocity =
        lp->getHalfStepPistonVelocity();
    halfStepPistonVelocity.transferFromDevice();
    fout.write(reinterpret_cast<const char *>(
                   halfStepPistonVelocity.getHostArray().data()),
               pistonDegreesOfFreedom * sizeof(double));

    // Write pistonNoseHooverPosition
    double pistonNoseHooverPosition = lp->getPistonNoseHooverPosition();
    fout.write(reinterpret_cast<const char *>(&pistonNoseHooverPosition),
               sizeof(double));

    // Write pistonNoseHooverVelocity
    double pistonNoseHooverVelocity = lp->getPistonNoseHooverVelocity();
    fout.write(reinterpret_cast<const char *>(&pistonNoseHooverVelocity),
               sizeof(double));

    // Write pistonNoseHooverVelocityPrevious
    double pistonNoseHooverVelocityPrevious =
        lp->getPistonNoseHooverVelocityPrevious();
    fout.write(
        reinterpret_cast<const char *>(&pistonNoseHooverVelocityPrevious),
        sizeof(double));

    // Write pistonNoseHooverForce
    double pistonNoseHooverForce = lp->getPistonNoseHooverForce();
    fout.write(reinterpret_cast<const char *>(&pistonNoseHooverForce),
               sizeof(double));

    // Write pistonNoseHooverForcePrevious
    double pistonNoseHooverForcePrevious =
        lp->getPistonNoseHooverForcePrevious();
    fout.write(reinterpret_cast<const char *>(&pistonNoseHooverForcePrevious),
               sizeof(double));
  }

  fout.close();

  // TESTING REMOVE LATER
  /////////////////////////////////////////////////////////////////////////
  // std::ifstream fin(m_FileName, std::ios::binary);
  // this->readHeader(fin);

  // numAtoms = -1;
  // fin.seekg(m_HeaderFields["numAtoms"], std::ios_base::beg);
  // fin.read(reinterpret_cast<char *>(&numAtoms), sizeof(int));
  // std::cout << "From Checkpoint file numAtoms = " << numAtoms << std::endl;

  // fin.close();
  /////////////////////////////////////////////////////////////////////////

  return;
}

void Checkpoint::writeHeader(std::ofstream &fout, const int numAtoms,
                             const int pistonDegreesOfFreedom) {
  const std::size_t NUM_FIELDS = m_HeaderFields.size();
  const std::size_t NUM_FIELDS_BYTES = NUM_FIELDS * sizeof(std::size_t);

  std::size_t pos = NUM_FIELDS_BYTES;
  std::size_t idx = 0;
  std::vector<std::size_t> buffer(NUM_FIELDS, 0);

  if (m_HeaderFields["numAtoms"] > 0) {
    buffer[idx] = pos; // Offset for numAtoms = size of the header
    m_HeaderFields["numAtoms"] = pos;
    pos += sizeof(int);
  }
  idx++;

  if (m_HeaderFields["coordsCharge"] > 0) {
    buffer[idx] = pos; // Offset for coordsCharge
    m_HeaderFields["coordsCharge"] = pos;
    pos += 4 * sizeof(double) * static_cast<std::size_t>(numAtoms);
  }
  idx++;

  if (m_HeaderFields["velocityMass"] > 0) {
    buffer[idx] = pos; // Offset for velocityMass
    m_HeaderFields["velocityMass"] = pos;
    pos += 4 * sizeof(double) * static_cast<std::size_t>(numAtoms);
  }
  idx++;

  if (m_HeaderFields["boxDimensions"] > 0) {
    buffer[idx] = pos; // Offset for boxDimensions
    m_HeaderFields["boxDimensions"] = pos;
    pos += 3 * sizeof(double);
  }
  idx++;

  if (m_HeaderFields["coordsDelta"] > 0) {
    buffer[idx] = pos; // Offset for coordsDelta
    m_HeaderFields["coordsDelta"] = pos;
    pos += 4 * sizeof(double) * static_cast<std::size_t>(numAtoms);
  }
  idx++;

  if (m_HeaderFields["pistonDegreesOfFreedom"] > 0) {
    buffer[idx] = pos; // Offset for pistonDegreesOfFreedom
    m_HeaderFields["pistonDegreesOfFreedom"] = pos;
    pos += sizeof(int);
  }
  idx++;

  if (m_HeaderFields["onStepPistonPosition"] > 0) {
    buffer[idx] = pos; // Offset for onStepPistonPosition
    m_HeaderFields["onStepPistonPosition"] = pos;
    pos += static_cast<std::size_t>(pistonDegreesOfFreedom) * sizeof(double);
  }
  idx++;

  if (m_HeaderFields["halfStepPistonPosition"] > 0) {
    buffer[idx] = pos; // Offset for halfStepPistonPosition
    m_HeaderFields["halfStepPistonPosition"] = pos;
    pos += static_cast<std::size_t>(pistonDegreesOfFreedom) * sizeof(double);
  }
  idx++;

  if (m_HeaderFields["onStepPistonVelocity"] > 0) {
    buffer[idx] = pos; // Offset for onStepPistonVelocity
    m_HeaderFields["onStepPistonVelocity"] = pos;
    pos += static_cast<std::size_t>(pistonDegreesOfFreedom) * sizeof(double);
  }
  idx++;

  if (m_HeaderFields["halfStepPistonVelocity"] > 0) {
    buffer[idx] = pos; // Offset for halfStepPistonVelocity
    m_HeaderFields["halfStepPistonVelocity"] = pos;
    pos += static_cast<std::size_t>(pistonDegreesOfFreedom) * sizeof(double);
  }
  idx++;

  if (m_HeaderFields["pistonNoseHooverPosition"] > 0) {
    buffer[idx] = pos; // Offset for pistonNoseHooverPosition
    m_HeaderFields["pistonNoseHooverPosition"] = pos;
    pos += sizeof(double);
  }
  idx++;

  if (m_HeaderFields["pistonNoseHooverVelocity"] > 0) {
    buffer[idx] = pos; // Offset for pistonNoseHooverVelocity
    m_HeaderFields["pistonNoseHooverVelocity"] = pos;
    pos += sizeof(double);
  }
  idx++;

  if (m_HeaderFields["pistonNoseHooverVelocityPrevious"] > 0) {
    buffer[idx] = pos; // Offset for pistonNoseHooverVelocityPrevious
    m_HeaderFields["pistonNoseHooverVelocityPrevious"] = pos;
    pos += sizeof(double);
  }
  idx++;

  if (m_HeaderFields["pistonNoseHooverForce"] > 0) {
    buffer[idx] = pos; // Offset for pistonNoseHooverForce
    m_HeaderFields["pistonNoseHooverForce"] = pos;
    pos += sizeof(double);
  }
  idx++;

  if (m_HeaderFields["pistonNoseHooverForcePrevious"] > 0) {
    buffer[idx] = pos; // Offset for pistonNoseHooverForcePrevious
    m_HeaderFields["pistonNoseHooverForcePrevious"] = pos;
    pos += sizeof(double);
  }
  idx++;

  fout.write(reinterpret_cast<char *>(buffer.data()), NUM_FIELDS_BYTES);

  return;
}

void Checkpoint::readHeader(std::ifstream &fin) {
  const std::size_t NUM_FIELDS = m_HeaderFields.size();
  const std::size_t NUM_FIELDS_BYTES = NUM_FIELDS * sizeof(std::size_t);

  std::vector<std::size_t> buffer(NUM_FIELDS, 0);

  fin.read(reinterpret_cast<char *>(buffer.data()), NUM_FIELDS_BYTES);

  std::size_t idx = 0;
  m_HeaderFields["numAtoms"] = buffer[idx++];
  m_HeaderFields["coordsCharge"] = buffer[idx++];
  m_HeaderFields["velocityMass"] = buffer[idx++];
  m_HeaderFields["boxDimensions"] = buffer[idx++];
  m_HeaderFields["coordsDelta"] = buffer[idx++];

  m_HeaderFields["pistonDegreesOfFreedom"] = buffer[idx++];
  m_HeaderFields["onStepPistonPosition"] = buffer[idx++];
  m_HeaderFields["halfStepPistonPosition"] = buffer[idx++];
  m_HeaderFields["onStepPistonVelocity"] = buffer[idx++];
  m_HeaderFields["halfStepPistonVelocity"] = buffer[idx++];
  m_HeaderFields["pistonNoseHooverPosition"] = buffer[idx++];
  m_HeaderFields["pistonNoseHooverVelocity"] = buffer[idx++];
  m_HeaderFields["pistonNoseHooverVelocityPrevious"] = buffer[idx++];
  m_HeaderFields["pistonNoseHooverForce"] = buffer[idx++];
  m_HeaderFields["pistonNoseHooverForcePrevious"] = buffer[idx++];

  return;
}

bool Checkpoint::doesFileExist(void) const {
  std::ifstream fin(m_FileName);
  if (!fin.is_open())
    return false;
  fin.close();
  return true;
}
