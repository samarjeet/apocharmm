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
#include <filesystem>
#include <fstream>

Checkpoint::Checkpoint(const std::string &fileName)
    : m_FileName(fileName), m_HeaderFields() {
  // Define the fields that are present in the header of the checkpoint file
  m_HeaderFields["numAtoms"] = 0;
  m_HeaderFields["coordsCharge"] = 1;
  m_HeaderFields["velocityMass"] = 2;
  m_HeaderFields["boxDimensions"] = 3;
  m_HeaderFields["coordaDelta"] = 4;
  m_HeaderFields["pistonDegreesOfFreedom"] = 5;
  m_HeaderFields["onStepPistonPosition"] = 6;
  m_HeaderFields["halfStepPistonPosition"] = 7;
  m_HeaderFields["onStepPistonVelocity"] = 8;
  m_HeaderFields["noseHooverPistonPosition"] = 9;
  m_HeaderFields["noseHooverPistonVelocity"] = 10;
  m_HeaderFields["noseHooverPistonForce"] = 11;
}

void Checkpoint::writeCheckpoint(std::shared_ptr<CharmmContext> ctx) {
  std::ofstream fout;
  int pistonDegreesOfFreedom = 0;

  if (this->doesFileExist()) {
    // Append to existing checkpoint file
    // fout.open(m_FileName, std::ios::app | std::ios::binary);
    // Check if pistonDegreesOfFreedom is in existing checkpoint file
    std::cout << "WARNING: Appending to checkpoint file is not yet supported, "
                 "overwriting existing file \""
              << m_FileName << "\"" << std::endl;
    fout.open(m_FileName, std::ios::binary);
  } else {
    // Create new checkpoint file
    fout.open(m_FileName, std::ios::binary);
  }

  int numAtoms = ctx->getNumAtoms();

  this->writeHeader(fout, numAtoms, pistonDegreesOfFreedom);

  // Write numAtoms
  std::cout << "numAtoms = " << numAtoms << std::endl;
  fout.write(reinterpret_cast<char *>(&numAtoms), sizeof(int));

  // Write coordsCharge
  auto coordsCharge = ctx->getCoordinatesCharges();
  coordsCharge.transferFromDevice();
  fout.write(reinterpret_cast<const char *>(coordsCharge.getHostArray().data()),
             4 * numAtoms * sizeof(double));

  // Write velocityMass
  auto velocityMass = ctx->getVelocityMass();
  velocityMass.transferFromDevice();
  fout.write(reinterpret_cast<const char *>(velocityMass.getHostArray().data()),
             4 * numAtoms * sizeof(double));

  // Write boxDimensions
  auto boxDimensions = ctx->getBoxDimensions();
  // boxDimensions.transferFromDevice();
  fout.write(reinterpret_cast<const char *>(boxDimensions.data()),
             3 * sizeof(double));

  fout.close();

  std::ifstream fin(m_FileName, std::ios::binary);
  this->readHeader(fin);

  return;
}

void Checkpoint::writeCheckpoint(std::shared_ptr<CudaIntegrator> integrator) {
  return;
}

void Checkpoint::writeHeader(std::ofstream &fout, const int numAtoms,
                             const int pistonDegreesOfFreedom) {
  const int NUM_FIELDS = static_cast<int>(m_HeaderFields.size());
  const int NUM_FIELDS_BYTES = NUM_FIELDS * sizeof(int);

  std::vector<int> buffer(NUM_FIELDS, -1);
  buffer[0] = NUM_FIELDS_BYTES; // Offset for numAtoms = size of the header
  buffer[1] = buffer[0] + sizeof(int); // Offset for coordsCharge
  buffer[2] =
      buffer[1] + 4 * sizeof(double) * numAtoms; // Offset for velocityMass
  buffer[3] =
      buffer[2] + 4 * sizeof(double) * numAtoms; // Offset for boxDimensions
  buffer[4] = buffer[3] + 3 * sizeof(double);    // Offset for coordsDelta
  buffer[5] = buffer[4] + 4 * sizeof(double) *
                              numAtoms; // Offset for pistonDegreesOfFreedom
  buffer[6] = buffer[5] + sizeof(int);  // Offset for onStepPistonPostiion
  buffer[7] =
      buffer[6] + pistonDegreesOfFreedom *
                      sizeof(double); // Offset for halfStepPistonPosition
  buffer[8] = buffer[7] + pistonDegreesOfFreedom *
                              sizeof(double); // Offset for onStepPistonVelocity
  buffer[9] =
      buffer[8] + pistonDegreesOfFreedom *
                      sizeof(double); // Offset for noseHooverPistonPosition
  buffer[10] =
      buffer[9] + sizeof(double); // Offset for noseHooverPistonVelocity
  buffer[11] = buffer[10] + sizeof(double); // Offset for noseHooverPistonForce

  fout.write(reinterpret_cast<char *>(buffer.data()), NUM_FIELDS_BYTES);

  return;
}

void Checkpoint::readHeader(std::ifstream &fin) {
  int tmp = -1;

  fin.read(reinterpret_cast<char *>(&tmp), sizeof(int));
  std::cout << "tmp = " << tmp << std::endl;

  return;
}

bool Checkpoint::doesFileExist(void) const {
  std::ifstream fin(m_FileName);
  if (!fin.is_open())
    return false;
  fin.close();
  return true;
}