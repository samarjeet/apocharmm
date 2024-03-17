// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "CharmmContext.h"
#include "CudaIntegrator.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include <map>
#include <string>
#include <vector>

class Checkpoint {
public:
  Checkpoint(const std::string &fileName);

  void writeCheckpoint(std::shared_ptr<CharmmContext> ctx);
  void writeCheckpoint(std::shared_ptr<CudaIntegrator> integrator);

private:
  void writeHeader(std::ofstream &fout, const int numAtoms,
                   const int pistonDegreesOfFreedom);
  void readHeader(std::ifstream &fin);
  bool doesFileExist(void) const;

private:
  std::string m_FileName;
  std::map<std::string, int> m_HeaderFields;
};