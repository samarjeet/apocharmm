// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include "Coordinates.h"

class PDB : public Coordinates {
public:
  PDB(const std::string &fileName);

private:
  void readPDBFile(std::string fileName);
};
