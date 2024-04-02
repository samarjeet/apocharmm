// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "Coordinates.h"

std::vector<std::vector<double>> Coordinates::getCoordinates(void) const {
  std::vector<std::vector<double>> doubleCoords(coords.size());

  for (std::size_t i = 0; i < coords.size(); i++)
    doubleCoords[i] = {coords[i].x, coords[i].y, coords[i].z, coords[i].w};

  return doubleCoords;
}

int Coordinates::getNumAtoms(void) const { return coords.size(); }
