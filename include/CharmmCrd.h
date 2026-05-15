// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "Coordinates.h"
#include <string>

/**
 * @brief Coordinate class generated from a CHARMM .crd or .cor file
 */
class CharmmCrd : public Coordinates {
public:
  /**
   * @brief Create a Coordinate object based on a CHARMM .crd or .cor file
   *
   * Initializes the coords vector with x,y and z coordinates as the first
   * three columns, and a 0.0 in the fourth column (charges are NOT set here).
   */
  CharmmCrd(const std::string &fileName);

private:
  /**
   * @brief Parses a .crd (or .cor) file into a 3N vector containing all
   * positions
   *
   * Also fills the 4th column with zeros.
   */
  void readCharmmCrdFile(const std::string &fileName);
};
