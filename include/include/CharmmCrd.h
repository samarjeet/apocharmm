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

  /**
   * @brief Create a Coordinate object using a list of floats.
   *
   * Initializes the coords vector with x,y and z coordinates as the first
   * three columns, and a 0.0 in the fourth column (charges are NOT set here).
   */
  CharmmCrd(const std::vector<float3> _inpCoords);

  /** @brief Creates a Coordinate object using a vector of vector (dimensions
   * N*3) */
  CharmmCrd(const std::vector<std::vector<double>> _inpCoords);

private:
  // int numAtoms;
  //  // std::vector<double3> coords;
  // std::vector<float4> coords;
  /**
   * @brief Parses a .crd (or .cor) file into a 3N vector containing all
   * positions
   *
   * Also fills the 4th column with zeros.
   */
  void readCharmmCrdFile(std::string fileName);
};
