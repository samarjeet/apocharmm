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

#include <cuda.h>
#include <string>
#include <vector>

/**
 * @brief Coordinates base class
 *
 * Should only be used as an input interface to setup a simulation.
 * Runtime operations are done on CudaContainer objects.
 *
 */
class Coordinates {
public:
  // Coordinates(const std::string &fileName);

  /** @brief Returns coordinates as a 4*N sized vector
   * @return vector<float4>
   */
  std::vector<std::vector<double>> getCoordinates(void) const;

  /** @brief Returns the size of the coords vector
   * @return int
   */
  int getNumAtoms(void) const;

protected:
  int numAtoms;
  // std::vector<double3> coords;
  std::vector<float4> coords;
  // void readCharmmCrdFile(std::string fileName);
};
