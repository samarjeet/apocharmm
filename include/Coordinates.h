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
  Coordinates(void);

  /**
   * @brief Create a Coordinate object using a list of doubles.
   *
   * Initializes the coords vector with x,y and z coordinates as the first
   * three columns, and a 0.0 in the fourth column (charges are NOT set here).
   */
  Coordinates(const std::vector<double3> &coords);

  /**
   * @brief Create a Coordinate object using a list of floats.
   *
   * Initializes the coords vector with x,y and z coordinates as the first
   * three columns, and a 0.0 in the fourth column (charges are NOT set here).
   */
  Coordinates(const std::vector<float3> &coords);

  /**
   * @brief Creates a Coordinate object using a vector of vector (dimensions
   * N*3)
   */
  Coordinates(const std::vector<std::vector<double>> &coords);

  /**
   * @brief Creates a Coordinate object using a vector of vector (dimensions
   * N*3)
   */
  Coordinates(const std::vector<std::vector<float>> &coords);

public:
  void setNumAtoms(const int numAtoms);

public:
  /**
   * @brief Returns the size of the coords vector
   * @return int
   */
  int getNumAtoms(void) const;

  const std::vector<double4> &getCoordinatesD(void) const;
  std::vector<double4> &getCoordinatesD(void);

  const std::vector<float4> &getCoordinatesF(void) const;
  std::vector<float4> &getCoordinatesF(void);

protected:
  int m_NumAtoms;
  std::vector<double4> m_CoordinatesD;
  std::vector<float4> m_CoordinatesF;
};
