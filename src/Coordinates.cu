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

Coordinates::Coordinates(void)
    : m_NumAtoms(-1), m_CoordinatesD(), m_CoordinatesF() {}

Coordinates::Coordinates(const std::vector<double3> &coords) : Coordinates() {
  m_NumAtoms = static_cast<int>(coords.size());
  m_CoordinatesD.resize(m_NumAtoms);
  m_CoordinatesF.resize(m_NumAtoms);

  for (int i = 0; i < m_NumAtoms; i++) {
    m_CoordinatesD[i] =
        make_double4(coords[i].x, coords[i].y, coords[i].z, 0.0);
    m_CoordinatesF[i] = make_float4(static_cast<float>(coords[i].x),
                                    static_cast<float>(coords[i].y),
                                    static_cast<float>(coords[i].z), 0.0);
  }
}

Coordinates::Coordinates(const std::vector<float3> &coords) : Coordinates() {
  m_NumAtoms = static_cast<int>(coords.size());
  m_CoordinatesD.resize(m_NumAtoms);
  m_CoordinatesF.resize(m_NumAtoms);

  for (int i = 0; i < m_NumAtoms; i++) {
    m_CoordinatesD[i] = make_double4(static_cast<double>(coords[i].x),
                                     static_cast<double>(coords[i].y),
                                     static_cast<double>(coords[i].z), 0.0);
    m_CoordinatesF[i] = make_float4(coords[i].x, coords[i].y, coords[i].z, 0.0);
  }
}

Coordinates::Coordinates(const std::vector<std::vector<double>> &coords)
    : Coordinates() {
  m_NumAtoms = static_cast<int>(coords.size());
  m_CoordinatesD.resize(m_NumAtoms);
  m_CoordinatesF.resize(m_NumAtoms);

  for (int i = 0; i < m_NumAtoms; i++) {
    m_CoordinatesD[i] =
        make_double4(coords[i][0], coords[i][1], coords[i][2], 0.0);
    m_CoordinatesF[i] = make_float4(static_cast<float>(coords[i][0]),
                                    static_cast<float>(coords[i][1]),
                                    static_cast<float>(coords[i][2]), 0.0);
  }
}

Coordinates::Coordinates(const std::vector<std::vector<float>> &coords)
    : Coordinates() {
  m_NumAtoms = static_cast<int>(coords.size());
  m_CoordinatesD.resize(m_NumAtoms);
  m_CoordinatesF.resize(m_NumAtoms);

  for (int i = 0; i < m_NumAtoms; i++) {
    m_CoordinatesD[i] = make_double4(static_cast<double>(coords[i][0]),
                                     static_cast<double>(coords[i][1]),
                                     static_cast<double>(coords[i][2]), 0.0);
    m_CoordinatesF[i] =
        make_float4(coords[i][0], coords[i][1], coords[i][2], 0.0);
  }
}

void Coordinates::setNumAtoms(const int numAtoms) {
  m_NumAtoms = numAtoms;
  m_CoordinatesD.resize(numAtoms);
  m_CoordinatesF.resize(numAtoms);
  return;
}

const std::vector<double4> &Coordinates::getCoordinatesD(void) const {
  return m_CoordinatesD;
}

std::vector<double4> &Coordinates::getCoordinatesD(void) {
  return m_CoordinatesD;
}

const std::vector<float4> &Coordinates::getCoordinatesF(void) const {
  return m_CoordinatesF;
}

std::vector<float4> &Coordinates::getCoordinatesF(void) {
  return m_CoordinatesF;
}

int Coordinates::getNumAtoms(void) const { return m_NumAtoms; }
