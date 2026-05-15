// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#ifndef NOCUDAC

#include "EnergyVirial.h"

#include <cstdlib>
#include <iostream>

//
// Insert new energy term.
// Note: term is only added if it doesn't already exist
//
void EnergyVirial::insert(const std::string &name) {
  if ((m_EnergyIndex.empty()) || (m_EnergyIndex.count(name) == 0)) {
    m_EnergyIndex.insert(std::pair<std::string, int>(name, m_NumEnergyTerms));
    m_NumEnergyTerms++;
  }
  return;
}

void EnergyVirial::insert(const char *name) {
  this->insert(std::string(name));
  return;
}

int EnergyVirial::getN(void) const { return m_NumEnergyTerms; }

//
// Class creator
//
EnergyVirial::EnergyVirial(void) : m_NumEnergyTerms(0), m_EnergyIndex() {}

//
// Class destructor
//
EnergyVirial::~EnergyVirial(void) {}

//
// Returns index of energy term
//
int EnergyVirial::getEnergyIndex(const std::string &name) {
  std::map<std::string, int>::iterator it = m_EnergyIndex.find(name);
  if (it != m_EnergyIndex.end())
    return it->second;
  else {
    std::cout << "Could not find \"" << name << "\" energy term" << std::endl;
    return -1;
  }
}

#endif // NOCUDAC
