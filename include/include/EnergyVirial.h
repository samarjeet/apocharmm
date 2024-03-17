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
#ifndef ENERGYVIRIAL_H
#define ENERGYVIRIAL_H
//
// Storage class for energy names
// (c) Antti-Pekka Hynninen, Feb 2015
// aphynninen@hotmail.com
//
#include <map>
#include <string>

/**
 * @brief Stores the energy and virial for a force.
 *
 */
class EnergyVirial {
private:
  /** @brief  Number of energy terms */
  int n;

  /**
   * @brief Energy term index map (name->index)
   */
  std::map<std::string, int> energyIndex;

protected:
  /** @brief Basic constructor */
  EnergyVirial();
  ~EnergyVirial() {}
  /** @brief  Returns index of energy term */
  int getEnergyIndex(std::string &name);

public:
  /** @brief Add energy term to the EnergyVirial object
   */
  void insert(std::string &name);
  void insert(const char *name);
  /** @brief Returns number of energy terms */
  int getN() const { return n; }
};

#endif // ENERGYVIRIAL_H
#endif // NOCUDAC
