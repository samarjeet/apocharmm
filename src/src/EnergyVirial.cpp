// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#ifndef NOCUDAC
#include <iostream>
#include <cstdlib>
#include "EnergyVirial.h"

//
// Class creator
//
EnergyVirial::EnergyVirial() { n = 0; }

//
// Insert new energy term.
// Note: term is only added if it doesn't already exist
//
void EnergyVirial::insert(std::string& name) {
    if (energyIndex.empty() || energyIndex.count(name) == 0) {
        energyIndex.insert(std::pair<std::string, int>(name, n));
        n++;
    }
}

void EnergyVirial::insert(const char* name) {
    std::string str(name);
    this->insert(str);
}

//
// Returns index of energy term
//
int EnergyVirial::getEnergyIndex(std::string& name) {
    std::map<std::string, int>::iterator it = energyIndex.find(name);
    if (it == energyIndex.end()) {
        //std::cerr << "EnergyVirial::getEnergyIndex, energy term " << name << " not found" << std::endl;
        throw std::invalid_argument("EnergyVirial::getEnergyIndex, energy term " + name + " not found\n" );
        exit(1);
    }
    return it->second;
}
#endif  // NOCUDAC
