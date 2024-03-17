// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include <iostream>
#include <memory>
#include "CharmmContext.h"
#include "ForceManager.h"
#include "CudaExternalForce.h"
#include "CudaRestraintForce.h"
#include "CudaBondedForce.h"
#include "catch.hpp"

//class TestForce{};

TEST_CASE("add_foce", "[force]") {
/*
    ForceManager fm;

    CudaEnergyVirial energyVirial;
    CudaBondedForce<long long int, float> bonded(energyVirial, "bond", "ureyb", "angle", "dihe", "imdihe", "cmap");
    fm.emplace_back(bonded);

    CudaExternalForce<long long int, float> external;
    fm.emplace_back(external);

    CudaRestraintForce<long long int, float> restraint;
    fm.emplace_back(restraint);
    
    CudaEnergyVirial directEnergyVirial;
    CudaPMEDirectForce<long long int, float> direct(directEnergyVirial, "vdw","elec", "ewex");
    //fm.emplace_back(direct);

    fm.calc_force();
*/
}

