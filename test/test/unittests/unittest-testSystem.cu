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
#include "TestSystem.h"
#include "catch.hpp"

TEST_CASE("testSystem", "[preparation]") {

    auto harmOsc = HarmonicOscillator("harmonicOscillator");
    auto harmOscContext = harmOsc.getCharmmContext();
    //auto harmOsc2 = HarmonicOscillator();
}
