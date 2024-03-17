// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "PDB.h"
#include "catch.hpp"
#include "helper.h"
#include "test_paths.h"
#include <iostream>
#include <memory>

// Tests :
// - that CharmmCrd reads coordinates correctly from .crd and .pdb files
// - that you can give a vector of float3 as input to create a CharmmCrd
TEST_CASE("readCrd", "[setup]") {
  auto path = getDataPath();
  SECTION("dhfr") {
    auto crd = CharmmCrd(path + "dhfr.crd");
    REQUIRE(22498 == crd.getNumAtoms());
    compareCoordsFromFile(crd.getCoordinates(),
                          path + "coordsrefxyz/dhfr_xyz.txt");
  }

  SECTION("ecor1 drude") {
    CharmmCrd crd(path + "ecor1_drude.crd");
    REQUIRE(25442 == crd.getNumAtoms());
    compareCoordsFromFile(crd.getCoordinates(),
                          path + "coordsrefxyz/ecor1_xyz.txt");
  }

  SECTION("2cba") {
    CharmmCrd crd(path + "2cba.crd");
    REQUIRE(7177 == crd.getNumAtoms());
    compareCoordsFromFile(crd.getCoordinates(),
                          path + "coordsrefxyz/2cba_xyz.txt");
  }

  SECTION("pdb") {
    PDB pdb(path + "dyn500.pdb");
    REQUIRE(267322 == pdb.getNumAtoms());
    compareCoordsFromFile(pdb.getCoordinates(),
                          path + "coordsrefxyz/dyn500_pdb.txt");
  }
}

TEST_CASE("CrdVector") {
  // test that you can give a vector of float3 as input to  create a CharmmCrd
  std::string dataPath = getDataPath();
  auto prm = std::make_shared<CharmmParameters>(dataPath + "argon.prm");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "argon_10.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions({50.0, 50.0, 50.0});
  fm->setFFTGrid(48, 48, 48);
  fm->setKappa(0.34);
  fm->setCutoff(10.0);
  fm->setCtonnb(7.0);
  fm->setCtofnb(8.0);
  fm->initialize();
  auto ctx = std::make_shared<CharmmContext>(fm);

  std::vector<float3> inpcrd;
  float3 singlecrd;
  for (int i = 0; i < 10; i++) {
    singlecrd = make_float3(static_cast<float>(i), static_cast<float>(i),
                            static_cast<float>(i));
    inpcrd.push_back(singlecrd);
  }
  auto charmmCrd = std::make_shared<CharmmCrd>(inpcrd);
  CHECK_NOTHROW(ctx->setCoordinates(charmmCrd));
}

// Test that one can give a vec(vec(double)) to create a coordinate
// No assertion, but hopefully explodes if it doesn't work
TEST_CASE("vectorvector") {
  std::vector<std::vector<double>> inpcrd;
  inpcrd.push_back({1.0, 2.0, 3.0});
  inpcrd.push_back({4.0, 5.0, 6.0});
  auto crd = CharmmCrd(inpcrd);
}