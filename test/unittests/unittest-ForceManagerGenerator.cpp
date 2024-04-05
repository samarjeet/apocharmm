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
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "ForceManagerGenerator.h"
#include "catch.hpp"
#include "helper.h"
#include "test_paths.h"
#include <iostream>

/*
TEST_CASE("ForceManagerGenerator", "[debug]") {
  std::string dataPath = getDataPath();
  SECTION("Base") {
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    AlchemicalForceManagerGenerator generator =
        AlchemicalForceManagerGenerator(fm);
    double lElec = 0.5;
    double lVdw = 1.0;
    // Without alchemical region defined, we should get an error
    CHECK_THROWS(generator.generateForceManager(lElec, lVdw));

    // Check alchemicalRegion get/set
    std::vector<int> alchRegion = {0, 1, 2};
    generator.setAlchemicalRegion(alchRegion);
    std::vector<int> alchRegionOut = generator.getAlchemicalRegion();
    compareVectors(alchRegion, alchRegionOut);

    auto fm2 = generator.generateForceManager(lElec, lVdw);

    // check that charge scaling works
    auto psf2 = fm2->getPSF();
    std::vector<double> oldCharges = psf->getAtomCharges();
    std::vector<double> newCharges = psf2->getAtomCharges();
    for (int i = 0; i < alchRegion.size(); i++) {
      REQUIRE(newCharges[i] == lElec * oldCharges[i]);
    }
  } // End section

  SECTION("Electrostatics") {
    // Check that energy of a ForceManager created with a generator and
    // "modifyElectrostatics"  match those of obtained with a manually
    // generated PSF (with scaled charges)
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50.0, 50.0, 50.0});
    fm->setFFTGrid(48, 48, 48);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    fm->initialize();
    auto generator = AlchemicalForceManagerGenerator(fm);
    double lElec = 0.8, lVdw = 1.0;

    std::vector<int> alchRegion;
    for (int i = 0; i < 11748; i++) {
      alchRegion.push_back(i);
    }
    generator.setAlchemicalRegion(alchRegion);

    // fm_generated: created using Generator
    auto fm_generated = generator.generateForceManager(lElec, lVdw);
    fm_generated->initialize();

    // fm_manual: created from other PSF input file, with manually scaled
    // charges.
    auto psf_manual = std::make_shared<CharmmPSF>(
        dataPath + "waterbox_scaled_charges_by_l0.8.psf");
    auto fm_manual = std::make_shared<ForceManager>(psf_manual, prm);
    fm_manual->setBoxDimensions({50.0, 50.0, 50.0});
    fm_manual->setFFTGrid(48, 48, 48);
    fm_manual->setKappa(0.34);
    fm_manual->setCutoff(10.0);
    fm_manual->setCtonnb(7.0);
    fm_manual->setCtofnb(8.0);
    fm_manual->initialize();

    auto ctx_gen = std::make_shared<CharmmContext>(fm_generated);
    auto ctx_man = std::make_shared<CharmmContext>(fm_manual);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    ctx_gen->setCoordinates(crd);
    ctx_man->setCoordinates(crd);
    float pe_gen = ctx_gen->calculatePotentialEnergy();
    float pe_man = ctx_man->calculatePotentialEnergy();
    REQUIRE(pe_gen == pe_man);

  } // End section

  SECTION("Vdw") {
    // Test alchemically modified forceManager where sterics have been scaled
    // Idea: compare energy obtained with a prm file modified beforehand with
    // energy obtained with a ForceManager generateed by ForceManagerGenerator

    // WIP ! DOES FAIL FOR NOW
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50.0, 50.0, 50.0});
    fm->setFFTGrid(48, 48, 48);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    fm->initialize();
    auto generator = AlchemicalForceManagerGenerator(fm);
    double lElec = 1.0, lVdw = 0.7;
    std::vector<int> alchRegion;
    for (int i = 0; i < 11748; i++) {
      alchRegion.push_back(i);
    }
    generator.setAlchemicalRegion(alchRegion);

    // fm_generated: created using Generator
    // Not implemented yet -> should THROW
    std::shared_ptr<ForceManager> fm_generated;
    CHECK_THROWS(fm_generated = generator.generateForceManager(lElec, lVdw));

    // fm_generated->initialize();
    // auto ctx_gen = std::make_shared<CharmmContext>(fm_generated);

    // // fm_manual: created from manually modified PRM file, serves as
    // reference auto prm_manual =
    //
std::make_shared<CharmmParameters>(dataPath+"toppar_water_ions_manual.str");
    // auto fm_manual = std::make_shared<ForceManager>(psf, prm_manual);
    // fm_manual->setBoxDimensions({50.0, 50.0, 50.0});
    // fm_manual->setFFTGrid(48, 48, 48);
    // fm_manual->setKappa(0.34);
    // fm_manual->setCutoff(10.0);
    // fm_manual->setCtonnb(7.0);
    // fm_manual->setCtofnb(8.0);
    // fm_manual->initialize();
    // auto ctx_man = std::make_shared<CharmmContext>(fm_manual);

    // auto crd = std::make_shared<CharmmCrd>(dataPath+"waterbox.crd");
    // ctx_gen->setCoordinates(crd);
    // ctx_man->setCoordinates(crd);
    // float pe_gen = ctx_gen->calculatePotentialEnergy();
    // float pe_man = ctx_man->calculatePotentialEnergy();
    // REQUIRE(pe_gen == pe_man);

  } // end vdw section

} // End test_case
*/