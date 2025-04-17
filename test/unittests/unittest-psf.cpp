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
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CharmmResidueTopology.h"
#include "ForceManager.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>
#include <memory>

#include "helper.h"



TEST_CASE("generate", "[setup]"){
  std::string dataPath = getDataPath();

  SECTION("PSF generation tests") {
    CharmmPSF psf;
    CharmmResidueTopology rtf;

    rtf.readRTF(dataPath + "top_all36_prot.rtf");

    // rtf.print();

    std::vector<std::string> sequence{"MET", "ILE"};
    std::string segment{"PROA"};
    psf.generate(rtf, sequence, segment);
    std::vector<std::string> water(1000, "TIP3");

    psf.append(rtf, sequence, "BWAT");
  }

}

TEST_CASE("readPSF", "[energy]") {
  std::string dataPath = getDataPath();

  SECTION("PSF generation tests") {
    CharmmPSF psf;
    CharmmResidueTopology rtf;

    rtf.readRTF(dataPath + "top_all36_prot.rtf");

    // rtf.print();

    std::vector<std::string> sequence{"MET", "ILE"};
    std::string segment{"PROA"};
    psf.generate(rtf, sequence, segment);
    std::vector<std::string> water(1000, "TIP3");

    psf.append(rtf, sequence, "BWAT");
  }
  /*
    // CharmmPSF psf("@CMAKE_SOURCE_DIR@/test/data/dhfr.psf");
    SECTION("DHFR PSF test") {
      CharmmPSF psf("../test/data/dhfr.psf");

      SECTION("Number of atoms, bonds, angles, etc") {
        REQUIRE(22498 == psf.getNumAtoms());
        REQUIRE(22521 == psf.getNumBonds());
        REQUIRE(11227 == psf.getNumAngles());
        REQUIRE(6701 == psf.getNumDihedrals());
        REQUIRE(436 == psf.getNumImpropers());

        compareFromFile(psf.getAtomTypes(), "../test/data/dhfr_atomTypes.txt");
        compareFromFile(psf.getAtomNames(), "../test/data/dhfr_atomNames.txt");
        compareFromFile(psf.getAtomMasses(),
    "../test/data/dhfr_atomMasses.txt"); compareFromFile(psf.getAtomCharges(),
                        "../test/data/dhfr_atomCharges.txt");
      }

      SECTION("iblo14 and inb14 tests") {
        auto iblo14 = psf.getIblo14();
        auto inb14 = psf.getInb14();

        compareAbsoluteFromFile<int>(iblo14, "../test/data/dhfr_iblo14.txt", 0);
        compareAbsoluteFromFile<int>(inb14, "../test/data/dhfr_inb14.txt", 0);
      }
    }

    SECTION("Waterbox PSF test") {
      CharmmPSF psf("../test/data/waterbox.psf");

      SECTION("Number of atoms, bonds, angles, etc") {
        REQUIRE(11748 == psf.getNumAtoms());
        REQUIRE(11748 == psf.getNumBonds());
        REQUIRE(3916 == psf.getNumAngles());
        REQUIRE(0 == psf.getNumDihedrals());
        REQUIRE(0 == psf.getNumImpropers());

        compareFromFile(psf.getAtomTypes(),
                        "../test/data/waterbox_atomTypes.txt");
        compareFromFile(psf.getAtomNames(),
                        "../test/data/waterbox_atomNames.txt");
        compareFromFile(psf.getAtomMasses(),
                        "../test/data/waterbox_atomMasses.txt");
        compareFromFile(psf.getAtomCharges(),
                        "../test/data/waterbox_atomCharges.txt");
      }

      SECTION("iblo14 and inb14 tests") {
        auto iblo14 = psf.getIblo14();
        auto inb14 = psf.getInb14();

        compareAbsoluteFromFile<int>(iblo14, "../test/data/waterbox_iblo14.txt",
                                     0);
        compareAbsoluteFromFile<int>(inb14, "../test/data/waterbox_inb14.txt",
    0);
      }
    }
    */
}
