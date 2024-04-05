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
#include "helper.h"
#include "test_paths.h"
#include <iostream>

// Check that TOPPAR and PAR files are rightly read
TEST_CASE("readPRM", "[preparation]") {
  std::string dataPath = getDataPath();
  std::vector<std::string> prmlist{dataPath + "toppar_water_ions.str",
                                   dataPath + "par_all36_prot.prm"};
  SECTION("readPrmFile") {
    auto prm = CharmmParameters(prmlist);

    // Before using a specific topology, check the quality of our prm parsing
    // Check bonds...
    BondKey bondkey("CT3", "NC2");
    auto bondprms = prm.getBonds();
    CHECK(390.0 == Approx(bondprms[bondkey].kb));
    CHECK(1.49 == Approx(bondprms[bondkey].b0));

    // Check angles (this one has UreyBradley terms to test as well)
    AngleKey anglekey("HB1", "CT1", "NH2");
    auto angleprms = prm.getAngles();
    CHECK(Approx(angleprms[anglekey].kTheta) == 38.0);
    CHECK(Approx(angleprms[anglekey].theta0 * 180.0 / std::acos(-1)) == 109.50);

    // Urey Bradley part
    AngleKey ureybkey("HB1", "CT1", "NH2");
    std::map<AngleKey, BondValues> ureybpms = prm.getUreyBradleys();
    CHECK(Approx(ureybpms[ureybkey].kb) == 50.0);
    CHECK(Approx(ureybpms[ureybkey].b0) == 2.14);

    // Dihedral
    // DihedralKey dihkey("NY", "CPT", "CPT", "CAI");
    DihedralKey dihkey("CAI", "CPT", "CPT", "NY");
    std::map<DihedralKey, std::vector<DihedralValues>> dihprms =
        prm.getDihedrals();
    // std::cout << "dihedral tuple: " << dihprms[dihkey][0] <<
    // dihprms[dihkey].size() << std::endl;
    DihedralValues importeddihval = dihprms[dihkey][0];
    CHECK(Approx(importeddihval.kChi) == 4.0);
    CHECK(Approx(importeddihval.delta) == 180.0);
    CHECK(importeddihval.n == 2);

    // Improper
    // DihedralKey imdihkey("NR1", "CPH2", "CPH1", "H");
    DihedralKey imdihkey("H", "CPH1", "CPH2", "NR1");
    std::map<DihedralKey, ImDihedralValues> imdihprms = prm.getImpropers();
    ImDihedralValues importedimdihval = imdihprms[imdihkey];
    CHECK(Approx(importedimdihval.kpsi) == 0.45);
    CHECK(Approx(importedimdihval.psi0) == 0.0);

    // vdw ?
    std::map<std::string, VdwParameters> vdwprms = prm.getVdwParameters();
    VdwParameters importedvdwval = vdwprms["CP1"];
    CHECK(Approx(importedvdwval.epsilon) == -0.02);
    CHECK(Approx(importedvdwval.rmin_2) == 2.275);
    // Vdw 1-4
    std::map<std::string, VdwParameters> vdw14prms = prm.getVdw14Parameters();
    VdwParameters importedvdw14val = vdw14prms["CP1"];
    CHECK(Approx(importedvdw14val.epsilon) == -0.01);
    CHECK(Approx(importedvdw14val.rmin_2) == 1.9);
  }

  // Read a first prm file.
  // Read a second prm file with modified values for certain floats.
  // Check that the new parameters have replaced the old ones.
  // ---> IT IS NOT THE CASE !
  SECTION("updatePrm") {
    auto prm = CharmmParameters(dataPath + "par_all36_cgenff.prm");
    // The four atoms whose parameters will change
    std::vector<std::string> modifiedAtoms{"HGA2", "HGA3", "CG321", "CG331"};

    std::map<std::string, VdwParameters> vdwprms = prm.getVdwParameters();
    std::vector<VdwParameters> prmsBefore, prmsAfter, prmsAfterBis;

    for (auto atom : modifiedAtoms) {
      prmsBefore.push_back(vdwprms[atom]);
    }

    // reading the modified prm file
    prm.readCharmmParameterFile(dataPath + "eds/par_all36_cgenff.prm.mod");
    // auto prmbis = CharmmParameters(dataPath +
    // "eds/par_all36_cgenff.prm.mod");
    vdwprms = prm.getVdwParameters();
    // auto vdwprmsbis = prmbis.getVdwParameters();
    for (auto atom : modifiedAtoms) {
      prmsAfter.push_back(vdwprms[atom]);
      // prmsAfterBis.push_back(vdwprmsbis[atom]);
    }

    INFO("Failure here means that loading a second parameter file does not "
         "update the parameters of the first one.")
    for (int i = 0; i < prmsAfter.size(); i++) {
      CHECK(prmsBefore[i].epsilon != prmsAfter[i].epsilon);
      CHECK(prmsBefore[i].rmin_2 != prmsAfter[i].rmin_2);
    }
  }
}

// Read prm file, read psf file,
// Assert that the built "bondedParamsAndLists" object has the right number of
// interactions and interaction types
// THIS IS PROBABLY USELESS ACTUALLY.
TEST_CASE("PRMandPSF") {
  std::string dataPath = getDataPath();
  // Since we have a small issue on vacuum DHFR, let's try on it first
  SECTION("vacuumDhfr") {
    auto prm = CharmmParameters(dataPath + "par_all36m_prot.prm");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "dhfr_vacuum.psf");
    auto bondedParamsAndLists = prm.getBondedParamsAndLists(psf);

    for (int i = 0; i < bondedParamsAndLists.paramsSize.size(); i++) {
      std::cout << "paramsSize[" << i
                << "] = " << bondedParamsAndLists.paramsSize[i] << std::endl;
    }
    for (int i = 0; i < bondedParamsAndLists.listsSize.size(); i++) {
      std::cout << "listsSize[" << i
                << "] = " << bondedParamsAndLists.listsSize[i] << std::endl;
    }

    // number of unique bonds ?
    CHECK(bondedParamsAndLists.listsSize[0] == 2523);
  }
  SECTION("dhfr") {
    std::vector<std::string> prmlist{dataPath + "toppar_water_ions.str",
                                     dataPath + "par_all36_prot.prm"};
    auto prm = CharmmParameters(prmlist);
    auto psf = std::make_shared<CharmmPSF>(dataPath + "dhfr.psf");
    auto result = prm.getBondedParamsAndLists(psf);

    CHECK(74 == result.paramsSize[0]);
    // CHECK(48 == result.paramsSize[1]);
    //  This 48 does not match apo's count.
    //  This number represents the number of UNIQUE urey-bradley interactions
    //  present in our system.
    //  To count this "manually", one would have to :
    //  - list all the angle interactions in the system
    //  - of this list, keep only the ones that have a Urey-Bradley term
    //  - find all the unique triplets of atom types represented in this list
    //  TODO unless I find a better idea
    CHECK(165 == result.paramsSize[2]);
    // CHECK(256 == result.paramsSize[3]);
    CHECK(15 == result.paramsSize[4]);
    CHECK(0 == result.paramsSize[5]);

    CHECK(22521 == result.listsSize[0]);
    // CHECK(2294 == result.listsSize[1]);
    CHECK(11227 == result.listsSize[2]);
    CHECK(6701 == result.listsSize[3]);
    CHECK(436 == result.listsSize[4]);
    CHECK(0 == result.listsSize[5]);

    // std::cout << "bond entry : " << result.listVal[0][0] << " "
    //           << result.listVal[0][1] << " " << result.listVal[0][2] << " "
    //           << result.listVal[0][3] << "\n";

    // std::cout << "bond param entry : "
    //           << result.paramsVal[result.listVal[0][2]][0] << " "
    //           << result.paramsVal[result.listVal[0][2]][1] << "\n";
    //
    auto vdwResult = prm.getVdwParamsAndTypes(psf);
    REQUIRE(psf->getNumAtoms() == vdwResult.vdwTypes.size());
  }
}

/*
SECTION("cholesterol") {
   //std::string tmppath = "/u/aviatfel/work/apocharmm/debug/";
   std::string tmppath = "/u/arice/tmp/test/";
   std::vector<std::string> prmnames = {"par_all36m_prot.prm",
"par_all36_lipid.prm", "toppar_all36_lipid_cholesterol.str",
"toppar_water_ions.str"}; std::vector<std::string> prmlist ; for (int i = 0; i <
prmnames.size(); i++) { std::cout << prmnames[i] << "   " ;
      prmlist.push_back(tmppath + prmnames[i]);
   }

   auto prm = std::make_shared<CharmmParameters>(prmlist);
   auto prm2 =
std::make_shared<CharmmParameters>(tmppath+"toppar_all36_lipid_cholesterol.str");
}*/
//  SECTION("antechamber prm") {
//
//    std::vector<std::string> prmFiles{
//        "/u/aviatfel/work/sampl9/revisit/cleanstart/setup/cpz.om.prm",
//        "/u/aviatfel/work/sampl9/revisit/cleanstart/setup/bcd.prm",
//        "/u/aviatfel/work/sampl9/revisit/cleanstart/setup/toppar_water.str"};
//
//    std::shared_ptr<CharmmParameters> prm =
//        std::make_shared<CharmmParameters>(prmFiles);
//
//    auto psf = std::make_shared<CharmmPSF>(
//        "/u/aviatfel/work/sampl9/revisit/cleanstart/setup/cpz.bcd.syst.psf");
//    auto result = prm->getBondedParamsAndLists(psf);
//    auto fm = std::make_shared<ForceManager>(psf, prm);
//    fm->setBoxDimensions({36.0, 36.0, 36.0});
//    fm->setFFTGrid(48, 48, 48);
//    fm->setKappa(0.34);
//    // fm->setKappa(0.0004);
//    fm->setCutoff(9.0);
//    fm->setCtonnb(7.0);
//    fm->setCtofnb(8.0);
//    fm->initialize();
//  }

// Investigating the issue of the order of prm loading that seem to produce
// different energy values => USELESS ! nvm
/*
TEST_CASE("dummyPrm") {
  std::string dataPath = getDataPath();
  std::string waterPrm = dataPath + "toppar_water_ions.str",
              cgenffPrm = dataPath + "par_all36_cgenff.prm",
              mobleyPrm = dataPath + "mobley_9979854.str", saiPrmPath, psfPath,
              crdPath;

  SECTION("vacuum") {
    saiPrmPath = dataPath + "../../examples/sai_params_vac/intst5/";
    psfPath = dataPath + "9979854.vacuum.psf";
    crdPath = dataPath + "9979854.vacuum.crd";
  }

  SECTION("solvated") {
    saiPrmPath = dataPath + "../../examples/sai_params/intst5/";
    psfPath = dataPath + "9979854.solv.psf";
    crdPath = dataPath + "9979854.solv.crd";
  }
  std::string dummyPrm = saiPrmPath + "dummy_parameters.prm";
  std::vector<std::string> prmlist1{waterPrm, cgenffPrm, mobleyPrm, dummyPrm},
      prmlist2{dummyPrm, waterPrm, cgenffPrm, mobleyPrm};

  auto psf = std::make_shared<CharmmPSF>(psfPath);
  auto crd = std::make_shared<CharmmCrd>(crdPath);

  // prm, then dummy
  auto prm1 = std::make_shared<CharmmParameters>(prmlist1);
  auto fm1 = std::make_shared<ForceManager>(psf, prm1);
  fm1->setBoxDimensions({36.0, 36.0, 36.0});
  auto ctx1 = std::make_shared<CharmmContext>(fm1);
  ctx1->setCoordinates(crd);
  ctx1->calculatePotentialEnergy();
  auto cc1 = ctx1->getPotentialEnergy();
  cc1.transferFromDevice();
  double ene1 = cc1[0];

  // dummy, then prm
  auto prm2 = std::make_shared<CharmmParameters>(prmlist2);
  auto fm2 = std::make_shared<ForceManager>(psf, prm2);
  fm2->setBoxDimensions({36.0, 36.0, 36.0});
  auto ctx2 = std::make_shared<CharmmContext>(fm2);
  ctx2->setCoordinates(crd);
  ctx2->calculatePotentialEnergy();
  auto cc2 = ctx2->getPotentialEnergy();
  cc2.transferFromDevice();
  double ene2 = cc2[0];

  REQUIRE(ene1 == Approx(ene2).epsilon(1e-9));
  std::cout << "ene1 = " << ene1 << " ene2 = " << ene2 << std::endl;
}
*/
