// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Felix Aviat, Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmParameters.h"
#include "CudaContainer.h"
#include "CudaEnergyVirial.h"
#include "ForceManager.h"
// #include "boost/algorithm/string.hpp"
#include "catch.hpp"
#include "compare.h"
#include "cpp_utils.h"
#include "helper.h"
#include "test_paths.h"
#include <iostream>

// Idea: test the values of the energies and forces for several systems, compare
// with reference.
// Reference value are produced using Apocharmm, meaning that we're NOT
// comparing to CHARMM but to Apocharmm at a given commit for now.

bool comparePotentialEnergy(std::shared_ptr<CharmmContext> ctxIn,
                            double refVal) {
  auto eneContainer = ctxIn->getPotentialEnergy();
  eneContainer.transferFromDevice();
  double val = eneContainer.getHostArray()[0];
  return (refVal == Approx(val));
}

template <typename Type4> void printTriple(Type4 inp) {
  std::cout << inp.x << " " << inp.y << " " << inp.z << std::endl;
}

// Given a CharmmContext as input, extracts the forces as a CudaContainer of
// double4's
CudaContainer<double4>
getForcesAsCudaContainer(std::shared_ptr<CharmmContext> ctxIn) {
  CudaContainer<double> fx, fy, fz;
  int numAtoms = ctxIn->getNumAtoms();
  fx.allocate(numAtoms);
  fy.allocate(numAtoms);
  fz.allocate(numAtoms);
  fx.setDeviceArray(ctxIn->getForces()->x());
  fy.setDeviceArray(ctxIn->getForces()->y());
  fz.setDeviceArray(ctxIn->getForces()->z());
  fx.transferFromDevice();
  fy.transferFromDevice();
  fz.transferFromDevice();

  std::vector<double4> forcesvec;
  for (int i = 0; i < numAtoms; i++) {
    double4 f;
    f.x = fx[i];
    f.y = fy[i];
    f.z = fz[i];
    forcesvec.push_back(f);
  }
  CudaContainer<double4> forceCC;
  forceCC.allocate(numAtoms);
  forceCC.set(forcesvec);
  return forceCC;
}

// Given a file containing three numbers per line, returns a vector of double4
std::vector<double4> getForcesFromRefFile(std::string fname) {
  std::vector<double4> forces;
  std::ifstream refFile(fname);
  std::string line;

  if (!refFile.is_open()) {
    throw std::invalid_argument("ERROR: Cannot open the file " + fname +
                                "\nExiting.\n");
  }

  int lineCount = 0;
  while (std::getline(refFile, line)) {
    trim(line);
    if (line[0] == '!') {
      continue;
    } // dont look at comments
    lineCount++;
    double4 f;
    std::stringstream ss(line);
    ss >> f.x >> f.y >> f.z;
    forces.push_back(f);
  }
  return forces;
}

TEST_CASE("waterDimer") {
  // To emulate a gas phase calculation (no PME), we use a very low value of
  // Kappa (10^-8), a big box (520) and large cutoffs (~250).
  std::string dataPath = getDataPath();
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterDimer.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);

  double dim = 520;
  fm->setBoxDimensions({dim, dim, dim});
  fm->setFFTGrid(48, 48, 48);
  fm->setCutoff(255.0);
  fm->setCtonnb(250.0);
  fm->setCtofnb(254.0);

  fm->setKappa(0.00000001);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterDimer.crd");
  ctx->setCoordinates(crd);

  ctx->calculatePotentialEnergy(true, true);
  int numAtoms = ctx->getNumAtoms();

  /* c47b1 reference values, obtained with same cutoffs as stated above (box
  520,
   * cuts ~250) and same kappa value (10^-8). The energies obtained are the same
   * as using c47b1 with cutoffs of 999 and no PME.
  ENER ENR:  Eval#     ENERgy      Delta-E         GRMS
  ENER INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals IMPRopers
  ENER EXTERN:        VDWaals         ELEC       HBONds          ASP USER ENER
  EWALD:          EWKSum       EWSElf       EWEXcl       EWQCor       EWUTil
   ----------       ---------    ---------    ---------    --------- ---------
  ENER>        0      0.12057      0.00000      0.03003
  ENER INTERN>        0.00003      0.00002      0.00000      0.00000 0.00000
  ENER EXTERN>       -0.00132      0.12185      0.00000      0.00000 0.00000
  ENER EWALD>         0.00000     -0.00000      0.00000      0.00000 0.00000

   Internal Virial:
          0.4435916880        0.0000000000        0.1160047166
         -0.0000000000       -0.0071274780        0.0000000000
          0.1160047166        0.0000000000        0.0512243359

      1    1 TIP3 OH2    0.04978   0.00609  -0.03093 SEG1 1      0.00000
      2    1 TIP3 H1     0.03447  -0.00839   0.01529 SEG1 1      0.00000
      3    1 TIP3 H2    -0.04558   0.00230   0.03272 SEG1 1      0.00000
      4    2 TIP3 OH2   -0.04978   0.00609   0.03093 SEG1 2      0.00000
      5    2 TIP3 H1     0.04558   0.00230  -0.03272 SEG1 2      0.00000
      6    2 TIP3 H2    -0.03447  -0.00839  -0.01529 SEG1 2      0.00000

  */

  SECTION("energy") {
    std::map<std::string, double> refEneDecompositionMap;
    refEneDecompositionMap["bond"] = 0.00003;
    refEneDecompositionMap["angle"] = 0.00002;
    refEneDecompositionMap["vdw"] = -0.00132;
    refEneDecompositionMap["elec"] = 0.12185;

    auto eneDecompositionMap = fm->getEnergyComponents();
    for (auto ene : refEneDecompositionMap) {
      CHECK(ene.second == Approx(eneDecompositionMap[ene.first]).margin(.0001));
    }
    refEneDecompositionMap["epot"] = 0.12057;
    auto ePotContainer = ctx->getPotentialEnergy();
    ePotContainer.transferFromDevice();
    double ePot = ePotContainer.getHostArray()[0];
    CHECK(ePot == Approx(refEneDecompositionMap["epot"]).margin(.0001));
  }

  SECTION("virial") {
    std::vector<double> refVir = {
        0.4435916880, 0.000000000,  0.1160047166, -0.0000000000, -0.007127478,
        0.0000000000, 0.1160047166, 0.000000000,  0.0512243359,
    };

    auto virContainer = ctx->getVirial();
    std::vector<double> vir = virContainer.getHostArray();
    // for (int i =0 ; i<9; i++) {
    //     std::cout << vir[i] << " | " << refVir[i] << std::endl;
    // }
    CHECK(compareVectors(vir, refVir, 0.0001));
  }

  SECTION("forces") {
    std::string refForcesFile = "waterDimer.forces.c47b1.dat";
    std::vector<double4> forcesRef =
        getForcesFromRefFile(dataPath + refForcesFile);
    auto forcesContainer = getForcesAsCudaContainer(ctx);
    std::vector<double4> forcesCalc = forcesContainer.getHostArray();
    for (int i = 0; i < numAtoms; i++) {
      CHECK(compareTriples(forcesRef[i], forcesCalc[i], 0.005));
    }
  }
}

TEST_CASE("argon10") {
  std::string dataPath = getDataPath();
  // prepare system
  auto prm = std::make_shared<CharmmParameters>(dataPath + "argon.prm");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "argon_10.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  double dim = 50.0;

  fm->setBoxDimensions({dim, dim, dim});
  fm->setFFTGrid(48, 48, 48);
  fm->setKappa(0.34);
  fm->setCutoff(9.0);
  fm->setCtonnb(7.0);
  fm->setCtofnb(8.0);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "argon_10.crd");
  ctx->setCoordinates(crd);

  ctx->calculatePotentialEnergy(true, true);
  int numAtoms = ctx->getNumAtoms();

  SECTION("energy") {
    double ePotRef = -0.17984; // obtained via CHARMM c43b1
    CHECK(comparePotentialEnergy(ctx, ePotRef));
  }

  SECTION("forces") {
    double refVals[numAtoms][3] = {
        {0.00567, -0.00087, -0.01986}, {-0.00791, 0.00989, 0.00778},
        {0.00513, -0.00134, 0.00932},  {-0.00427, -0.00227, -0.00540},
        {-0.00334, 0.00684, 0.01087},  {-0.00301, -0.01385, -0.02313},
        {0.01136, 0.00056, -0.00589},  {-0.00778, 0.00468, 0.00800},
        {0.00337, 0.00260, 0.00934},   {0.00078, -0.00625, 0.00898}};
    // refVals obtained from CHARMM c43b1

    CudaContainer<double> fxc, fyc, fzc;
    fxc.allocate(numAtoms);
    fyc.allocate(numAtoms);
    fzc.allocate(numAtoms);
    fxc.setDeviceArray(ctx->getForces()->x());
    fyc.setDeviceArray(ctx->getForces()->y());
    fzc.setDeviceArray(ctx->getForces()->z());
    fxc.transferFromDevice();
    fyc.transferFromDevice();
    fzc.transferFromDevice();

    double4 val, refval;
    for (int i = 0; i < numAtoms; i++) {
      val.x = fxc[i];
      val.y = fyc[i];
      val.z = fzc[i];
      refval.x = refVals[i][0];
      refval.y = refVals[i][1];
      refval.z = refVals[i][2];
      CHECK(compareTriples(val, refval, 0.00001));
    }
  }
  SECTION("virial") {
    // Values obtained using c47b1:
    // Internal Virial:
    //       -0.0095837290        0.0066020280        0.0224834946
    //        0.0066020280       -0.0307467278       -0.0684782285
    //        0.0224834946       -0.0684782285       -0.0510912119

    std::vector<double> refVir = {-0.0095837290, 0.0066020280,  0.0224834946,
                                  0.0066020280,  -0.0307467278, -0.0684782285,
                                  0.0224834946,  -0.0684782285, -0.0510912119};

    CudaContainer<double> virContainer = ctx->getVirial();
    virContainer.transferFromDevice();
    std::vector<double> val = virContainer.getHostArray();

    CHECK(CompareVectors1(val, refVir, 1e-6, true));
  }
}

TEST_CASE("waterbox") {
  std::string dataPath = getDataPath();
  // prepare system
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  double dim = 50.0;

  fm->setBoxDimensions({dim, dim, dim});
  fm->setFFTGrid(48, 48, 48);
  fm->setKappa(0.34);
  fm->setCutoff(10.0);
  fm->setCtonnb(7.0);
  fm->setCtofnb(8.0);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  ctx->setCoordinates(crd);

  ctx->calculatePotentialEnergy(true, true);
  int numAtoms = ctx->getNumAtoms();

  SECTION("energy") {
    // ref values obtained via CHARMM c47b1 with domdec (single cpu)
    double ePotRef = -49755.02125, eBndRef = 1915.85339, eAngRef = 1071.29551,
           eVdwRef = 8697.43632, eElecRef = -56211.95319, eEwDirRef = 156.45211,
           eEwSelfRef = -260256.60420, eEwExclRef = 254872.49980;

    auto ePotContainer = ctx->getPotentialEnergy();
    ePotContainer.transferFromDevice();
    double ePot = ePotContainer.getHostArray()[0];
    auto eneDecompositionMap = fm->getEnergyComponents();

    CHECK(eneDecompositionMap["bond"] == Approx(eBndRef));
    CHECK(eneDecompositionMap["angle"] == Approx(eAngRef));
    CHECK(eneDecompositionMap["ewks"] == Approx(eEwDirRef));
    CHECK(eneDecompositionMap["ewse"] == Approx(eEwSelfRef));
    CHECK(eneDecompositionMap["ewex"] == Approx(eEwExclRef));
    CHECK(eneDecompositionMap["elec"] == Approx(eElecRef));
    CHECK(eneDecompositionMap["vdw"] == Approx(eVdwRef));
    CHECK(ePot == Approx(ePotRef));
  }

  SECTION("virial") {
    // Values obtained with c47b1
    // Internal Virial:
    //    -7071.0775896888     -315.6040942237     -605.0510971046
    //     -315.6040942237    -6636.9115981748     -255.8901308852
    //     -605.0510971047     -255.8901308853    -6935.7815227056
    std::vector<double> refVir = {
        -7071.0775896888, -315.6040942237,  -605.0510971046,
        -315.6040942237,  -6636.9115981748, -255.8901308852,
        -605.0510971047,  -255.8901308853,  -6935.7815227056};

    auto virContainer = ctx->getVirial();
    std::vector<double> vir = virContainer.getHostArray();

    // Tolerance would deserve a little bit more thinking...
    double tolerance = abs(.0001 * vir[0]);
    CHECK(compareVectors(vir, refVir, tolerance));
  }

  SECTION("forces") {
    // get reference force values from file
    std::string refForcesFile = "waterbox.forces.c47b1.dat";
    std::vector<double4> forcesRef =
        getForcesFromRefFile(dataPath + refForcesFile);

    // get calculated force values into a new container
    auto forcesContainer = getForcesAsCudaContainer(ctx);
    std::vector<double4> forcesCalc = forcesContainer.getHostArray();

    for (int i = 0; i < numAtoms; i++) {
      CHECK(compareTriples(forcesRef[i], forcesCalc[i], 0.01));
    }
  }
}

TEST_CASE("protein") {
  std::string dataPath = getDataPath();
  std::vector<std::string> prmlist = {dataPath + "par_all36m_prot.prm",
                                      dataPath + "toppar_water_ions.str"};
  auto prm = std::make_shared<CharmmParameters>(prmlist);
  auto psf = std::make_shared<CharmmPSF>(
      dataPath + "dhfr_30k.psf"); //"dhfr.charmm_gui.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);

  double boxdim = 69.0;

  fm->setBoxDimensions({boxdim, boxdim, boxdim});
  fm->setFFTGrid(64, 64, 64);
  fm->setKappa(0.34);
  fm->setCutoff(10.0);
  fm->setCtofnb(8.0);
  fm->setCtonnb(7.0);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "dhfr_30k.crd");
  ctx->setCoordinates(crd);

  ctx->calculatePotentialEnergy();
  int numAtoms = ctx->getNumAtoms();

  SECTION("energy") {
    /*
        // ref values obtained via CHARMM c47b1, domdec, skip cmap, 8 cpus
    ENER ENR:  Eval#     ENERgy      Delta-E         GRMS
    ENER INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals IMPRopers
    ENER EXTERN:        VDWaals         ELEC       HBONds          ASP USER ENER
    EWALD:          EWKSum       EWSElf       EWEXcl       EWQCor       EWUTil
     ----------       ---------    ---------    ---------    --------- ---------
    ENER>        0-120502.02161      0.00000      0.92869
    ENER INTERN>     4368.33733   2891.80976     29.95231
    1545.61401     22.57524 ENER EXTERN>    17390.92101-127504.18018 0.00000
    0.00000      0.00000 ENER EWALD>      1029.61007-635156.01213 614879.35097
    0.00000      0.00000
     ----------       ---------    ---------    ---------    ---------
    ---------*/
    double ePotRef = -120502.02161;
    std::map<std::string, double> refEneDecompositionMap;

    refEneDecompositionMap["bond"] = 4368.33733;
    refEneDecompositionMap["angle"] = 2891.80976;
    refEneDecompositionMap["ureyb"] = 29.95231;
    refEneDecompositionMap["dihe"] = 1545.61401;
    refEneDecompositionMap["imdihe"] = 22.57524;
    refEneDecompositionMap["ewks"] = 1029.61007;
    refEneDecompositionMap["ewse"] = -635156.01213;
    refEneDecompositionMap["ewex"] = 614879.35097;
    refEneDecompositionMap["elec"] = -127504.18018;
    refEneDecompositionMap["vdw"] = 17390.92101;

    auto eneDecompositionMap = fm->getEnergyComponents();
    for (auto ene : eneDecompositionMap) {
      CHECK(ene.second == Approx(refEneDecompositionMap[ene.first]));
    }

    auto ePotContainer = ctx->getPotentialEnergy();
    ePotContainer.transferFromDevice();
    double ePot = ePotContainer.getHostArray()[0];

    CHECK(ePot == Approx(ePotRef));
  }

  SECTION("virial") {
    /*  Values obtained with c47b1, domdec, 8 cpus, skip cmap, no nbfix, vfsw
    Internal Virial:
   -24825.2468277618     1027.9234994995    -1519.8511086677
     1027.9234994994   -22478.9127476412     -308.3314307795
    -1519.8511086677     -308.3314307794   -21590.6634904437*/

    std::vector<double> refVir = {
        -24825.2468277618, 1027.9234994995,   -1519.8511086677,
        1027.9234994994,   -22478.9127476412, -308.3314307795,
        -1519.8511086677,  -308.3314307794,   -21590.6634904437};

    auto virContainer = ctx->getVirial();
    std::vector<double> vir = virContainer.getHostArray();
    // Tolerance would deserve a little bit more thinking...
    for (int i = 0; i < 9; i++) {
      std::cout << vir[i] << " | " << refVir[i] << std::endl;
    }

    // double tolerance = abs(.0001 * vir[0]);
    double tolerance = .2;
    CHECK(compareVectors(vir, refVir, tolerance));
  }

  SECTION("forces") {
    std::string refForcesFile = "dhfr_30k.forces.c47b1.dat";
    std::vector<double4> forcesRef =
        getForcesFromRefFile(dataPath + refForcesFile);
    auto forcesContainer = getForcesAsCudaContainer(ctx);
    std::vector<double4> forcesCalc = forcesContainer.getHostArray();
    for (int i = 0; i < numAtoms; i++) {
      CHECK(compareTriples(forcesRef[i], forcesCalc[i], 0.005));
    }
  }
}
// Protein alone in vacuum
TEST_CASE("vacuumDHFR") {
  std::string dataPath = getDataPath();
  INFO("USING ABSOLUTE PATHS IN THERE");
  // auto prm = std::make_shared<CharmmParameters>(dataPath +
  // "par_all36m_prot.prm");
  auto prm = std::make_shared<CharmmParameters>(
      "/u/aviatfel/work/charmm/toppar/par_all36m_prot.prm");
  // auto psf = std::make_shared<CharmmPSF>(dataPath + "dhfr_vacuum.psf");
  auto psf = std::make_shared<CharmmPSF>(
      "/u/aviatfel/work/charmm/goldenstandardruns/dhfr_vacuum/vacuum_dhfr.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setCutoff(490.0);
  fm->setCtofnb(489.0);
  fm->setCtonnb(488.0);
  fm->setKappa(1e-9);
  fm->setBoxDimensions({1000., 1000., 1000.});
  fm->setFFTGrid(1000, 1000, 1000);

  auto ctx = std::make_shared<CharmmContext>(fm);
  // auto crd = std::make_shared<CharmmCrd>(dataPath + "dhfr_vacuum.crd");
  auto crd =
      std::make_shared<CharmmCrd>("/u/aviatfel/work/charmm/goldenstandardruns/"
                                  "dhfr_vacuum/vacuum_dhfr_mini.crd");
  ctx->setCoordinates(crd);
  fm->setPrintEnergyDecomposition(true);
  ctx->calculatePotentialEnergy();
  int numAtoms = ctx->getNumAtoms();

  SECTION("energy") {
    /* Ref values obtained via c47b1 domdec, 8 procs, vfsw, skip CMAP
    ENER ENR:  Eval#     ENERgy      Delta-E         GRMS
    ENER INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals IMPRopers
    ENER EXTERN:        VDWaals         ELEC       HBONds          ASP USER
    ----------       ---------    ---------    ---------    --------- ---------
    ENER>        0  -1797.32651      0.00000      1.31399
    ENER INTERN>      165.26978    491.21350     27.36517  1540.88685  25.83385
    ENER EXTERN>     -659.85308  -3388.04258      0.00000     0.00000   0.00000
    ----------       ---------    ---------    ---------    --------- ---------
    */

    auto dihedrals = psf->getDihedrals();
    std::cout << "Number of dihedrals: " << dihedrals.size() << std::endl;

    double ePotRef = -1779.07837;
    std::map<std::string, double> refEneDecompositionMap;
    refEneDecompositionMap["bond"] = 165.26978;
    refEneDecompositionMap["angle"] = 491.21350;
    refEneDecompositionMap["ureyb"] = 27.36517;
    refEneDecompositionMap["dihe"] = 1540.88685;
    refEneDecompositionMap["imdihe"] = 25.83385;
    // TODO: Rerun in CHARMM to get the right value
    refEneDecompositionMap["vdw"] = -659.85308;
    // refEneDecompositionMap["elec"] = -3388.04258;

    auto eneDecompositionMap = fm->getEnergyComponents();
    for (const auto &ene : refEneDecompositionMap) {
      INFO("Energy component: " << ene.first);
      CHECK(eneDecompositionMap[ene.first] ==
            Approx(refEneDecompositionMap[ene.first]));
    }
    auto ePotContainer = ctx->getPotentialEnergy();
    ePotContainer.transferFromDevice();
    double ePot = ePotContainer.getHostArray()[0];
    // TODO: Rerun in CHARMM to get the right value
    // CHECK(ePot == Approx(ePotRef));
  }
}

// System walp uses c36, includes prot/lipids/wat/ions.
TEST_CASE("walp") {
  std::string dataPath = getDataPath();
  std::vector<std::string> prmFiles{dataPath + "par_all36m_prot.prm",
                                    dataPath + "par_all36_lipid.prm",
                                    dataPath + "toppar_water_ions.str"};
  std::shared_ptr<CharmmParameters> prm =
      std::make_shared<CharmmParameters>(prmFiles);
  std::shared_ptr<CharmmPSF> psf =
      std::make_shared<CharmmPSF>(dataPath + "walp.psf");

  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions({53.463, 53.463, 80.493});
  fm->setFFTGrid(64, 64, 64);
  fm->setKappa(0.34);
  fm->setCutoff(9.0);
  fm->setCtonnb(7.0);
  fm->setCtofnb(8.0);
  fm->initialize();

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "walp.crd");
  fm->setPrintEnergyDecomposition(true);
  ctx->setCoordinates(crd);
  ctx->calculatePotentialEnergy();
  int numAtoms = ctx->getNumAtoms();

  SECTION("energy") {
    /* Ref values obtained via CHARMM c47b1 domdec, 8 proc, vfsw, skip CMAP
      ENER ENR:  Eval#     ENERgy      Delta-E         GRMS
      ENER INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals
      IMPRopers ENER EXTERN:        VDWaals         ELEC       HBONds ASP USER
      ENER EWALD:          EWKSum       EWSElf       EWEXcl       EWQCor EWUTil
       ----------       ---------    ---------    ---------    ---------
      --------- ENER>        0 -38664.44197      0.00000     13.96531 ENER
      INTERN>     1298.17587   4981.26712   1809.80188   5078.85866     58.69799
      ENER EXTERN>     1992.66312 -43783.46705      0.00000      0.00000 0.00000
      ENER EWALD>       768.33542-324603.04623 313734.27125      0.00000 0.00000
       ----------       ---------    ---------    ---------    ---------
      ---------
      */

    double ePotRef = -38664.44197;
    std::map<std::string, double> refEneDecompositionMap;

    refEneDecompositionMap["bond"] = 1298.17587;
    refEneDecompositionMap["angle"] = 4981.26712;
    refEneDecompositionMap["ureyb"] = 1809.80188;
    refEneDecompositionMap["dihe"] = 5078.85866;
    refEneDecompositionMap["imdihe"] = 58.69799;
    refEneDecompositionMap["ewks"] = 768.33542;
    refEneDecompositionMap["ewse"] = -324603.04623;
    refEneDecompositionMap["ewex"] = 313734.27125;
    refEneDecompositionMap["elec"] = -43783.46705;
    refEneDecompositionMap["vdw"] = 1992.66312;

    auto eneDecompositionMap = fm->getEnergyComponents();
    for (auto ene : eneDecompositionMap) {
      INFO("Energy component: " << ene.first);
      CHECK(ene.second ==
            Approx(refEneDecompositionMap[ene.first]).margin(.01));
    }

    auto ePotContainer = ctx->getPotentialEnergy();
    ePotContainer.transferFromDevice();
    double ePot = ePotContainer.getHostArray()[0];

    CHECK(ePot == Approx(ePotRef).margin(.01));
  }

  SECTION("virial") {
    /* Ref values obtained via CHARMM c47b1 domdec, 8procs, vfsw, skip CMAP
    Internal Virial:
39224.1707098938     -433.5312145424      -67.7859568847
 -433.5312145424    38137.4671985712      592.5360859065
  -67.7859568847      592.5360859065    36366.5128544932*/
    std::vector<double> refVir = {
        39224.1707098938, -433.5312145424,  -67.7859568847,
        -433.5312145424,  38137.4671985712, 592.5360859065,
        -67.7859568847,   592.5360859065,   36366.5128544932};

    auto virContainer = ctx->getVirial();
    std::vector<double> vir = virContainer.getHostArray();
    for (int i = 0; i < 9; i++) {
      std::cout << vir[i] << " | " << refVir[i] << std::endl;
    }

    // Tolerance would deserve a little bit more thinking...
    double tolerance = abs(0.00001 * refVir[0]);
    CHECK(compareVectors(vir, refVir, tolerance));
  }
}

// A 136 DPPG bilayer in water, with ions, generated with CHARMM-GUI
TEST_CASE("lipidBilayer") {
  std::string dataPath = getDataPath();
  std::vector<std::string> prmFiles{dataPath + "par_all36_lipid.prm",
                                    dataPath + "toppar_water_ions.str"};
  std::shared_ptr<CharmmParameters> prm =
      std::make_shared<CharmmParameters>(prmFiles);
  std::shared_ptr<CharmmPSF> psf =
      std::make_shared<CharmmPSF>(dataPath + "bilayer.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);

  std::vector<double> boxDim = {65.4523, 65.4523, 85.};
  fm->setBoxDimensions(boxDim);
  fm->setFFTGrid(72, 72, 90);
  fm->setCtonnb(8.);
  fm->setCtofnb(9.);
  fm->setCutoff(10.);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "bilayer.crd");
  ctx->setCoordinates(crd);

  ctx->calculatePotentialEnergy();
  int numAtoms = ctx->getNumAtoms();

  SECTION("energy") {
    /* Ref values obtained via CHARMM c47b1 domdec, 8 proc, vfsw, skip CMAP
    ENER ENR:  Eval#     ENERgy      Delta-E         GRMS
    ENER INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals IMPRopers
    ENER EXTERN:        VDWaals         ELEC       HBONds          ASP USER ENER
    EWALD:          EWKSum       EWSElf       EWEXcl       EWQCor       EWUTil
    ----------       ---------    ---------    ---------    --------- ---------
    ENER>        0 -28652.62960      0.00000      9.06674
    ENER INTERN>     2742.34318   4256.24402   1320.45121
    7063.85465     52.82515 ENER EXTERN>     7533.53305 -64758.94150 0.00000
    0.00000      0.00000 ENER EWALD>     33535.72630-472054.06129 451655.39562
    0.00000      0.00000
    ----------       ---------    ---------    ---------    --------- ---------
  */
    double ePotRef = -28652.62960;
    std::map<std::string, double> refEneDecompositionMap;
    refEneDecompositionMap["bond"] = 2742.34318;
    refEneDecompositionMap["angle"] = 4256.24402;
    refEneDecompositionMap["ureyb"] = 1320.45121;
    refEneDecompositionMap["dihe"] = 7063.85465;
    refEneDecompositionMap["imdihe"] = 52.82515;
    refEneDecompositionMap["ewks"] = 33535.72630;
    refEneDecompositionMap["ewse"] = -472054.06129;
    refEneDecompositionMap["ewex"] = 451655.39562;
    refEneDecompositionMap["elec"] = -64758.94150;
    refEneDecompositionMap["vdw"] = 7533.53305;

    auto eneDecompositionMap = fm->getEnergyComponents();
    for (auto ene : eneDecompositionMap) {
      INFO("Energy component:" + ene.first);
      CHECK(ene.second == Approx(refEneDecompositionMap[ene.first]));
    }

    auto ePotContainer = ctx->getPotentialEnergy();
    ePotContainer.transferFromDevice();
    double ePot = ePotContainer.getHostArray()[0];
    CHECK(ePot == Approx(ePotRef));
  }

  SECTION("virial") {
    /* Ref values obtained via CHARMM c47b1 domdec, 8procs, vfsw, skip CMAP
    Internal Virial:
    42822.6304901300       30.8378683793    -1119.4148925409
    30.8378683793    44519.9650364109      740.6580284485
    -1119.4148925409      740.6580284485   -27364.0573639815*/
    std::vector<double> refVir = {
        42822.6304901300, 30.8378683793,    -1119.4148925409,
        30.8378683793,    44519.9650364109, 740.6580284485,
        -1119.4148925409, 740.6580284485,   -27364.0573639815};

    auto virContainer = ctx->getVirial();
    std::vector<double> vir = virContainer.getHostArray();
    double tolerance = abs(0.00001 * refVir[0]);
    CHECK(compareVectors(vir, refVir, tolerance));
  }

  SECTION("forces") {
    std::vector<double4> forcesRef =
        getForcesFromRefFile(dataPath + "bilayer.forces.c47b1.dat");
    auto forcesContainer = getForcesAsCudaContainer(ctx);
    std::vector<double4> forcesCalc = forcesContainer.getHostArray();
    for (int i = 0; i < numAtoms; i++) {
      CHECK(compareTriples(forcesRef[i], forcesCalc[i], 0.005));
    }
  }
}

TEST_CASE("waterboxDoubleExponential") {
  std::string dataPath = getDataPath();
  // prepare system
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  double dim = 50.0;

  fm->setBoxDimensions({dim, dim, dim});
  fm->setFFTGrid(48, 48, 48);
  fm->setKappa(0.34);
  fm->setCutoff(10.0);
  fm->setCtonnb(7.0);
  fm->setCtofnb(8.0);

  fm->setVdwType(VDW_DBEXP);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  ctx->setCoordinates(crd);

  ctx->calculatePotentialEnergy(true, true);
  int numAtoms = ctx->getNumAtoms();
  SECTION("dbexp") { REQUIRE(fm->getVdwType() == VDW_DBEXP); }
  SECTION("energy") {
    // ref values obtained via CHARMM c47b1 with domdec (single cpu)
    double ePotRef = -49755.02125, eBndRef = 1915.85339, eAngRef = 1071.29551,
           eVdwRef = 8697.43632, eElecRef = -56211.95319, eEwDirRef = 156.45211,
           eEwSelfRef = -260256.60420, eEwExclRef = 254872.49980;

    auto ePotContainer = ctx->getPotentialEnergy();
    ePotContainer.transferFromDevice();
    double ePot = ePotContainer.getHostArray()[0];
    auto eneDecompositionMap = fm->getEnergyComponents();

    CHECK(eneDecompositionMap["bond"] == Approx(eBndRef));
    CHECK(eneDecompositionMap["angle"] == Approx(eAngRef));
    CHECK(eneDecompositionMap["ewks"] == Approx(eEwDirRef));
    CHECK(eneDecompositionMap["ewse"] == Approx(eEwSelfRef));
    CHECK(eneDecompositionMap["ewex"] == Approx(eEwExclRef));
    CHECK(eneDecompositionMap["elec"] == Approx(eElecRef));
    // TODO:rerun in CHARMM with dbexp to get the right value
    // CHECK(eneDecompositionMap["vdw"] == Approx(eVdwRef));
    // CHECK(ePot == Approx(ePotRef));
  }
}