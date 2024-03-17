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
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "MBARForceManager.h"
#include "MBARSubscriber.h"
#include "XYZQ.h"
#include "catch.hpp"
#include "helper.h"
#include "test_paths.h"
#include <iostream>
#include <vector>

// TEST_CASE("forceManager", "[force]") {
//  SECTION("2water") {
//    std::unique_ptr<CharmmParameters> prm =
//    std::make_unique<CharmmParameters>(
//        dataPath+"toppar_water_ions.str");
//    std::unique_ptr<CharmmPSF> psf =
//        std::make_unique<CharmmPSF>(dataPath+"water2.psf");
//
//    // TODO : this isn't how we'll be using it
//    // Only charmmcontext will have to be created,
//    // we will take the force manager from it.
//    auto fm = std::make_shared<ForceManager>(psf, prm);
//    fm->setBoxDimensions({50.0, 50.0, 50.0});
//    fm->setKappa(0.34);
//    fm->setFFTGrid(48, 48, 48);
//    fm->setCutoff(8.0);
//    fm->initialize();
//
//    auto ctx = std::make_shared<CharmmContext>(fm);
//    auto crd = std::make_shared<CharmmCrd>(dataPath+"water2.crd");
//    ctx->setCoordinates(crd);
//
//    ctx->calculatePotentialEnergy(true, true);
//    // TODO : compare forces
//  }
//}

TEST_CASE("ForceManager", "[unit]") {
  std::string dataPath = getDataPath();
  SECTION("Input files") {
    // Check how psf/prm files are handled.
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    // check with filenames
    std::string prmfile = dataPath + "par_all36_cgenff.prm";
    std::vector<std::string> prmlist = {dataPath + "par_all36_cgenff.prm",
                                        dataPath + "toppar_water_ions.str"};
    CHECK_NOTHROW(fm->addPRM(prmfile));
    CHECK_NOTHROW(fm->addPRM(prmlist));
    REQUIRE(fm->isInitialized() == false);
    std::string psffile = dataPath + "argon_10.psf";
    CHECK_NOTHROW(fm->addPSF(psffile));

    // numAtoms
    REQUIRE(fm->getNumAtoms() == 10);

    // setBoxDimensions
    std::vector<double> sizeIn = {12., 12., 12.};
    fm->setBoxDimensions(sizeIn);
    auto sizeOut = fm->getBoxDimensions();
    std::vector<double> sizeInFloat = {(double)sizeIn[0], (double)sizeIn[1],
                                       (double)sizeIn[2]};
    REQUIRE(compareVectors(sizeInFloat, sizeOut));
  }
  SECTION("Initialization") {
    auto prm = std::make_shared<CharmmParameters>(dataPath + "argon.prm");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "argon_10.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);

    // Without fft grid and box dimension specified, should throw
    CHECK_THROWS(fm->initialize());
    CHECK(fm->isInitialized() == false);

    fm->setFFTGrid(12, 12, 12);
    CHECK_THROWS(fm->setBoxDimensions({-1., 12., 13.}));
    fm->setBoxDimensions({32., 32., 32.});

    // check that initialize does change the initialized flag \o/ (yes, that has
    // been an issue...)
    fm->initialize();
    CHECK(fm->isInitialized() == true);
  }
  SECTION("Copy constructor") {
    auto prm = std::make_shared<CharmmParameters>(dataPath + "argon.prm");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "argon_10.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    float a = 44.44, b = 55.55, c = 66.66;

    fm->setBoxDimensions({a, a, a});
    fm->setFFTGrid(48, 48, 48);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(9.9);
    fm->setCtofnb(10.1);

    auto fm2 = std::make_shared<ForceManager>(*fm);
    REQUIRE(compareVectors(fm2->getFFTGrid(), fm->getFFTGrid()));

    // Check that copied FM has same box dim
    REQUIRE(compareVectors({a, a, a}, fm2->getBoxDimensions()));

    // Check that modifying copied FM did not modify original FM
    fm2->setBoxDimensions({b, b, b});
    REQUIRE(compareVectors({a, a, a}, fm->getBoxDimensions()));

    // Check that modifying original FM does not modify copied FM
    fm->setBoxDimensions({c, c, c});
    REQUIRE(compareVectors({b, b, b}, fm2->getBoxDimensions()));

    // Check that all options are indeed transferred/copied
    REQUIRE(fm->getKappa() == fm2->getKappa());
    REQUIRE(fm->getCutoff() == fm2->getCutoff());
    REQUIRE(fm->getCtonnb() == fm2->getCtonnb());
    REQUIRE(fm->getCtofnb() == fm2->getCtofnb());
  }
  SECTION("Runtime") {
    auto prm = std::make_shared<CharmmParameters>(dataPath + "argon.prm");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "argon1000.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    std::vector<double> sizeIn = {50., 50., 50.};
    fm->setBoxDimensions(sizeIn);
    fm->setFFTGrid(48, 48, 48);
    fm->setCutoff(12.0);

    // Giving a neg value for cutoff should FAIL
    std::shared_ptr<CharmmContext> ctxfail;
    fm->setCutoff(-12.0);
    CHECK_THROWS(ctxfail = std::make_shared<CharmmContext>(fm));

    fm->setCutoff(12.0);
    auto ctx = std::make_shared<CharmmContext>(fm);
    fm->setCharmmContext(ctx);

    auto crd = std::make_shared<CharmmCrd>(dataPath + "argon_1000.crd");
    ctx->setCoordinates(crd);
  }

  // Test the FFTGrid setter
  SECTION("FFTGrid") {
    auto prm = std::make_shared<CharmmParameters>(dataPath + "argon.prm");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "argon_10.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    float a = 44.44, b = 55.55, c = 66.66;
    int nfft = 48;
    std::vector<int> nfftvec = {nfft, nfft, nfft};
    std::vector<int> nfftbis;
    std::vector<int> nfftter{44, 44, 44};

    fm->setBoxDimensions({a, a, a});
    fm->initialize();

    nfftbis = fm->getFFTGrid();
    CHECK(nfftbis == nfftter);

    fm->setFFTGrid(nfft, nfft, nfft);
    CHECK(fm->getFFTGrid() == nfftvec);
  }
  // Functions remaining to be tested:
  //----------------------------------
  // setKappa
  // setCutoff
  // setCtonnb
  // setCtofnb
  // setPmeSplineOrder
  // setPeriodicBoundaryCondition
  // getForceStride
  // getBonds
  // getPotentialEnergy
  // getVirial
  //
  // to be tested in CudaPMEDirectForce:
  //  * resetNeighborList
  // to be tested by ForceManager:
  //  * calc_force
  //  * getForces
  //  * emplace_back
}

TEST_CASE("ForceManagerComposite", "[unit]") {
  std::string dataPath = getDataPath();
  SECTION("Constructors") {
    std::shared_ptr<ForceManager> fm1, fm2;
    std::vector<std::shared_ptr<ForceManager>> fmlist;
    std::shared_ptr<ForceManagerComposite> fmc, fmc2;
    CHECK_NOTHROW(fmc = std::make_shared<ForceManagerComposite>());

    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    fm1 = std::make_shared<ForceManager>(psf, prm);
    fm2 = std::make_shared<ForceManager>(psf, prm);
    fmlist = {fm1, fm2};

    CHECK_NOTHROW(fmc2 = std::make_shared<ForceManagerComposite>(fmlist));
  }

  SECTION("Basics") {
    auto fmc = std::make_shared<ForceManagerComposite>();
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto fm1 = std::make_shared<ForceManager>(psf, prm);
    auto fm2 = std::make_shared<ForceManager>(psf, prm);

    // addForceManager
    fmc->addForceManager(fm1);
    CHECK_NOTHROW(fmc->addForceManager(fm2));
    // getCompositeSize
    REQUIRE(fmc->getCompositeSize() == 2);
    // isComposite
    REQUIRE(fmc->isComposite());

    fmc->setBoxDimensions({32., 32., 32.});
    fmc->setFFTGrid(12, 12, 12);
    CHECK(fmc->isInitialized() == false);
    fmc->initialize();
    CHECK(fmc->isInitialized());
  }

  // Test that getPotentialEnergy returns PE from EACH child
  SECTION("ChildEnergy") {
    auto fmc = std::make_shared<ForceManagerComposite>();
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto fm1 = std::make_shared<ForceManager>(psf, prm);
    auto fm2 = std::make_shared<ForceManager>(psf, prm);
    fmc->addForceManager(fm1);
    fmc->addForceManager(fm2);
    fmc->setBoxDimensions({50., 50., 50.});
    auto ctx = std::make_shared<CharmmContext>(fmc);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    ctx->setCoordinates(crd);
    ctx->calculatePotentialEnergy();
    auto peCC = ctx->getPotentialEnergy();
    peCC.transferFromDevice();
    std::cout << "PE from each child: " << peCC[0] << "   " << peCC[1]
              << std::endl;
  }
}
TEST_CASE("MBARForceManager", "[unit]") {
  // Test that we can compute energy for each child
  // We'll use Transformato-generated input for that
  // (methane -> CH3-(dummy))
  std::string dataPath = getDataPath();
  // Only assertion here: the energy values are non-zero
  SECTION("ChildEnergy") {
    int nIntermediates = 3;
    std::string stateDir, ccsaiDataPath = dataPath + "methaneCCSAI/";
    std::vector<std::string> prmlist = {dataPath + "toppar_water_ions.str",
                                        dataPath + "par_all36_cgenff.prm",
                                        "tobereplaced"};

    auto fmc = std::make_shared<MBARForceManager>();
    for (int i = 0; i < nIntermediates; i++) {
      stateDir = ccsaiDataPath + "intst" + std::to_string(i + 1) + "/";
      prmlist[2] = stateDir + "dummy_parameters.prm";
      auto prm = std::make_shared<CharmmParameters>(prmlist);
      auto psf = std::make_shared<CharmmPSF>(stateDir + "methane_aq.psf");
      auto fm = std::make_shared<ForceManager>(psf, prm);
      fm->setBoxDimensions({30., 30., 30.});
      fmc->addForceManager(fm);
    }

    fmc->setSelectorVec({0., 1., 0.});
    fmc->initialize();
    auto ctx = std::make_shared<CharmmContext>(fmc);
    auto crd = std::make_shared<CharmmCrd>(stateDir + "methane_aq.crd");
    ctx->setCoordinates(crd);
    ctx->assignVelocitiesAtTemperature(300.0);

    auto integrator = CudaLangevinThermostatIntegrator(0.001, 300.0, 5.0);
    integrator.setSimulationContext(ctx);
    integrator.propagate(10);

    // Now to get some energies...
    XYZQ *xyzq = ctx->getXYZQ();
    xyzq->transferFromDevice();
    float4 *xyzqPointer = xyzq->xyzq;
    auto childEnergies = fmc->computeAllChildrenPotentialEnergy(xyzqPointer);
    childEnergies.transferFromDevice();
    for (int i = 0; i < nIntermediates; i++) {
      std::cout << childEnergies[i] << std::endl;
      CHECK(childEnergies[i] != 0.0);
    }

    // Trying to change selectorVec ?
    fmc->setSelectorVec({0., 0., 1.});
    childEnergies = fmc->computeAllChildrenPotentialEnergy(xyzqPointer);
    childEnergies.transferFromDevice();
    for (int i = 0; i < nIntermediates; i++) {
      CHECK(childEnergies[i] != 0.0);
    }
  }

  // Assert that MBARfm energy corresponds to the child[selector] energy
  SECTION("ASFE") {
    int molid = 9979854;
    int numberOfAlchemicalStates = 13;
    std::vector<std::shared_ptr<ForceManager>> fmList;
    std::string saiDataPath = dataPath + "../../examples/sai_params";

    // Create force manager of each alchemical window
    for (int stateNumber = 0; stateNumber < numberOfAlchemicalStates;
         stateNumber++) {
      std::vector<std::string> prmlist = {
          dataPath + "toppar_water_ions.str", dataPath + "par_all36_cgenff.prm",
          dataPath + "mobley_" + std::to_string(molid) + ".str"};

      std::string stateDir =
          saiDataPath + "/intst" + std::to_string(stateNumber) + "/";
      if (stateNumber > 3) { // TODO: uncomment when needed
        prmlist.push_back(stateDir + "dummy_parameters.prm");
      }

      auto prm = std::make_shared<CharmmParameters>(prmlist);
      auto psf = std::make_shared<CharmmPSF>(stateDir + "/ligand.psf");
      auto fm = std::make_shared<ForceManager>(psf, prm);
      fm->setBoxDimensions({30., 30., 30.});
      fmList.push_back(fm);
    };

    // Create MBARForceManager
    std::vector<float> selectorVec(numberOfAlchemicalStates);
    std::fill(selectorVec.begin(), selectorVec.end(), 0.);
    selectorVec[6] = 1.0;
    auto mbarfm = std::make_shared<MBARForceManager>(fmList);
    mbarfm->setBoxDimensions({30., 30., 30.});
    mbarfm->initialize();

    auto crd = std::make_shared<CharmmCrd>(
        dataPath + "/" + std::to_string(molid) + ".solvated.crd");

    // First test: compare energies of MBARFM vs FM
    /////////////////////////////////////////////////////

    // Get energies of EACH single FM
    std::vector<double> singleEneList;
    for (int i = 0; i < numberOfAlchemicalStates; i++) {
      auto singlectx = std::make_shared<CharmmContext>(fmList[i]);
      singlectx->setCoordinates(crd);
      singlectx->calculatePotentialEnergy();
      auto singleenecc = singlectx->getPotentialEnergy();
      singleenecc.transferFromDevice();
      std::cout << "FM " << i << " energy: " << singleenecc[0] << std::endl;
      singleEneList.push_back(singleenecc[0]);
    }

    auto mbarctx = std::make_shared<CharmmContext>(mbarfm);
    auto ctx = std::make_shared<CharmmContext>(fmList[6]);
    ctx->setCoordinates(crd);

    mbarctx->setCoordinates(crd);
    // mbarctx->linkBackForceManager();
    mbarfm->setSelectorVec(selectorVec);

    ctx->calculatePotentialEnergy();
    mbarctx->calculatePotentialEnergy();

    auto mbarenecc = mbarctx->getPotentialEnergy();
    mbarenecc.transferFromDevice();
    double mbarene = mbarenecc[6];

    auto enecc = ctx->getPotentialEnergy();
    enecc.transferFromDevice();
    double ene = enecc[0];

    auto mbarchildcc =
        mbarctx->getForceManager()->getChildren()[6]->getPotentialEnergy();
    mbarchildcc.transferFromDevice();
    double mbarchild = mbarchildcc[0];

    REQUIRE(mbarene != 0.0);
    CHECK(mbarene == ene);

    // Test another energy (say number 10)
    selectorVec[6] = 0.0;
    selectorVec[10] = 1.0;
    mbarfm->setSelectorVec(selectorVec);
    mbarctx->calculatePotentialEnergy();
    mbarenecc = mbarctx->getPotentialEnergy();
    mbarenecc.transferFromDevice();
    mbarene = mbarenecc[10];
    REQUIRE(mbarene != 0.0);
    CHECK(mbarene == Approx(singleEneList[10]).epsilon(1e-9));

    // Check propagation with langevin piston
    auto integrator = CudaLangevinPistonIntegrator(0.002);
    integrator.setBathTemperature(310.0);
    integrator.setSimulationContext(mbarctx);
    integrator.setPistonFriction(12.0);
    integrator.setCrystalType(CRYSTAL::TETRAGONAL);

    CHECK_NOTHROW(integrator.propagate(1000));
  }
}
