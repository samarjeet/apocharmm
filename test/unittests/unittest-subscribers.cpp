// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, Félix Aviat
//
// ENDLICENSE

#include "ForceManager.h"
#include "ForceManagerGenerator.h"
#include "StateSubscriber.h"
#include "Subscriber.h"
#include "catch.hpp"
#include "compare.h"
// #include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "DcdSubscriber.h"
#include "DynaSubscriber.h"
#include "MBARSubscriber.h"
#include "RestartSubscriber.h"
#include "XYZSubscriber.h"
#include "helper.h"
#include "test_paths.h"
#include <iostream>
#include <stdio.h>
#include <unistd.h>

// I'd like to create test cases for the abstract "Subscriber" class,
// but I can't create an instance of it as it's abstract. So here's testing on
// its functions on a child class (StateSubscriber)
//
//
// Beware : flushing seem to be done upon file closing (or sthg like that).
// So it might be worth to do the testing after finishing everything ?

// print float4
void double4printer(double4 data) {
  std::cout << data.x << " " << data.y << " " << data.z << " " << data.w
            << std::endl;
}

// split line containing three numbers
double3 lineSplitter(std::string line) {
  double3 outCoord;
  std::stringstream ss(line);
  double x, y, z;
  ss >> x >> y >> z;
  outCoord.x = x;
  outCoord.y = y;
  outCoord.z = z;

  return outCoord;
}

// Get boxdim from restart file (for testing purposes)
std::vector<float> getBoxDimFromRestartFile(std::string fname) {
  std::ifstream f(fname);
  std::string line;
  while (std::getline(f, line)) {
    if (line.find("!BOXX") != std::string::npos) {
      break;
    }
  }
  std::getline(f, line);
  std::stringstream ss(line);
  float x, y, z;
  ss >> x >> y >> z;
  f.close();
  std::vector<float> boxDim = {x, y, z};
  return boxDim;
}

// Get single-line entry frm restart file (boxdim, On step piston velocity... )
std::vector<double> getRestartFileEntry(std::string fname,
                                        std::string entryName) {
  std::ifstream f(fname);
  std::string line, currentString;
  bool found = false;
  while (std::getline(f, line)) {
    if (line.find(entryName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    std::cout << "Entry " << entryName << " not found in " << fname
              << std::endl;
    throw std::invalid_argument("Entry not found in restart file.");
  }
  std::getline(f, line);
  std::stringstream ss(line);
  // Could be one, two or three values on the line
  std::vector<double> outVec;
  // Split line
  while (std::getline(ss, currentString, ' ')) {
    outVec.push_back(std::stod(currentString));
  }
  f.close();
  return outVec;
}

// Get multiple line entry from restart file (such as coords, velocity)
std::vector<std::vector<double>>
getRestartFileEntryMultiLine(std::string fname, std::string entryName) {
  std::ifstream f(fname);
  std::string line, currentString;
  bool found = false;
  while (std::getline(f, line)) {
    if (line.find(entryName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    std::cout << "Entry " << entryName << " not found in " << fname
              << std::endl;
    throw std::invalid_argument("Entry not found in restart file.");
  }
  std::vector<std::vector<double>> outVec;
  while (std::getline(f, line)) {
    // if line is empty or line is a newline, break
    if (line.empty() || line == "\n") {
      break;
    }
    std::stringstream ss(line);
    std::vector<double> currentVec;
    while (std::getline(ss, currentString, ' ')) {
      currentVec.push_back(std::stod(currentString));
    }
    outVec.push_back(currentVec);
  }
  f.close();
  return outVec;
}

TEST_CASE("Subscriber", "[unit]") {
  std::string dataPath = getDataPath();
  auto mysub = std::make_shared<StateSubscriber>("testout.state");
  auto mysub2 = std::make_shared<StateSubscriber>("testout2.state", 5);
  SECTION("Report frequency handling") {
    mysub->setReportFreq(12);
    // Check report frequencies are as set
    REQUIRE(mysub->getReportFreq() == 12);
    REQUIRE(mysub2->getReportFreq() == 5);
  }
  SECTION("Output file handling") {
    // Check proper files are created ?!
    std::ifstream ifile;
    ifile.open("testout.state");
    CHECK(ifile);
    ifile.close();
    ifile.open("notarealfile.state");
    CHECK_FALSE(ifile);

    // Check that wrong path yields error
    std::cout << "An error message should appear next line..." << std::endl;
    CHECK_THROWS(mysub->setFileName("non/existing/dir/file.out"));

    // Check that no path yields no error
    CHECK_NOTHROW(mysub->setFileName("file2.out"));

    // Check that fileName is as set
    REQUIRE(mysub->getFileName() == "file2.out");

    // Comment sections
    auto mysub3 = std::make_shared<StateSubscriber>("GenericSub.state");
    mysub3->addCommentSection("this is a string surrounded by line breaks\n");
  }
  SECTION("StateSubscriber", "[unit]") {
    // Check initialization of StateSubscriber output based on string
    std::string reportString = "potentialenergy, KineticEnergy, VOLUME";
    auto stateSub = std::make_shared<StateSubscriber>("StateSub.state");
    stateSub->readReportFlags(reportString);
    std::map<std::string, bool> rflags = stateSub->getReportFlags();
    CHECK(rflags["potentialenergy"] == true);
    CHECK(rflags["kineticenergy"] == true);
    CHECK(rflags["volume"] == true);
    reportString = "ALL";
    stateSub->readReportFlags(reportString);
    rflags = stateSub->getReportFlags();
    for (auto it = rflags.begin(); it != rflags.end(); it++) {
      CHECK(it->second == true);
    }
  }
  // Comment sections
  auto mysub3 = std::make_shared<StateSubscriber>("GenericSub.txt");
  mysub3->addCommentSection("this is a string surrounded by line breaks\n");
}

// Test basic outputs of simple subscibers through a basic dynamics run.
// Does not test the restart file (done in TEST_CASE("restart"))
TEST_CASE("basicDynamics", "[unit]") {
  // let's make a minimal simulation on which to test things
  std::string dataPath = getDataPath();
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions({50.0, 50.0, 50.0});
  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300);

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(.001);
  integrator->setPistonFriction(20.0);
  integrator->setCharmmContext(ctx);
  integrator->setCrystalType(CRYSTAL::CUBIC);
  std::vector<double> pistonmass = {500.0};
  integrator->setPistonMass(pistonmass);

  SECTION("StateSubscriber") {
    auto myStateSub = std::make_shared<StateSubscriber>("StateSub.state");
    myStateSub->setReportFreq(10);
    integrator->subscribe(myStateSub);
    integrator->propagate(100);
    // Check that there are TWO lines reported (reportFreq=10, nstep=20) + the
    // comment line Add comment section, propagate further, check line number
    myStateSub->addCommentSection("# This line as a comment");
    integrator->propagate(50);
    std::cout << lineCounter("StateSub.state");
    REQUIRE(lineCounter("StateSub.state") == 18);
  }

  // check that unsub/resub works ?
  SECTION("unsubResub") {
    auto mysub = std::make_shared<StateSubscriber>("tmpsub.txt", 1);
    integrator->subscribe(mysub);
    integrator->propagate(1);
    integrator->unsubscribe(mysub);
    CHECK(integrator->getReportFreqList().size() == 0);
    mysub = std::make_shared<StateSubscriber>("tmpsub2.state", 1);
    integrator->subscribe(mysub);
    CHECK_NOTHROW(integrator->propagate(2));
  }

  SECTION("DcdSubscriber") {
    int reportFrequency = 10;
    auto myDcdSub =
        std::make_shared<DcdSubscriber>("DcdSub.dcd", reportFrequency);
    integrator->subscribe(myDcdSub);
    CHECK_NOTHROW(integrator->propagate(1000));

    // Read the content of the binary file "DcdSub.dcd"
    std::ifstream dcdFile("DcdSub.dcd", std::ios::binary);
    REQUIRE(dcdFile);
    // Check the header
    int size;
    dcdFile.read(reinterpret_cast<char *>(&size), sizeof(int));
    REQUIRE(size == 84);
    char header[84];
    dcdFile.read(header, size);
    REQUIRE(header[0] == 'C');
    REQUIRE(header[1] == 'O');
    REQUIRE(header[2] == 'R');
    REQUIRE(header[3] == 'D');

    int readReportFrequency = (int)header[12];
    REQUIRE(readReportFrequency == reportFrequency);

    int end_size;
    dcdFile.read(reinterpret_cast<char *>(&end_size), sizeof(int));
    REQUIRE(end_size == size);

    dcdFile.read(reinterpret_cast<char *>(&size), sizeof(int));
    REQUIRE(size == 164);
    size = 164;
    int numTitleLines; // = 2
    dcdFile.read(reinterpret_cast<char *>(&numTitleLines), sizeof(int));
    REQUIRE(numTitleLines == 2);
    char header2[160];

    // dcdFile.read(reinterpret_cast<char *>(&end_size), sizeof(int));
    //  REQUIRE(end_size == size);
  }

  SECTION("Multiple subscribers") {
    auto sub1 = std::make_shared<StateSubscriber>("sub1.state");
    auto sub2 = std::make_shared<StateSubscriber>("sub2.state");
    std::vector<std::shared_ptr<Subscriber>> sublist{sub1, sub2};
    sub1->setReportFreq(23);
    sub2->setReportFreq(46);
    integrator->subscribe(sublist);
    integrator->propagate(100);

    REQUIRE(lineCounter("sub1.state") == 6);
    REQUIRE(lineCounter("sub2.state") == 4);
  }

  SECTION("XYZSubscriber") {
    auto myXYZSub = std::make_shared<XYZSubscriber>("xyz.out", 10);
    integrator->subscribe(myXYZSub);
    integrator->propagate(30);
    // Check the content of "xyz.out"
    REQUIRE(lineCounter("xyz.out") == ctx->getNumAtoms() * 30 / 10);
  }

  SECTION("PressureInStateSub") {
    auto myStateSub =
        std::make_shared<StateSubscriber>("StateSub4pressure.state", 2);
    std::string myFlags = "pressurescalar, pressurecomponents";
    myStateSub->readReportFlags(myFlags);
    integrator->subscribe(myStateSub);
    integrator->propagate(20);
    // REQUIRE(lineCounter("StateSub4pressure.state") == 12);
  }

  SECTION("DynaSubscriber") {
    auto myDynaSub = std::make_shared<DynaSubscriber>("dyna.out", 1);
    integrator->setDebugPrintFrequency(1);
    integrator->subscribe(myDynaSub);
    integrator->propagate(4);
    // Check the content of "dyna.out"
  }
}

TEST_CASE("MBARSubscriber") {
  std::string dataPath = getDataPath();
  std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};
  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  std::vector<double> boxDim = {50.0, 50.0, 50.0};
  fm->setBoxDimensions(boxDim);
  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300);

  SECTION("save") {
    auto integrator =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002, 300., 12.);
    integrator->setCharmmContext(ctx);
    auto myRestartSubWrite =
        std::make_shared<RestartSubscriber>("restartWaterThermostat.res", 10);
    integrator->subscribe(myRestartSubWrite);
    integrator->propagate(10);
    integrator->unsubscribe(myRestartSubWrite); // should close file

    // Save values as references to compare later them with the subscriber
    auto ctxBoxDim = ctx->getBoxDimensions();
    CudaContainer<double4> coords = ctx->getCoordinatesCharges(),
                           vel = ctx->getVelocityMass(),
                           coordsDeltaPrevious =
                               integrator->getCoordsDeltaPrevious();
    coords.transferFromDevice();
    vel.transferFromDevice();
    coordsDeltaPrevious.transferFromDevice();

    auto myRestartSub =
        std::make_shared<RestartSubscriber>("restartWaterThermostat.res", 30);
    auto restartFileBoxDim = myRestartSub->readBoxDimensions();
    CHECK(compareVectors(restartFileBoxDim, ctxBoxDim));
    auto restartFilePositions = myRestartSub->readPositions();
    auto restartFileVelocities = myRestartSub->readVelocities();
    auto restartFileCoordsDeltaPrevious =
        myRestartSub->readCoordsDeltaPrevious();
    // Using tolerance of 1e-4 because of the precision of the restart file
    CHECK(CompareVectors(restartFilePositions, coords.getHostArray(), 1e-4));
    CHECK(CompareVectors(restartFileVelocities, vel.getHostArray(), 1e-5));
    CHECK(CompareVectors(restartFileCoordsDeltaPrevious,
                         coordsDeltaPrevious.getHostArray(), 1e-6));
  }

  SECTION("readWrapped") {
    auto readRestartSub =
        std::make_shared<RestartSubscriber>("restartWaterThermostat.res", 5000);

    auto integrator =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002, 300., 12.);
    integrator->setCharmmContext(ctx);

    integrator->subscribe(readRestartSub);

    CHECK_NOTHROW(readRestartSub->readRestart());
    // Assert the integrator AND context have the right values distributes. Only
    // checks first value for pos, vel, cdprev .
    //  Context: boxdim, coords,
    // velocities, Integrator: coordsdeltaprevious
    auto boxDimRef = getRestartFileEntry("restartWaterThermostat.res", "!BOXX");
    auto coordRestartFile =
        getRestartFileEntryMultiLine("restartWaterThermostat.res", "!X,");
    auto velRestartFile =
        getRestartFileEntryMultiLine("restartWaterThermostat.res", "!VX,");
    auto coordsDeltaPreviousRestartFile = getRestartFileEntryMultiLine(
        "restartWaterThermostat.res", "!coordsDeltaPrevious");

    auto poschrg = ctx->getCoordinatesCharges();
    poschrg.transferFromDevice();
    auto velmass = ctx->getVelocityMass();
    velmass.transferFromDevice();
    auto coordsDeltaPrevious = integrator->getCoordsDeltaPrevious();
    coordsDeltaPrevious.transferFromDevice();

    CHECK(CompareVectorsPBC(coordRestartFile, poschrg.getHostArray(),
                            ctx->getBoxDimensions(), 1e-4));
    CHECK(CompareVectors(velRestartFile, velmass.getHostArray()));
    CHECK(CompareVectors(coordsDeltaPreviousRestartFile,
                         coordsDeltaPrevious.getHostArray()));
  }
}

// Debugging unittest. Remove before merge once ok.
TEST_CASE("debugpy") {
  int nsteps = 100;
  int savefreq = 50;
  std::string dataPath = getDataPath();

  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions({50., 50., 50.});

  // COMMENT OUT P21 LINE AND IT RUNS ! Otherwise NaN KE error.
  fm->setPeriodicBoundaryCondition(PBC::P21);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox_p21_min.crd");
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300.0);

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  integrator->setPistonFriction(10.0);
  integrator->setCharmmContext(ctx);
  integrator->setCrystalType(CRYSTAL::TETRAGONAL);
  integrator->setPistonMass({500.0, 500.0});

  auto restartSub = std::make_shared<RestartSubscriber>("restart.out", 100);
  integrator->subscribe(restartSub);
  // restartSub->readRestart();
  integrator->propagate(200);
}

/*
// Test MBAR subscriber on the methane part of the methane->toluene
// transformation as generated by S. Boresch's transformato
SECTION("MethaneCCSAI") {
  // transformato generated 3 intermediate states to switch from methane to
  // CH3-(Dummy)
  std::string stateDir, ccsaiDataPath = dataPath + "methaneCCSAI/";
  std::vector<std::shared_ptr<ForceManager>> fmlist;
  std::vector<std::string> prmlist = {dataPath + "toppar_water_ions.str",
                                      dataPath + "par_all36_cgenff.prm",
                                      "tobereplaced"};

    for (int i = 1; i < 4; i++) {
      stateDir = ccsaiDataPath + "/intst" + std::to_string(i) + "/";
      auto psf = std::make_shared<CharmmPSF>(stateDir + "methane_aq.psf");
      prmlist[2] = stateDir + "dummy_parameters.prm";
      auto prm = std::make_shared<CharmmParameters>(prmlist);
      auto stateFM = std::make_shared<ForceManager>(psf, prm);
      stateFM->setBoxDimensions({30., 30., 30.});
      fmlist.push_back(stateFM);
    }
    std::shared_ptr<MBARForceManager> mbarfm =
        std::make_shared<MBARForceManager>(fmlist);
    std::vector<float> oneHotVector = {0., 1., 0.};
    // Pb : only the one-hot encoded is printed out...

    // THESE ARE MBARFM TESTS !
    // TODO : move to MBAR unittest / FM unittest / somewhere like that
    // Should throw if we havent set a selectorVec
    CHECK_THROWS(mbarfm->initialize());
    // Should throw if we give a bad selector vector
    CHECK_THROWS(mbarfm->setSelectorVec({0, 1, 2}));
    // Should not for a good one...
    CHECK_NOTHROW(mbarfm->setSelectorVec(oneHotVector));
    //////////////////////////////////

    auto ctx = std::make_shared<CharmmContext>(mbarfm);
    auto crd = std::make_shared<CharmmCrd>(stateDir + "methane_aq.crd");
    ctx->setCoordinates(crd);
    ctx->assignVelocitiesAtTemperature(300.);

    auto integrator =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.001, 300., 5.);
    integrator->setCharmmContext(ctx);

    auto mbarsub = std::make_shared<MBARSubscriber>("mbarsub.out", 10);
    integrator->subscribe(mbarsub);

    integrator->propagate(100);
    CHECK(lineCounter("mbarsub.out") == 10);
    std::string line;
    std::ifstream outputMbarFile("mbarsub.out");
    std::getline(outputMbarFile, line);
    double3 energies = lineSplitter(line);
    REQUIRE(energies.x != 0.0);
    REQUIRE(energies.y != 0.0);
    REQUIRE(energies.z != 0.0);
  }
}
*/
