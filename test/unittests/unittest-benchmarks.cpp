// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaMinimizer.h"
#include "CudaVelocityVerletIntegrator.h"
#include "DcdSubscriber.h"
#include "PDB.h"
#include "RestartSubscriber.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

std::size_t InitRandCoordsChargesNeut(
    std::vector<double> &coordsX, std::vector<double> &coordsY,
    std::vector<double> &coordsZ, std::vector<double> &charges,
    const double boxDimX, const double boxDimY, const double boxDimZ,
    const std::size_t numAtomsPerDimX, const std::size_t numAtomsPerDimY,
    const std::size_t numAtomsPerDimZ, const std::size_t seed) {
  const std::size_t numAtoms =
      numAtomsPerDimX * numAtomsPerDimY * numAtomsPerDimZ;
  const double gapX = boxDimX / static_cast<double>(numAtomsPerDimX);
  const double gapY = boxDimY / static_cast<double>(numAtomsPerDimY);
  const double gapZ = boxDimZ / static_cast<double>(numAtomsPerDimZ);

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::random_device rd;
  std::mt19937 rng((seed == 0) ? rd() : seed);
  double averageCharge = 0.0;

  coordsX.clear();
  coordsY.clear();
  coordsZ.clear();
  charges.clear();

  // Take the regular grid of positions and add a small random shift
  for (std::size_t ix = 0; ix < numAtomsPerDimX; ix++) {
    double x = (static_cast<double>(ix) + 0.5) * gapX + 2.0 * dist(rng) - 1.0;
    x = (x >= boxDimX) ? boxDimX - 1e-5 : x;
    x = (x <= 0.0) ? 1e-5 : x;
    for (std::size_t iy = 0; iy < numAtomsPerDimY; iy++) {
      double y = (static_cast<double>(iy) + 0.5) * gapY + 2.0 * dist(rng) - 1.0;
      y = (y >= boxDimY) ? boxDimY - 1e-5 : y;
      y = (y <= 0.0) ? 1e-5 : y;
      for (std::size_t iz = 0; iz < numAtomsPerDimZ; iz++) {
        double z =
            (static_cast<double>(iz) + 0.5) * gapZ + 2.0 * dist(rng) - 1.0;
        z = (z >= boxDimZ) ? boxDimZ - 1e-5 : z;
        z = (z <= 0.0) ? 1e-5 : z;

        double q = 2.0 * dist(rng);

        coordsX.push_back(x);
        coordsY.push_back(y);
        coordsZ.push_back(z);
        charges.push_back(q);
        averageCharge += q;
      }
    }
  }

  // Ensure that the total charge of the system is neutral
  averageCharge /= static_cast<double>(numAtoms);
  for (double &charge : charges)
    charge -= averageCharge;

  return numAtoms;
}

TEST_CASE("waterbox") {
  std::string dataPath = getDataPath();

  std::vector<std::string> prmFiles = {dataPath + "toppar_water_ions.str"};
  std::vector<double> boxDim = {50.0, 50.0, 50.0};

  // Parameters, PSF, and coordinates
  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

  // Force manager
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions(boxDim);

  // CHARMM context
  auto ctx = std::make_shared<CharmmContext>(fm);
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);

  ctx->calculatePotentialEnergy(true, true);

  // std::cout << "No. of Atoms: " << ctx->getNumAtoms() << "\n";

  // const std::vector<double> boxDim = {256.0, 256.0, 256.0};
  // const std::vector<std::size_t> numAtomsPerDim = {256, 256, 256};

  // std::vector<double> x, y, z, q;
  // InitRandCoordsChargesNeut(x, y, z, q, boxDim[0], boxDim[1], boxDim[2],
  //                           numAtomsPerDim[0], numAtomsPerDim[1],
  //                           numAtomsPerDim[2], 314159);
}

/*
TEST_CASE("pore", "[dynamics]") {
  std::string dataPath = "/u/samar/toppar/";
  std::string filePath = "/u/samar/Documents/git/test_gpu/jmin_pore/";
  SECTION("20p") {
    std::vector<std::string> prmFiles{dataPath + "par_all36_lipid.prm",
                                      dataPath +
                                          "toppar_all36_lipid_bacterial.str",
                                      dataPath + "toppar_water_ions.str"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto psf = std::make_shared<CharmmPSF>(filePath + "jmin_pore_20p_min.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({160.0, 160.0, 100.0});
    fm->setFFTGrid(160, 160, 100);
    fm->setKappa(0.34);
    fm->setCutoff(9.0);
    fm->setCtonnb(8.0);
    fm->setCtofnb(8.5);

    // fm->setPrintEnergyDecomposition(true);

    auto ctx = std::make_shared<CharmmContext>(fm);
    // auto crd = std::make_shared<CharmmCrd>(filePath + "equil.cor");
    auto crd = std::make_shared<CharmmCrd>(filePath + "jmin_pore_20p_min.crd");

    ctx->setCoordinates(crd);
    ctx->calculatePotentialEnergy(true, true);

    std::cout << "DOFs : " << ctx->getDegreesOfFreedom() << "\n";
    ctx->assignVelocitiesAtTemperature(300);
    auto pe = ctx->getPotentialEnergy();
    pe.transferFromDevice();

    CudaMinimizer minimizer;
    minimizer.setCharmmContext(ctx);
    // minimizer.minimize(100);

    ctx->calculatePotentialEnergy(true, true);
    pe = ctx->getPotentialEnergy();
    pe.transferFromDevice();

    std::cout << "PE : " << pe[0] << "\n";

    fm->setPrintEnergyDecomposition(false);

    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    langevinThermostat->setFriction(5.0);
    langevinThermostat->setBathTemperature(300.0);
    langevinThermostat->setCharmmContext(ctx);
    // langevinThermostat->setDebugPrintFrequency(100);

    //auto nvtdcdSubscriber =
    //    std::make_shared<DcdSubscriber>("jmin_nvt_pore.dcd", 1000);
    //langevinThermostat.subscribe(nvtdcdSubscriber);

    langevinThermostat->propagate(
        1e5); // You can propagate longer when you assert something !

    auto equilibrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    equilibrator->setPistonFriction(12.0);
    equilibrator->setCharmmContext(ctx);
    equilibrator->setNoseHooverPistonMass(1000.0);
    equilibrator->setCrystalType(CRYSTAL::TETRAGONAL);
    equilibrator->setPistonMass({1000.0, 1000.0});
    // equilibrator->setDebugPrintFrequency(100);
    //  equilibrator.propagate(1e4);

    ctx->useHolonomicConstraints(false);

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.001);
    integrator->setPistonFriction(0.0);
    integrator->setCharmmContext(ctx);
    // integrator->setNoseHooverPistonMass(1000.0);
    integrator->setCrystalType(CRYSTAL::TETRAGONAL);
    // integrator->setPistonMass({std::numeric_limits<double>::max(), 1000.0});
    integrator->setPistonMass({0.0, 0.0});
    integrator->setNoseHooverFlag(false);

    // integrator->setDebugPrintFrequency(1000);

    // auto dcdSubscriber =
    //     std::make_shared<DcdSubscriber>("jmin_npat_pore.dcd", 1000);

    //auto restartSubscriber =
    //    std::make_shared<RestartSubscriber>("jmin_npat_pore.restart", 10000);
    //integrator.subscribe({dcdSubscriber, restartSubscriber});

    // auto stateSubscriber =
    //    std::make_shared<StateSubscriber>("jmin_npat_pore.txt", 1000);
    // std::vector<Subscriber> subscribers{dcdSubscriber, stateSubscriber};
    // integrator.subscribe(dcdSubscriber);

    std::cout << "Starting simulation\n";
    // integrator.propagate(1e7);
  }
}
*/

TEST_CASE("amber_benchmark", "[dynamics]") {
  std::string dataPath = getDataPath();
  /*SECTION("JAC") {
    std::vector<std::string> prmFiles{dataPath + "par_all36m_prot.prm",
                                      dataPath + "toppar_water_ions.str"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto psf =
        std::make_shared<CharmmPSF>("/u/samar/Documents/git/test_gpu/JAC.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({61.64472316, 61.64472316, 61.64472316});
    fm->setFFTGrid(64, 64, 64);
    fm->setKappa(0.34);
    fm->setCutoff(9.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd =
        std::make_shared<CharmmCrd>("/u/samar/Documents/git/test_gpu/JAC.crd");
    ctx->setCoordinates(crd);
    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);
  }
  */
}

TEST_CASE("blade_benchmark", "[dynamics]") {
  SECTION("dhfr") {
    std::string dataPath = "/u/samar/packages/BLaDE/test/dhfr/";
    std::vector<std::string> prmFiles{dataPath + "par_all22_prot.inp"};

    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto psf = std::make_shared<CharmmPSF>(dataPath + "5dfr.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({62.23, 62.23, 62.23});
    fm->setFFTGrid(64, 64, 64);
    fm->setKappa(0.34);
    fm->setCutoff(9.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(7.5);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "5dfr.crd");
    ctx->setCoordinates(crd);
    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);

    double timeStep = 0.002;
    double numSteps = 100;

    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    // CudaVelocityVerletIntegrator langevinThermostat(0.002);

    langevinThermostat->setFriction(0.0);
    langevinThermostat->setBathTemperature(300.0);
    langevinThermostat->setCharmmContext(ctx);

    auto start = std::chrono::high_resolution_clock::now();

    langevinThermostat->propagate(numSteps);
    auto end = std::chrono::high_resolution_clock::now();

    // time in seconds
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "No. of Atoms: " << ctx->getNumAtoms() << "\n";
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << "speed in ns/day : "
              << (numSteps * timeStep) / (elapsed_seconds.count() * 1e3) * 86400
              << "\n";
    std::cout << std::endl;

    // time for 10,000 energy calls
    /*start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
      ctx->calculatePotentialEnergy(false, false);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    */
  }

  SECTION("dmpg") {
    std::string dataPath = "/u/samar/packages/BLaDE/test/dmpg/";
    std::vector<std::string> prmFiles{
        dataPath + "toppar/par_all36_lipid.prm",
        dataPath + "toppar/ions-noskov+roux-X=Onbfix.prm"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto psf = std::make_shared<CharmmPSF>(dataPath + "dmpg290k.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({140.1228, 186.8304, 114.109});
    fm->setFFTGrid(144, 192, 108);
    fm->setKappa(0.34);
    fm->setCutoff(9.5);
    fm->setCtonnb(7.5);
    fm->setCtofnb(9.0);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "dmpg290k.crd");
    ctx->setCoordinates(crd);
    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);

    double timeStep = 0.002;
    double numSteps = 1000;

    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    // CudaVelocityVerletIntegrator langevinThermostat(0.002);
    langevinThermostat->setFriction(0.0);
    langevinThermostat->setBathTemperature(300.0);
    langevinThermostat->setCharmmContext(ctx);

    auto start = std::chrono::high_resolution_clock::now();

    langevinThermostat->propagate(numSteps);
    auto end = std::chrono::high_resolution_clock::now();

    // time in seconds
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "No. of Atoms: " << ctx->getNumAtoms() << "\n";
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << "speed in nd/day : "
              << (numSteps * timeStep) / (elapsed_seconds.count() * 1e3) * 86400
              << "\n";
    std::cout << std::endl;
  }
}

TEST_CASE("microtubule", "[dyna]") {
  std::string dataPath = getDataPath();
  SECTION("monomer") {
    std::string filePath = "/u/samar/projects/benchmark/";
    auto psf0 = std::make_shared<CharmmPSF>(filePath + "3ryfi.psf");

    std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str",
                                      dataPath + "par_all36_prot.prm",
                                      dataPath + "par_all36_na.prm",
                                      dataPath + "toppar_all36_na_nad_ppi.str"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto fm0 = std::make_shared<ForceManager>(psf0, prm);

    double boxLengthX = 365.313493;
    double boxLengthYZ = 142.975477;
    fm0->setBoxDimensions({boxLengthX, boxLengthYZ, boxLengthYZ});
    fm0->setFFTGrid(384, 144, 144);
    fm0->setPmeSplineOrder(4);
    fm0->setKappa(0.34);
    fm0->setCutoff(13.0);
    fm0->setCtonnb(10.0);
    fm0->setCtofnb(12.0);

    auto ctx = std::make_shared<CharmmContext>(fm0);
    auto crd = std::make_shared<CharmmCrd>(filePath + "3ryfi.cor");
    ctx->setCoordinates(crd);

    ctx->calculateForces(false, true, true);
    auto forces = ctx->getForces();

    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);

    double timeStep = 0.002;
    double numSteps = 1000;

    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    // CudaVelocityVerletIntegrator langevinThermostat(0.002);
    langevinThermostat->setFriction(0.0);
    langevinThermostat->setBathTemperature(300.0);
    langevinThermostat->setCharmmContext(ctx);

    auto start = std::chrono::high_resolution_clock::now();

    langevinThermostat->propagate(numSteps);
    auto end = std::chrono::high_resolution_clock::now();

    // time in seconds
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "No. of Atoms: " << ctx->getNumAtoms() << "\n";
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << "speed in nd/day : "
              << (numSteps * timeStep) / (elapsed_seconds.count() * 1e3) * 86400
              << "\n";
    std::cout << std::endl;
  }
}

TEST_CASE("namd_benchmark", "[dynamics]") {
  SECTION("stmv") {
    std::string dataPath = "/u/samar/projects/benchmark/stmv/";
    std::vector<std::string> prmFiles{dataPath + "par_all27_prot_na.inp"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto psf = std::make_shared<CharmmPSF>(dataPath + "stmv.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({216.832, 216.832, 216.832});
    fm->setFFTGrid(216, 216, 216);
    fm->setKappa(0.34);
    fm->setCutoff(8.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(7.5);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<PDB>(dataPath + "stmv.pdb");
    ctx->setCoordinates(crd);
    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);

    /*CudaMinimizer minimizer;
    minimizer.setCharmmContext(ctx);
    // minimizer.minimize();

    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    // CudaVelocityVerletIntegrator langevinThermostat(0.002);
    langevinThermostat->setFriction(0.0);
    langevinThermostat->setBathTemperature(300.0);
    langevinThermostat->setCharmmContext(ctx);

    langevinThermostat->propagate(21000);
    */
    double timeStep = 0.002;
    double numSteps = 1000;

    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    // CudaVelocityVerletIntegrator langevinThermostat(0.002);
    langevinThermostat->setFriction(0.0);
    langevinThermostat->setBathTemperature(300.0);
    langevinThermostat->setCharmmContext(ctx);

    auto start = std::chrono::high_resolution_clock::now();

    langevinThermostat->propagate(numSteps);
    auto end = std::chrono::high_resolution_clock::now();

    // time in seconds
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "No. of Atoms: " << ctx->getNumAtoms() << "\n";
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << "speed in nd/day : "
              << (numSteps * timeStep) / (elapsed_seconds.count() * 1e3) * 86400
              << "\n";
    std::cout << std::endl;
  }

  SECTION("apoa1") {
    std::string dataPath = "/u/samar/projects/benchmark/apoa1/";
    std::string dataPath1 = getDataPath();

    std::vector<std::string> prmFiles{
        dataPath1 + "par_all36_lipid.prm", dataPath1 + "par_all22_prot.prm",
        dataPath + "par_all22_popc.xplor"
        // dataPath + "par_all22_prot_lipid.xplor"
    };

    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto psf = std::make_shared<CharmmPSF>(dataPath + "../apoa11.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({108.8612, 108.8612, 77.758});
    fm->setFFTGrid(96, 96, 96);
    fm->setKappa(0.34);
    fm->setCutoff(8.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(7.5);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<PDB>(dataPath + "../apoa11.pdb");
    ctx->setCoordinates(crd);
    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);

    double timeStep = 0.002;
    double numSteps = 1000;

    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    // CudaVelocityVerletIntegrator langevinThermostat(0.002);
    langevinThermostat->setFriction(0.0);
    langevinThermostat->setBathTemperature(300.0);
    langevinThermostat->setCharmmContext(ctx);

    auto start = std::chrono::high_resolution_clock::now();

    langevinThermostat->propagate(numSteps);
    auto end = std::chrono::high_resolution_clock::now();

    // time in seconds
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "No. of Atoms: " << ctx->getNumAtoms() << "\n";
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << "speed in nd/day : "
              << (numSteps * timeStep) / (elapsed_seconds.count() * 1e3) * 86400
              << "\n";
    std::cout << std::endl;
  }
  SECTION("apoa1_new") {
    std::string dataPath = "/u/samar/projects/benchmark/apoa1_new/";
    std::string dataPath1 = getDataPath();

    std::vector<std::string> prmFiles{
        dataPath + "toppar/par_all36_lipid.prm",
        dataPath + "toppar/par_all36m_prot.prm",
        dataPath + "toppar/toppar_water_ions.str"
        // dataPath + "par_all22_prot_lipid.xplor"
    };

    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto psf = std::make_shared<CharmmPSF>(dataPath + "apoa1_new.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({108.8612, 108.8612, 77.758});
    fm->setFFTGrid(96, 96, 96);
    fm->setKappa(0.34);
    fm->setCutoff(9.5);
    fm->setCtonnb(7.5);
    fm->setCtofnb(9.0);

    // fm->setPrintEnergyDecomposition(true);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "apoa1.crd");
    // auto crd = std::make_shared<PDB>(dataPath + "apoa1_new.pdb");
    ctx->setCoordinates(crd);
    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);
    auto coords = ctx->getCoordinates();
    // std::cout << coords[0][0] << "  " << coords[0][1] << "  " << coords[0][2]
    //           << "\n";

    double timeStep = 0.002;
    double numSteps = 10000;

    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    // CudaVelocityVerletIntegrator langevinThermostat(0.002);
    langevinThermostat->setFriction(0.0);
    langevinThermostat->setBathTemperature(300.0);
    langevinThermostat->setCharmmContext(ctx);

    auto start = std::chrono::high_resolution_clock::now();

    langevinThermostat->propagate(numSteps);
    auto end = std::chrono::high_resolution_clock::now();

    // time in seconds
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "No. of Atoms: " << ctx->getNumAtoms() << "\n";
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << "speed in ns/day : "
              << (numSteps * timeStep) / (elapsed_seconds.count() * 1e3) * 86400
              << "\n";
    std::cout << std::endl;
  }
}

/*
TEST_CASE("benchmark", "[dynamics]") {
  std::string prmPath = "/u/zjarin/apoe/toppar/";
  SECTION("beta") {
    std::vector<std::string> prmFiles{
        prmPath + "par_all36m_prot.prm",
        prmPath + "par_all36_na.prm",
        prmPath + "par_all36_carb.prm",
        prmPath + "par_all36_lipid.prm",
        prmPath + "par_all36_cgenff.prm",
        prmPath + "toppar_all36 _prot_model.str",
        prmPath + "toppar_all36_prot_modify_res.str",
        prmPath + "toppar_all36_lipid_cholesterol.str",
        prmPath + "toppar_all36_lipid_tag_vanni.str",
        prmPath + "toppar_water_ions.str"};

    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto psf = std::make_shared<CharmmPSF>(
        "/u/zjarin/for_samar/beta_sheet/fullsystem.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({146.586, 146.586, 144.101});
    // fm->setFFTGrid(48, 48, 48);
    fm->setFFTGrid(128, 128, 128);
    fm->setKappa(0.32);
    fm->setCutoff(14.0);
    fm->setCtonnb(12.0);
    fm->setCtofnb(8.0);

    // fm->setPrintEnergyDecomposition(true);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd =
        std::make_shared<PDB>("/u/zjarin/for_samar/beta_sheet/input.pdb");
    ctx->setCoordinates(crd);
    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(310);
    // std::cout << "KE : " << ctx->calculateKineticEnergyOld() << "\n";
    // int inp;
    // std::cin >> inp;
    CudaMinimizer minimizer;
    minimizer.setCharmmContext(ctx);
    // minimizer.minimize(100);

    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    // CudaVelocityVerletIntegrator langevinThermostat(0.002);
    langevinThermostat->setFriction(5.0);
    langevinThermostat->setBathTemperature(310.0);
    langevinThermostat->setCharmmContext(ctx);

    langevinThermostat->propagate(5000);

    // integrator.propagate(50000);

    auto subscriber = std::make_shared<DcdSubscriber>("vv_walp.dcd", ctx);
    ctx->subscribe(subscriber);
    auto stateSubscriber =
        std::make_shared<StateSubscriber>("vv_walp.txt", ctx);
    ctx->subscribe(stateSubscriber);

    integrator.setReportSteps(5000);
    integrator.propagate(1000000);
  }
}
*/
