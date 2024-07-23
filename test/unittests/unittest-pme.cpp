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
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "ForceManager.h"
#include "RestartSubscriber.h"
#include "catch.hpp"
#include "compare.h"
#include "helper.h"
#include "test_paths.h"
#include <iostream>

TEST_CASE("jiyeon") {
  std::string dataPath =
      "/u/jmin/Documents/project_pore/PCPG/run_apo/200_20p_e5e6_makeerror/";

  std::string filePath =
      "/u/jmin/Documents/project_pore/PCPG/last_frames/400_20p_namd_3/run_apo/";

  std::string paramPath = "/u/jmin/toppars/toppar_c36_jul22/";
  std::vector<std::string> prmFiles{
      paramPath + "par_all36m_prot.prm", paramPath + "par_all36_lipid.prm",
      paramPath + "stream/lipid/toppar_all36_lipid_bacterial.str",
      paramPath + "toppar_water_ions.str"};

  auto prm = std::make_shared<CharmmParameters>(prmFiles);

  auto psf = std::make_shared<CharmmPSF>(filePath + "04.psf");
  auto crd = std::make_shared<CharmmCrd>(filePath + "04.crd");

  auto fm = std::make_shared<ForceManager>(psf, prm);

  std::vector<double> boxDim1 = {50.0, 50.0, 50.0};
  //  std::vector<double> boxDim = {170.0, 170.0, 120.0};
  //  std::vector<double> boxDim = {112.931997, 112.931997, 93.500177};
  // std::vector<double> boxDim1 = {162.4579, 158.83378, 109.55576};
  std::vector<double> boxDim2 = {163.4579, 159.83378, 110.55576};

  auto boxDim = boxDim1;
  fm->setBoxDimensions(boxDim);
  fm->setCtonnb(8.0);
  fm->setCtofnb(10.0);
  fm->setCutoff(12.0);
  fm->setFFTGrid(128, 128, 128);
  fm->setPrintEnergyDecomposition(true);

  auto ctx = std::make_shared<CharmmContext>(fm);
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(310.15);

  ctx->calculatePotentialEnergy();

  boxDim = boxDim2;
  ctx->setBoxDimensions(boxDim);
  ctx->resetNeighborList();
  ctx->calculatePotentialEnergy();

  // second context
  auto fm2 = std::make_shared<ForceManager>(psf, prm);
  fm2->setBoxDimensions(boxDim);
  fm2->setCtonnb(8.0);
  fm2->setCtofnb(10.0);
  fm2->setCutoff(12.0);
  fm2->setFFTGrid(128, 128, 128);
  fm2->setPrintEnergyDecomposition(true);

  auto ctx2 = std::make_shared<CharmmContext>(fm2);
  ctx2->setCoordinates(crd);
  ctx2->assignVelocitiesAtTemperature(310.15);

  ctx2->calculatePotentialEnergy();

  // boxDim = {50.0, 50.0, 50.0};
  boxDim = boxDim1;
  ctx2->setBoxDimensions(boxDim);
  ctx2->resetNeighborList();

  ctx2->calculatePotentialEnergy();
}
