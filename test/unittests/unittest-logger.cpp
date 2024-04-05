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
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "ForceManager.h"
#include "Logger.h"
#include "catch.hpp"
#include "test_paths.h"

TEST_CASE("logger", "[unit]") {
  std::string dataPath = getDataPath();
  // Test basic functions
  // Setup a basic waterbox for that purpose
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions({50., 50., 50.});
  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300.0);

  auto integrator = std::make_shared<CudaVelocityVerletIntegrator>(0.001);
  integrator->setCharmmContext(ctx);

  integrator->propagate(100);

  // 1. Check that the logger default name matches what we expect
  // CHECK(false);
}
