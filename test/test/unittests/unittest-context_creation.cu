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
#include "ForceManager.h"
#include "catch.hpp"
#include <iostream>
#include <memory>

TEST_CASE("context_creation", "[energy]") {
  /*
  SECTION("water2") {

    std::unique_ptr<CharmmParameters> prm =
        std::make_unique<CharmmParameters>("../test/data/par_all36_prot.prm");
    prm->readCharmmParameterFile("../test/data/toppar_water_ions.str");
    std::unique_ptr<CharmmPSF> psf =
        std::make_unique<CharmmPSF>("../test/data/water2.psf");

    //auto fm1 = ForceManager(psf, prm); // why does this not work ??
    auto fm = std::make_unique<ForceManager>(psf, prm);
    fm->setBoxDimensions({62.3, 62.3, 62.3});
    fm->setKappa(0.34);
    fm->setCutoff(9.0);
    fm->setPmeSplineOrder(4);
    fm->initialize();
    REQUIRE(psf == nullptr);
    REQUIRE(prm == nullptr);

    auto charmmContext = std::make_unique<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>("../test/data/water2.crd");
    charmmContext->setCoordinates(crd);

    charmmContext->calculatePotentialEnergy(true);

    // integrator
    // auto integrator = VelocityVerletIntegrator(300);
    // integrator.setContext(CharmmContext);
    // integrator.propagate(100);

  }
  */
  SECTION("dhfr_only") {

    auto prm = std::make_shared<CharmmParameters>("../test/data/par_all36_prot.prm");
    prm->readCharmmParameterFile("../test/data/toppar_water_ions.str");
    std::unique_ptr<CharmmPSF> psf =
        //std::make_unique<CharmmPSF>("../test/data/dhfr_only.psf");
        //std::make_unique<CharmmPSF>("../test/data/dhfr.psf");
        std::make_unique<CharmmPSF>("../test/data/dhfr_neutral.psf");

    //auto fm1 = ForceManager(psf, prm); // why does this not work ??
    auto fm = std::make_shared<ForceManager>(psf, prm);
    //fm->setBoxDimensions({62.3, 62.3, 62.3});
    fm->setBoxDimensions({100.0, 100.0, 100.0});
    fm->setKappa(0.34);
    fm->setCutoff(12.0);
    fm->setPmeSplineOrder(4);
    fm->initialize();
    REQUIRE(psf == nullptr);
    REQUIRE(prm == nullptr);

    auto charmmContext = std::make_unique<CharmmContext>(fm);
    //auto crd = std::make_unique<CharmmCrd>("../test/data/dhfr_only.crd");
    //auto crd = std::make_unique<CharmmCrd>("../test/data/dhfr.crd");
    auto crd = std::make_shared<CharmmCrd>("../test/data/dhfr_neutral.crd");
    charmmContext->setCoordinates(crd);

    charmmContext->calculatePotentialEnergy(true);

    // integrator
    // auto integrator = VelocityVerletIntegrator(300);
    // integrator.setContext(CharmmContext);
    // integrator.propagate(100);

  }
}
