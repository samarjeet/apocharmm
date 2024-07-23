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
#include "CudaEMap.h"
#include "catch.hpp"
#include <iostream>
#include <string>
#include <vector>

TEST_CASE("emap", "[emap]") {
  SECTION("generate") {
    std::vector<std::string> prmFiles{"../test/data/par_all22_prot.prm",
                                      "../test/data/toppar_water_ions.str"};
    std::shared_ptr<CharmmParameters> prm =
        std::make_shared<CharmmParameters>(prmFiles);
    std::shared_ptr<CharmmPSF> psf =
        std::make_shared<CharmmPSF>("../test/data/jac_5dhfr.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);

    fm->setBoxDimensions({62.23, 62.23, 62.23});
    fm->setFFTGrid(64, 64, 64);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    fm->initialize();

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>("../test/data/jac_5dhfr.crd");
    ctx->setCoordinates(crd);

    std::shared_ptr<CudaEMap> emap = std::make_shared<CudaEMap>(ctx);
    emap->generate();
  }
}
