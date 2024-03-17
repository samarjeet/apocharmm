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
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "DcdSubscriber.h"
#include "NetCDFSubscriber.h"
#include "StateSubscriber.h"
#include "XYZSubscriber.h"
#include "catch.hpp"
#include "helper.h"
#include "test_paths.h"
#include <iostream>
#include <string>
#include <vector>

void printCoord(double4 coord) {
  std::cout << coord.x << " " << coord.y << " " << coord.z << "\n";
}
TEST_CASE("holonomic constraints", "[dynamics]") {
  auto path = getDataPath();
  SECTION("jac") {
    std::vector<std::string> prmFiles{path + "par_all22_prot.prm",
                                      path + "toppar_water_ions.str"};
    std::shared_ptr<CharmmParameters> prm =
        std::make_shared<CharmmParameters>(prmFiles);

    std::shared_ptr<CharmmPSF> psf =
        std::make_shared<CharmmPSF>(path + "jac_5dhfr.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);

    fm->setBoxDimensions({62.23, 62.23, 62.23});
    fm->setFFTGrid(64, 64, 64);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(path + "jac_5dhfr.crd");
    ctx->setCoordinates(crd);

    ctx->calculatePotentialEnergy(true, true);

    ctx->assignVelocitiesAtTemperature(300);

    auto integrator = std::make_shared<CudaLangevinThermostatIntegrator>(0.001);
    integrator->setFriction(0.0);
    integrator->setBathTemperature(300.0);
    integrator->setSimulationContext(ctx);

    auto coords = ctx->getCoordinatesCharges();
    coords.transferFromDevice();

    integrator->propagate(1000);

    coords = ctx->getCoordinatesCharges();
    coords.transferFromDevice();
    //  printCoord(coords[j]);

    auto shakeAtoms = ctx->getShakeAtoms();
    shakeAtoms.transferFromDevice();

    // Check that the shakeAtoms has the correct cluster of bonded atoms
    REQUIRE((shakeAtoms[0].x == 0 && shakeAtoms[0].y == 1 &&
             shakeAtoms[0].z == 2 && shakeAtoms[0].w == 3) == true);
    REQUIRE((shakeAtoms[1].x == 4 && shakeAtoms[1].y == 5 &&
             shakeAtoms[1].z == -1 && shakeAtoms[1].w == -1) == true);
    REQUIRE((shakeAtoms[2].x == 6 && shakeAtoms[2].y == 7 &&
             shakeAtoms[2].z == 8 && shakeAtoms[2].w == -1) == true);
    auto shakeParams = ctx->getShakeParams();
    shakeParams.transferFromDevice();

    float tol = 0.001;
    for (int shakeId = 0; shakeId < shakeAtoms.size(); ++shakeId) {
      auto d2 = shakeParams[shakeId].z;
      int i = shakeAtoms[shakeId].x;
      int j = shakeAtoms[shakeId].y;
      auto dxij = coords[j].x - coords[i].x;
      auto dyij = coords[j].y - coords[i].y;
      auto dzij = coords[j].z - coords[i].z;
      auto dij = dxij * dxij + dyij * dyij + dzij * dzij;
      CHECK(dij == Approx(d2).margin(tol));

      if (shakeAtoms[shakeId].z != -1) {
        int k = shakeAtoms[shakeId].z;

        auto dxik = coords[k].x - coords[i].x;
        auto dyik = coords[k].y - coords[i].y;
        auto dzik = coords[k].z - coords[i].z;

        auto dik = dxik * dxik + dyik * dyik + dzik * dzik;
        CHECK(dik == Approx(d2).margin(tol));

        if (shakeAtoms[shakeId].w != -1) {
          int l = shakeAtoms[shakeId].w;

          auto dxil = coords[l].x - coords[i].x;
          auto dyil = coords[l].y - coords[i].y;
          auto dzil = coords[l].z - coords[i].z;

          auto dil = dxil * dxil + dyil * dyil + dzil * dzil;
          CHECK(dil == Approx(d2).margin(tol));
        }
      }
    }

    auto waters = ctx->getWaterMolecules();
    waters.transferFromDevice();

    float d2OH = 0.91623184, d2HH = 2.29189;

    for (int water = 0; water < waters.size(); ++water) {
      int i = waters[water].x, j = waters[water].y, k = waters[water].z;
      auto dxij = coords[j].x - coords[i].x;
      auto dyij = coords[j].y - coords[i].y;
      auto dzij = coords[j].z - coords[i].z;

      auto dxik = coords[k].x - coords[i].x;
      auto dyik = coords[k].y - coords[i].y;
      auto dzik = coords[k].z - coords[i].z;

      auto dxjk = coords[j].x - coords[k].x;
      auto dyjk = coords[j].y - coords[k].y;
      auto dzjk = coords[j].z - coords[k].z;

      auto dij = dxij * dxij + dyij * dyij + dzij * dzij;
      auto dik = dxik * dxik + dyik * dyik + dzik * dzik;
      auto djk = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;

      CHECK(dij == Approx(d2OH).margin(tol));
      CHECK(dik == Approx(d2OH).margin(tol));
      CHECK(djk == Approx(d2HH).margin(tol));
    }

    // integrator.propagate(100000);
  }
}

TEST_CASE("pore", "") {
  std::string dataPath = "/u/samar/toppar/";
  std::string filePath = "/u/samar/Documents/git/test_gpu/jmin_pore/";
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
  fm->setCutoff(12.0);
  fm->setCtonnb(8.0);
  fm->setCtofnb(10.0);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(filePath + "jmin_pore_20p_min.crd");

  ctx->setCoordinates(crd);
  // ctx->calculatePotentialEnergy(true, true);
  // ctx->assignVelocitiesAtTemperature(300);

  SECTION("shake") {
    auto shakeAtoms = ctx->getShakeAtoms();
    auto shakeParams = ctx->getShakeParams();

    shakeAtoms.transferFromDevice();
    shakeParams.transferFromDevice();

    CHECK(shakeAtoms.size() == 22298);
    // Count the number of 2-body, 3-body and 4-body constraints
    int n2 = 0, n3 = 0, n4 = 0;
    for (int i = 0; i < shakeAtoms.size(); ++i) {
      if (shakeAtoms[i].z == -1)
        n2++;
      else if (shakeAtoms[i].w == -1)
        n3++;
      else
        n4++;
    }

    CHECK(n2 == 2218);
    CHECK(n3 == 18438);
    CHECK(n4 == 1642);
  }

  SECTION("settle") {
    auto waters = ctx->getWaterMolecules();
    waters.transferFromDevice();

    CHECK(waters.size() == 53222);
  }
}
