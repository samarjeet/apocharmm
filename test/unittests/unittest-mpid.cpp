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
#include "MPIDForce.h"
#include "catch.hpp"
#include <iostream>
#include <map>
#include <string>
#include <vector>

TEST_CASE("mpid", "[mpid]") {
  SECTION("waterdimer") {
    std::vector<std::string> prmFiles{"../test/data/par_all22_prot.prm",
                                      "../test/data/toppar_water_ions.str"};
    std::shared_ptr<CharmmParameters> prm =
        std::make_shared<CharmmParameters>(prmFiles);
    std::shared_ptr<CharmmPSF> psf =
        std::make_shared<CharmmPSF>("../test/data/jac_5dhfr.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);

    fm->setBoxDimensions({20.0, 20.0, 20.0});
    fm->setFFTGrid(64, 64, 64);
    fm->setKappa(0.34);
    fm->setCutoff(8.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    fm->initialize();

    auto ctx = std::make_shared<CharmmContext>(fm);
    // auto crd = std::make_shared<CharmmCrd>("../test/data/jac_5dhfr.crd");
    const std::vector<std::vector<double>> coords{
        {2.000000, 2.000000, 2.000000}, {2.500000, 2.000000, 3.000000},
        {1.500000, 2.000000, 3.000000}, {0.000000, 0.000000, 0.000000},
        {0.500000, 0.000000, 1.000000}, {-0.500000, 0.000000, 1.000000}};
    // ctx->setCoordinates(coords);

    CudaEnergyVirial multipoleEnergyVirial;

    auto mpidForce = std::make_shared<MPIDForce<long long int, float>>(
        multipoleEnergyVirial);
    mpidForce->setNumAtoms(6);
    mpidForce->setCutoff(6.0);
    double box = 20.0;
    mpidForce->setBoxDimensions({box, box, box});
    mpidForce->setup();

    auto mpidStream = std::make_shared<cudaStream_t>();
    mpidForce->setStream(mpidStream);

    double od[3] = {0.0, 0.0, 0.00755612136146};
    double hd[3] = {-0.00204209484795, 0.0, -0.00307875299958};
    std::vector<double> odv(&od[0], &od[3]);
    std::vector<double> hdv(&hd[0], &hd[3]);

    std::map<std::string, std::vector<double>> dipolemap;
    dipolemap["O"] = odv;
    dipolemap["H1"] = hdv;
    dipolemap["H2"] = hdv;

    std::vector<float> dipoles;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        dipoles.push_back(od[j]);
      }
      for (int j = 0; j < 3; j++) {
        dipoles.push_back(hd[j]);
      }

      for (int j = 0; j < 3; j++) {
        dipoles.push_back(hd[j]);
      }
    }
    mpidForce->setDipoles(dipoles);

    double oq[6] = {0.000354030721139, 0.0, -0.000390257077096, 0.0, 0.0,
                    3.62263559571e-05};
    double hq[6] = {-3.42848248983e-05, 0.0, -0.000100240875193,
                    -1.89485963908e-06, 0.0, 0.000134525700091};

    std::vector<double> oqv(&oq[0], &oq[6]);
    std::vector<double> hqv(&hq[0], &hq[6]);

    std::map<std::string, std::vector<double>> quadrupolemap;
    quadrupolemap["O"] = oqv;
    quadrupolemap["H1"] = hqv;
    quadrupolemap["H2"] = hqv;

    std::vector<float> quadrupoles;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 6; ++j) {
        quadrupoles.push_back(oq[j]);
      }
      for (int j = 0; j < 6; ++j) {
        quadrupoles.push_back(hq[j]);
      }
      for (int j = 0; j < 6; ++j) {
        quadrupoles.push_back(hq[j]);
      }
    }
    mpidForce->setQuadrupoles(quadrupoles);

    double oo[10] = {0,
                     0,
                     0,
                     0,
                     -6.285758282686837e-07,
                     0,
                     -9.452653225954594e-08,
                     0,
                     0,
                     7.231018665791977e-07};
    double ho[10] = {-2.405600937552608e-07,
                     0,
                     -6.415084018183151e-08,
                     0,
                     -1.152422607026746e-06,
                     0,
                     -2.558537436767218e-06,
                     3.047102424084479e-07,
                     0,
                     3.710960043793964e-06};
    std::vector<double> oov(&oo[0], &oo[10]);
    std::vector<double> hov(&ho[0], &ho[10]);

    std::map<std::string, std::vector<double>> octopolemap;
    octopolemap["O"] = oov;
    octopolemap["H1"] = hov;
    octopolemap["H2"] = hov;

    std::vector<float> octopoles;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 10; j++) {
        octopoles.push_back(oo[j]);
      }
      for (int j = 0; j < 10; j++) {
        octopoles.push_back(ho[j]);
      }
      for (int j = 0; j < 10; j++) {
        octopoles.push_back(ho[j]);
      }
    }
    mpidForce->setOctopoles(octopoles);

    // auto coordinates = ctx->getCoordinates();
    const double coordinates[6][3] = {
        {2.000000, 2.000000, 2.000000}, {2.500000, 2.000000, 3.000000},
        {1.500000, 2.000000, 3.000000}, {0.000000, 0.000000, 0.000000},
        {0.500000, 0.000000, 1.000000}, {-0.500000, 0.000000, 1.000000}};
    // std::cout << coords[4] << std::endl;

    // print coordinates
    for (auto &coord : coordinates) {
      std::cout << coord[0] << " " << coord[1] << " " << coord[2] << std::endl;
    }
    float4 *xyzq_h = (float4 *)malloc(sizeof(float4) * 6);
    for (int i = 0; i < 6; i++) {
      xyzq_h[i].x = coordinates[i][0];
      xyzq_h[i].y = coordinates[i][1];
      xyzq_h[i].z = coordinates[i][2];
      xyzq_h[i].w = 0.0;
    }

    std::vector<float4> xyzq_h_vec;
    xyzq_h_vec.push_back({2.000000, 2.000000, 2.000000, 0.0});
    xyzq_h_vec.push_back({2.500000, 2.000000, 3.000000, 0.0});
    xyzq_h_vec.push_back({1.500000, 2.000000, 3.000000, 0.0});
    xyzq_h_vec.push_back({0.000000, 0.000000, 0.000000, 0.0});
    xyzq_h_vec.push_back({0.500000, 0.000000, 1.000000, 0.0});
    xyzq_h_vec.push_back({-0.500000, 0.000000, 1.000000, 0.0});

    CudaContainer<float4> xyzqContainer;
    xyzqContainer.allocate(6);
    xyzqContainer.set(xyzq_h_vec);
    float4 *xyzq = xyzqContainer.getDeviceArray().data();

    // auto xyzq = ctx->getXYZQ()->getDeviceXYZQ();
    // mpidForce->calc_force(xyzq, true, false);
    mpidForce->calculateForce(xyzq, true, false);

    mpidForce->printSphericalDipoles();
    mpidForce->printSphericalQuadrupoles();
  }
}
