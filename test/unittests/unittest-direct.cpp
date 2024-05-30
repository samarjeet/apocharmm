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
#include "catch.hpp"
#include "helper.h"
#include "test_paths.h"
#include <cuda_runtime.h>
#include <iostream>

TEST_CASE("kernel", "[Energy]") {

  std::string dataPath = getDataPath();

  std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};
  std::shared_ptr<CharmmParameters> prm =
      std::make_shared<CharmmParameters>(prmFiles);
  std::shared_ptr<CharmmPSF> psf =
      std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");

  // auto fm = std::make_shared<ForceManager>(psf, prm);

  // fm->setBoxDimensions({50.0, 50.0, 50.0});
  // fm->setFFTGrid(48, 48, 48);
  // fm->setKappa(0.34);
  // fm->setCutoff(12.0);

  // fm->setCtonnb(8.0);
  // fm->setCtofnb(10.0);

  SECTION("force") {
    std::cout << "Testing kernel\n";
    // Direct

    auto directStream = std::make_shared<cudaStream_t>();
    cudaStreamCreate(directStream.get());
    // directForceValues = std::make_shared<Force<long long int>>();
    // directForceValues->realloc(numAtoms, 1.5f);

    // auto iblo14 = psf->getIblo14();
    // auto inb14 = psf->getInb14();
    // auto vdwParamsAndTypes = prm->getVdwParamsAndTypes(psf);
    // auto inExLists = psf->getInclusionExclusionLists();

    // directForcePtr = std::make_unique<CudaPMEDirectForce<long long int,
    // float>>(
    //     directEnergyVirial, "vdw", "elec", "ewex");
    // bool q_p21 = false;
    // if (pbc == PBC::P21)
    //   q_p21 = true;
    // directForcePtr->setup(boxx, boxy, boxz, kappa, ctofnb, ctonnb, 1.0,
    // vdwType,
    //                       //                      CFSWIT, q_p21);
    //                       EWALD, q_p21);
    // directForcePtr->setBoxDimensions({boxx, boxy, boxz});
    // directForcePtr->setStream(directStream);
    // directForcePtr->setForce(directForceValues);
    // directForcePtr->setNumAtoms(numAtoms);
    // directForcePtr->setCutoff(cutoff);
    // directForcePtr->setupSorted(numAtoms);
    // directForcePtr->setupTopologicalExclusions(numAtoms, iblo14, inb14);
    // directForcePtr->setupNeighborList(numAtoms);

    // directForcePtr->set_vdwparam(vdwParamsAndTypes.vdwParams);
    // directForcePtr->set_vdwparam14(vdwParamsAndTypes.vdw14Params);
    // directForcePtr->set_vdwtype(vdwParamsAndTypes.vdwTypes);
    // directForcePtr->set_vdwtype14(vdwParamsAndTypes.vdw14Types);

    // directForcePtr->set_14_list(inExLists.sizes, inExLists.in14_ex14);
  }
}
