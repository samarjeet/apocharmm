// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "ForceManagerGenerator.h"
#include <iostream>

ForceManagerGenerator::ForceManagerGenerator() { ; }

ForceManagerGenerator::ForceManagerGenerator(std::shared_ptr<ForceManager> fmIn)
    : baseForceManager(fmIn) {
  ;
}

std::shared_ptr<ForceManager> ForceManagerGenerator::generateForceManager() {
  std::cout << "Not Implemented -- should not be called ?" << std::endl;
  exit(1);
  return baseForceManager;
}

AlchemicalForceManagerGenerator::AlchemicalForceManagerGenerator(
    std::shared_ptr<ForceManager> forceManagerIn) {
  baseForceManager = forceManagerIn;
}

void AlchemicalForceManagerGenerator::setAlchemicalRegion(
    std::vector<int> alchRegionIn) {
  alchemicalRegion = alchRegionIn;
}

std::vector<int> AlchemicalForceManagerGenerator::getAlchemicalRegion() {
  return alchemicalRegion;
}

std::shared_ptr<ForceManager>
AlchemicalForceManagerGenerator::generateForceManager(
    double lambdaElecIn, double lambdaVdWIn = 1.0) {
  // generate a SINGLE force Manager from a given lambda_elec value ?
  // For now, does not take lambda_vdw into account
  if (lambdaVdWIn != 1.0) {
    throw std::invalid_argument("Not Implemented Yet ! "
                                "[AlchemicalForceManagerGenerator::"
                                "generateForceManager, lambdaVdWIn != 1.0 ]");
  }
  // Generated ForceManager object: should be same as base with scaled charges.
  // How to create that :  several possibilities. Choose one.
  // 1) create a brand new with a changed psf ?
  //   > auto newPSF = modifyPSF(lambdaIn, alchRegion);
  //   > auto newFM = ForceManager(newPSF, basicFM->getParameters() );
  // 2)  modify the forces within newFM
  //   > auto newFM = ForceManager( baseForceManager); // implicit Copy
  //   Constructor > newFM.modifyForces(alchRegion, lambdaIn);

  // Make sure that alchemical region has been defined
  if (alchemicalRegion.size() <= 0) {
    throw std::invalid_argument(
        "Size of alchemicalRegion is invalid. alchemicalRegion was not "
        "defined, or was badly defined.");
  }

  std::shared_ptr<ForceManager> newFM =
      std::make_shared<ForceManager>(*baseForceManager);
  // Modify electrostatics
  modifyElectrostatics(newFM, lambdaElecIn);
  // Modify vdW
  modifyvdW(newFM, lambdaVdWIn);

  // TODO
  // - Recompute numAtoms once modif has been done ?
  // - link CharmmContext !
  return newFM;
}

void AlchemicalForceManagerGenerator::modifyElectrostatics(
    std::shared_ptr<ForceManager> fmIn, double lambdaIn) {
  int ialch, i;
  double tmpcharge;
  // Easiest/inelegant version : generate a whole new PSF with scaled charges,
  // then initialize the ForceManager object.

  // More elegant version: modify the baseForceManager object directly.
  // Charges are initially coming from the PSF file (not contained in the
  // .prm, .str files).
  // Within Composite FMs, each child's xyzq is initialized when
  // "addForceManager"

  // Get PSF's charges
  auto psf = fmIn->getPSF();
  std::vector<double> charges = psf->getAtomCharges();

  // For each element in the alch region, scale the charges by lambdaIn
  for (i = 0; i < alchemicalRegion.size(); i++) {
    ialch = alchemicalRegion[i];
    tmpcharge = charges[ialch];
    tmpcharge = tmpcharge * lambdaIn;
    charges[ialch] = tmpcharge;
  }

  // New PSF object to give new charges to
  std::shared_ptr<CharmmPSF> newPSF = std::make_shared<CharmmPSF>(*psf);
  newPSF->setAtomCharges(charges);
  fmIn->setPSF(newPSF);
}

void AlchemicalForceManagerGenerator::modifyvdW(
    std::shared_ptr<ForceManager> fmIn, float lambdaIn) {
  // Modify vdW params
  // Basic idea : create new atom types for the alchemical region atoms, modify
  // the vdw parameters for these new atoms types
  // 1. Identify unique atom types in the alchemical region
  // 2. Generate modified atom type for each, with e_i *= sqrt(lambda)

  throw std::invalid_argument("Not implemented yet ! [modifyvdW]");
}
