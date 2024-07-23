// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CudaBondedForce.h"
#include "ForceManager.h"
#include "gpu_utils.h"
#include <chrono>
#include <iostream>
#include <sstream>

ForceManager::ForceManager() {}

ForceManager::ForceManager(std::shared_ptr<CharmmPSF> psfIn,
                           std::shared_ptr<CharmmParameters> prmIn)
    : psf(psfIn), prm(prmIn) {
  numAtoms = psf->getNumAtoms();
  pmeSplineOrder = 4;
  kappa = 0.34;
  cutoff = 14.0;
  ctofnb = 12.;
  ctonnb = 10.0;
  nfftx = -1;
  nffty = -1;
  nfftz = -1;

  boxx = 0.0;
  boxy = 0.0;
  boxz = 0.0;

  virial.allocate(9);
  directVirial.allocate(9);
  bondedVirial.allocate(9);
  reciprocalVirial.allocate(9);

  pbc = PBC::P1;
  vdwType = VDW_VFSW;
  initialized = false;
}

// Copy-constructor
ForceManager::ForceManager(const ForceManager &fmIn)
    : psf(new CharmmPSF(*fmIn.psf.get())),
      prm(new CharmmParameters(*fmIn.prm.get())), nfftx(fmIn.nfftx),
      nffty(fmIn.nffty), nfftz(fmIn.nfftz), boxx(fmIn.boxx), boxy(fmIn.boxy),
      boxz(fmIn.boxz), pmeSplineOrder(fmIn.pmeSplineOrder), kappa(fmIn.kappa),
      cutoff(fmIn.cutoff), ctonnb(fmIn.ctonnb), ctofnb(fmIn.ctofnb),
      vdwType(fmIn.vdwType) {
  numAtoms = psf->getNumAtoms();
  boxDimensions = {boxx, boxy, boxz};
  initialized = false;
  virial.allocate(9);

  forceManagerStream = std::make_shared<cudaStream_t>();
  pbc = fmIn.getPeriodicBoundaryCondition();
  initialized = false;

  // TODO unittest this (check that copied fm has same attributes, check that
  // acting on copied fm does not change original fm attributes)
}

bool isHydrogen(std::string atomType) { return (atomType[0] == 'H'); }

void ForceManager::initializeHolonomicConstraintsVariables() {
  auto bonds = psf->getBonds();
  auto atomNames = psf->getAtomNames();
  auto atomTypes = psf->getAtomTypes();
  auto atomMasses = psf->getAtomMasses();

  std::vector<int> numBondsH(numAtoms, 0);
  std::vector<std::vector<int>> hydrogenBonds(numAtoms);

  auto h_shakeAtoms = shakeAtoms.getHostArray();
  auto h_shakeParams = shakeParams.getHostArray();
  auto bondParams = prm->getBondParams();

  for (auto bond : bonds) {

    // TODO : refine these selection criteria
    if (isHydrogen(atomTypes[bond.atom1]) ||
        isHydrogen(atomTypes[bond.atom2])) {
      if (!((atomTypes[bond.atom1] == "OT" and atomTypes[bond.atom2] == "HT") ||
            (atomTypes[bond.atom1] == "HT" and atomTypes[bond.atom2] == "OT") ||
            (atomTypes[bond.atom1][0] == 'H' and
             atomTypes[bond.atom2][0] == 'H'))) {
        int heavyAtom;
        int hydrogenAtom;
        if (isHydrogen(atomTypes[bond.atom1])) {
          heavyAtom = bond.atom2;
          hydrogenAtom = bond.atom1;
        } else {
          heavyAtom = bond.atom1;
          hydrogenAtom = bond.atom2;
        }
        numBondsH[heavyAtom]++;
        hydrogenBonds[heavyAtom].push_back(hydrogenAtom);
      }
    }
  }

  for (int i = 0; i < numAtoms; ++i) {
    if (numBondsH[i]) {
      std::vector<int> group;
      group = {i, -1, -1, -1};
      float totalMass = (float)atomMasses[i];
      float hydrogenMass;
      BondValues bondValue;
      for (int j = 0; j < hydrogenBonds[i].size(); ++j) {
        int hyd = hydrogenBonds[i][j];
        group[j + 1] = hyd;
        totalMass += atomMasses[j];
        hydrogenMass = atomMasses[hyd];

        std::string atomType0, atomType1;
        if (atomTypes[i] < atomTypes[hyd]) {
          atomType0 = atomTypes[i];
          atomType1 = atomTypes[hyd];
        } else {
          atomType0 = atomTypes[hyd];
          atomType1 = atomTypes[i];
        }
        BondKey bondKey(atomType0, atomType1);
        bondValue = bondParams[bondKey];
      }
      float avgMass = totalMass / (hydrogenBonds[i].size() + 1);
      h_shakeAtoms.push_back({group[0], group[1], group[2], group[3]});
      float4 p{1.0f / (float)atomMasses[i], avgMass,
               bondValue.b0 * bondValue.b0, 1.0f / hydrogenMass};
      h_shakeParams.push_back(p);
    }
  }

  shakeAtoms.setHostArray(h_shakeAtoms);
  shakeAtoms.transferToDevice();
  shakeParams.setHostArray(h_shakeParams);
  shakeParams.transferToDevice();
}

void ForceManager::initialize() {

  // Some sanity checks before starting
  if (boxx == 0 || boxy == 0 || boxz == 0)
    throw std::invalid_argument("Error : box dimension 0\n");
  if (cutoff <= 0 || cutoff > boxx / 2) {
    std::stringstream tmpexc;
    tmpexc << "Error: cutoff value (" << cutoff << ") not valid (boxx: " << boxx
           << ").\n";
    throw std::invalid_argument(tmpexc.str());
  }

  // If nfft not given, use values via truncating
  if (nfftx <= 0 || nffty <= 0 || nfftz <= 0) {
    auto nffts = computeFFTGridSize();
    if (nffts[0] <= 0 || nffts[1] <= 0 || nffts[2] <= 0) {
      std::stringstream tmpexc;
      tmpexc << "Error: computed FFT grid size not valid.\n";
      throw std::invalid_argument(tmpexc.str());
    }
    setFFTGrid(nffts[0], nffts[1], nffts[2]);
  }

  totalForceValues = std::make_shared<Force<double>>();
  totalForceValues->realloc(numAtoms, 1.5f);

  // Bonded
  bondedStream = std::make_shared<cudaStream_t>();
  cudaStreamCreate(bondedStream.get());
  auto bondedParamsAndList = prm->getBondedParamsAndLists(psf);
  bondedForceValues = std::make_shared<Force<long long int>>();
  bondedForceValues->realloc(numAtoms, 1.5f);

  bondedForcePtr = std::make_unique<CudaBondedForce<long long int, float>>(
      bondedEnergyVirial, "bond", "ureyb", "angle", "dihe", "imdihe", "cmap");
  bondedForcePtr->setup_list(bondedParamsAndList.listsSize,
                             bondedParamsAndList.listVal, *bondedStream);
  bondedForcePtr->setup_coef(bondedParamsAndList.paramsSize,
                             bondedParamsAndList.paramsVal);

  bondedForcePtr->setBoxDimensions({boxx, boxy, boxz});
  bondedForcePtr->setForce(bondedForceValues);
  bondedForcePtr->setStream(bondedStream);

  // Direct
  directStream = std::make_shared<cudaStream_t>();
  cudaStreamCreate(directStream.get());
  directForceValues = std::make_shared<Force<long long int>>();
  directForceValues->realloc(numAtoms, 1.5f);

  auto iblo14 = psf->getIblo14();
  auto inb14 = psf->getInb14();
  auto vdwParamsAndTypes = prm->getVdwParamsAndTypes(psf);
  auto inExLists = psf->getInclusionExclusionLists();

  directForcePtr = std::make_unique<CudaPMEDirectForce<long long int, float>>(
      directEnergyVirial, "vdw", "elec", "ewex");
  bool q_p21 = false;
  if (pbc == PBC::P21)
    q_p21 = true;
  // TODO this seems to do the job twice ?
  // 1. "directForcePtr->setup(boxx, kappa, ctofnb, (...) );"
  // 2. "directForcePtr->setBoxDimensions({boxx...});"
  directForcePtr->setup(boxx, boxy, boxz, kappa, ctofnb, ctonnb, 1.0, vdwType,
                        //                      CFSWIT, q_p21);
                        EWALD, q_p21);
  directForcePtr->setBoxDimensions({boxx, boxy, boxz});
  directForcePtr->setStream(directStream);
  directForcePtr->setForce(directForceValues);
  directForcePtr->setNumAtoms(numAtoms);
  directForcePtr->setCutoff(cutoff);
  directForcePtr->setupSorted(numAtoms);
  directForcePtr->setupTopologicalExclusions(numAtoms, iblo14, inb14);
  directForcePtr->setupNeighborList(numAtoms);

  directForcePtr->set_vdwparam(vdwParamsAndTypes.vdwParams);
  directForcePtr->set_vdwparam14(vdwParamsAndTypes.vdw14Params);
  directForcePtr->set_vdwtype(vdwParamsAndTypes.vdwTypes);
  directForcePtr->set_vdwtype14(vdwParamsAndTypes.vdw14Types);

  directForcePtr->set_14_list(inExLists.sizes, inExLists.in14_ex14);

  // Reciprocal
  reciprocalStream = std::make_shared<cudaStream_t>();
  cudaStreamCreate(reciprocalStream.get());
  reciprocalForceValues = std::make_shared<Force<long long int>>();
  reciprocalForceValues->realloc(numAtoms, 1.5f);

  reciprocalForcePtr =
      std::make_unique<CudaPMEReciprocalForce>(reciprocalEnergyVirial);
  reciprocalForcePtr->setPBC(pbc);
  reciprocalForcePtr->setParameters(nfftx, nffty, nfftz, pmeSplineOrder, kappa,
                                    *reciprocalStream);
  reciprocalForcePtr->setNumAtoms(numAtoms);
  reciprocalForcePtr->setBoxDimensions({boxx, boxy, boxz});

  reciprocalForcePtr->setForce(reciprocalForceValues);
  reciprocalForcePtr->setStream(reciprocalStream);

  cudaCheck(cudaDeviceSynchronize());
  totalPotentialEnergy.allocate(1); // doing it for diffave and difflc; for Now

  initializeHolonomicConstraintsVariables();

  forceManagerStream = std::make_shared<cudaStream_t>();
  cudaStreamCreate(forceManagerStream.get());
  initialized = true;
}

void ForceManager::setPeriodicBoundaryCondition(const PBC _pbc) {
  pbc = _pbc;
  // If changing the PBC, set "initialized" flag to FALSE
  initialized = false;
}

void ForceManager::addPSF(std::string psfFile) {
  psf = std::make_shared<CharmmPSF>(psfFile);
  // If changing the CharmmPSF, set "initialized" flag to FALSE
  initialized = false;
}

void ForceManager::addPRM(std::string prmFile) {
  prm = std::make_unique<CharmmParameters>(prmFile);
  // If changing the CharmmParameters, set "initialized" flag to FALSE
  initialized = false;
}

void ForceManager::addPRM(std::vector<std::string> prmList) {
  prm = std::make_unique<CharmmParameters>(prmList);
  initialized = false;
}

int ForceManager::getNumAtoms() const { return psf->getNumAtoms(); }
bool ForceManager::isInitialized() const { return initialized; }

void ForceManager::resetNeighborList(const float4 *xyzq) {
  directForcePtr->resetNeighborList(xyzq, numAtoms);
}

void ForceManager::setPSF(std::shared_ptr<CharmmPSF> psfIn) {
  psf = psfIn;
  initialized = false;
}

__global__ void convertLLIToFloat(int numAtoms, int stride,
                                  const long long int *__restrict__ forceLLI,
                                  float *__restrict__ forceF) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    float fx = ((float)forceLLI[index]) * INV_FORCE_SCALE;
    float fy = ((float)forceLLI[index + stride]) * INV_FORCE_SCALE;
    float fz = ((float)forceLLI[index + 2 * stride]) * INV_FORCE_SCALE;

    forceF[index] = fx;
    forceF[index + stride] = fy;
    forceF[index + 2 * stride] = fz;
  }
}

// Sums 10 potential enery terms given as pointers (e0-e9) into a double
// pointer *pe
__global__ void updatePotentialEnergy(
    const double *__restrict__ e0, const double *__restrict__ e1,
    const double *__restrict__ e2, const double *__restrict__ e3,
    const double *__restrict__ e4, const double *__restrict__ e5,
    const double *__restrict__ e6, const double *__restrict__ e7,
    const double *__restrict__ e8, const double *__restrict__ e9, double *pe) {
  pe[0] = e0[0] + e1[0] + e2[0] + e3[0] + e4[0] + e5[0] + e6[0] + e7[0] +
          e8[0] + e9[0];
}

/*
 * clearing using each ForceValues's  clear seems to be taking a lot of time
 * Testing with a consolidated kernel
 */
__global__ void clearContent() {}
void ForceManager::calc_force_part1(const float4 *xyzq, bool reset,
                                    bool calcEnergy, bool calcVirial) {

  /* FOR FUTURE USE
  std::for_each(
std::begin(forces_), std::end(forces_),
[](std::unique_ptr<force_concept> const &force) { force->calc_force(); });
*/

  if (reset) {
    /* FOR FUTURE USE
    // forces_[1].resetNeighborList(xyzq, numAtoms, directStream);
    // throw std::invalid_argument("calc_force reset option not implemented\n");
    */
  } else {
    // updateSortedKernel();
  }

  // calcEnergy = true;
  auto bStream = *bondedStream;
  auto rStream = *reciprocalStream;
  auto dStream = *directStream;

  if (!clearGraphCreated) {
    cudaStreamBeginCapture(*forceManagerStream, cudaStreamCaptureModeGlobal);

    // Clear the forces
    directForceValues->clear(*forceManagerStream);
    bondedForceValues->clear(*forceManagerStream);
    reciprocalForceValues->clear(*forceManagerStream);
    cudaStreamEndCapture(*forceManagerStream, &clearGraph);
    cudaGraphInstantiate(&cleargraphInstance, clearGraph, NULL, NULL, 0);
    clearGraphCreated = true;
  }
  cudaGraphLaunch(cleargraphInstance, *forceManagerStream);

  // Clear the virials and energy
  if (calcEnergy) {
    directEnergyVirial.clear(dStream);
    bondedEnergyVirial.clear(bStream);
    reciprocalEnergyVirial.clear(rStream);
  }
  cudaStreamSynchronize(*forceManagerStream);
}

void ForceManager::calc_force_part2(const float4 *xyzq, bool reset,
                                    bool calcEnergy, bool calcVirial) {
  /*
  // For future use...
  for (auto &&force : forces_) {
    force.calc_force(xyzq);
  }
  */

  // calcEnergy = true;

  gpu_range_start("bonded");
  bondedForcePtr->calc_force(xyzq, calcEnergy, calcVirial);
  gpu_range_stop();

  gpu_range_start("reciprocal");
  reciprocalForcePtr->calc_force(xyzq, calcEnergy, calcVirial);
  gpu_range_stop();

  gpu_range_start("direct");
  directForcePtr->calc_force(xyzq, calcEnergy, calcVirial);
  gpu_range_stop();
  //}
}
void ForceManager::calc_force_part3(const float4 *xyzq, bool reset,
                                    bool calcEnergy, bool calcVirial) {

  totalForceValues->clear(*forceManagerStream);

  cudaCheck(cudaStreamSynchronize(*bondedStream));
  totalForceValues->add<double>(*bondedForceValues, *forceManagerStream);

  cudaCheck(cudaStreamSynchronize(*reciprocalStream));
  totalForceValues->add<double>(*reciprocalForceValues, *forceManagerStream);

  cudaCheck(cudaStreamSynchronize(*directStream));
  totalForceValues->add<double>(*directForceValues, *forceManagerStream);

  // TODO : find a better way.
  // For now, as virial requires forces to be double
  cudaCheck(cudaStreamSynchronize(*forceManagerStream));

  if (calcVirial) {
    // Are we not computing virial twice ? I thought
    // directForcePtr->calc_force(xyzq, calcEnergy, calcVirial) would already
    // do it
    bondedForceValues->convert<double>(*bondedStream);
    reciprocalForceValues->convert<double>(*reciprocalStream);
    directForceValues->convert<double>(*directStream);

    cudaCheck(cudaStreamSynchronize(*bondedStream));
    cudaCheck(cudaStreamSynchronize(*reciprocalStream));
    cudaCheck(cudaStreamSynchronize(*directStream));
    bondedEnergyVirial.calcVirial(
        numAtoms, xyzq, boxDimensions[0], boxDimensions[1], boxDimensions[2],
        getForceStride(), (double *)bondedForceValues->xyz(), *bondedStream);
    // Reciprocal space virial has already been calculated in the scalar_sum
    // reciprocalEnergyVirial.calcVirial(
    //     numAtoms, xyzq, boxDimensions[0], boxDimensions[1], boxDimensions[2],
    //     getForceStride(), (double *)reciprocalForceValues->xyz(),
    //     *reciprocalStream);
    directEnergyVirial.calcVirial(
        numAtoms, xyzq, boxDimensions[0], boxDimensions[1], boxDimensions[2],
        getForceStride(), (double *)directForceValues->xyz(), *directStream);

    cudaCheck(cudaStreamSynchronize(*bondedStream));
    cudaCheck(cudaStreamSynchronize(*reciprocalStream));
    cudaCheck(cudaStreamSynchronize(*directStream));
  }

  // Copy everything (all EnergyVirials) to Host, add together
  float totalBondedEnergy = 0.0f;
  float totalNonBondedEnergy = 0.0f;
  if (calcEnergy) {
    // cudaDeviceSynchronize();
    bondedEnergyVirial.copyToHost();
    reciprocalEnergyVirial.copyToHost();
    directEnergyVirial.copyToHost();
    cudaDeviceSynchronize();

    totalBondedEnergy = bondedEnergyVirial.getEnergy("bond") +
                        bondedEnergyVirial.getEnergy("angle") +
                        bondedEnergyVirial.getEnergy("ureyb") +
                        bondedEnergyVirial.getEnergy("dihe") +
                        bondedEnergyVirial.getEnergy("imdihe");

    totalNonBondedEnergy = directEnergyVirial.getEnergy("ewex") +
                           directEnergyVirial.getEnergy("elec") +
                           directEnergyVirial.getEnergy("vdw") +
                           reciprocalEnergyVirial.getEnergy("ewks") +
                           reciprocalEnergyVirial.getEnergy("ewse");

    updatePotentialEnergy<<<1, 1, 0, *forceManagerStream>>>(
        bondedEnergyVirial.getEnergyPointer("bond"),
        bondedEnergyVirial.getEnergyPointer("angle"),
        bondedEnergyVirial.getEnergyPointer("ureyb"),
        bondedEnergyVirial.getEnergyPointer("dihe"),
        bondedEnergyVirial.getEnergyPointer("imdihe"),
        directEnergyVirial.getEnergyPointer("ewex"),
        directEnergyVirial.getEnergyPointer("elec"),
        directEnergyVirial.getEnergyPointer("vdw"),
        reciprocalEnergyVirial.getEnergyPointer("ewks"),
        reciprocalEnergyVirial.getEnergyPointer("ewse"),
        totalPotentialEnergy.getDeviceArray().data());
    cudaCheck(cudaStreamSynchronize(*forceManagerStream));

    printEnergyDecomposition = true;

    if (printEnergyDecomposition) {
      std::cout << "Bond energy         : "
                << bondedEnergyVirial.getEnergy("bond") << "\n";
      std::cout << "angle energy        : "
                << bondedEnergyVirial.getEnergy("angle") << "\n";
      std::cout << "ureyb energy        : "
                << bondedEnergyVirial.getEnergy("ureyb") << "\n";
      std::cout << "dihe energy         : "
                << bondedEnergyVirial.getEnergy("dihe") << "\n";
      std::cout << "imdihe energy       : "
                << bondedEnergyVirial.getEnergy("imdihe") << "\n";

      std::cout << "recip kspace energy : "
                << reciprocalEnergyVirial.getEnergy("ewks") << "\n";
      std::cout << "recip  self energy  : "
                << reciprocalEnergyVirial.getEnergy("ewse") << "\n";

      std::cout << "ewex energy         : "
                << directEnergyVirial.getEnergy("ewex") << "\n";
      std::cout << "elec energy         : "
                << directEnergyVirial.getEnergy("elec") << "\n";
      std::cout << "vdw energy          : "
                << directEnergyVirial.getEnergy("vdw") << "\n";
      std::cout << "Total potential energy : "
                << totalBondedEnergy + totalNonBondedEnergy << " ---\n\n";
    }
  }
}

float ForceManager::calc_force(const float4 *xyzq, bool reset, bool calcEnergy,
                               bool calcVirial) {
  calc_force_part1(xyzq, reset, calcEnergy, calcVirial);
  calc_force_part2(xyzq, reset, calcEnergy, calcVirial);
  calc_force_part3(xyzq, reset, calcEnergy, calcVirial);
  /*
   if (pbc == PBC::P21) {
     // reverse the y and z components of the force on atoms with centers
     // in the inverted asymmetric unit
     auto groups = getPSF()->getGroups();

     // find a better place for this
     int numGroups = groups.size();
     int numThreads = 128;
     int numBlocks = (numGroups - 1) / numThreads + 1;

     float3 box = {(float)boxDimensions[0], (float)boxDimensions[1],
                   (float)boxDimensions[2]};

    invertForcesAsymmetric<<<numBlocks, numThreads, 0, *forceManagerStream>>>(
         numGroups, groups.getDeviceArray().data(), box.x, xyzq,
         getForceStride(), totalForceValues->xyz());
     cudaCheck(cudaStreamSynchronize(*forceManagerStream));

   }
  */
  return 0.0;
}

std::shared_ptr<Force<double>> ForceManager::getForces() {
  cudaCheck(cudaStreamSynchronize(*forceManagerStream));
  return totalForceValues;
}

int ForceManager::getForceStride() {
  int stride = totalForceValues->stride();
  return stride;
}

const std::vector<double> &ForceManager::getBoxDimensions() {
  return boxDimensions;
}

bool ForceManager::checkBoxDimensions(const std::vector<double> &size) {
  for (auto dim : size) {
    if (dim < 0) {
      std::stringstream tmpexc;
      tmpexc << "Box dimensions (" << size[0] << " " << size[1] << " "
             << size[2] << ") are not valid.\n";
      throw std::invalid_argument(tmpexc.str());
    }
  }
  return true;
}

void ForceManager::setBoxDimensions(const std::vector<double> &size) {
  checkBoxDimensions(size);
  boxDimensions = {size[0], size[1], size[2]};

  boxx = size[0];
  boxy = size[1];
  boxz = size[2];

  if (bondedForcePtr != nullptr)
    bondedForcePtr->setBoxDimensions(boxDimensions);
  if (directForcePtr != nullptr)
    directForcePtr->setBoxDimensions(boxDimensions);
  if (reciprocalForcePtr != nullptr)
    reciprocalForcePtr->setBoxDimensions(boxDimensions);
}

std::vector<Bond> ForceManager::getBonds() { return psf->getBonds(); }

bool ForceManager::isComposite() const { return false; }

void ForceManager::addForceManager(std::shared_ptr<ForceManager> fm) {
  std::stringstream tmpexc;
  tmpexc << "ForceManager can be added only to composite FM\n";
  throw std::invalid_argument(tmpexc.str());
}

std::vector<int> ForceManager::computeFFTGridSize() {
  checkBoxDimensions(boxDimensions);
  int fx = 2 * (int(boxx) / 2);
  int fy = 2 * (int(boxy) / 2);
  int fz = 2 * (int(boxz) / 2);
  if (fx < 2) {
    fx = 2;
    std::cout << "Warning: boxx seems very small (" << boxx
              << "), setting associated fft grid size to2\n";
  }
  if (fy < 2) {
    fy = 2;
    std::cout << "Warning: boxy seems very small (" << boxy
              << "), setting associated fft grid size to2\n";
  }
  if (fz < 2) {
    fz = 2;
    std::cout << "Warning: boxz seems very small (" << boxz
              << "), setting associated fft grid size to2\n";
  }
  return {fx, fy, fz};
}

CudaContainer<double>
ForceManager::computeAllChildrenPotentialEnergy(const float4 *xyzq) {
  std::stringstream tmpexc;
  tmpexc << "computeAllChildrenPotentialEnergy should only be called from "
            "composite FM\n";
  throw std::invalid_argument(tmpexc.str());
}

/////////////////////////////////
// ForceManagerComposite
////////////////////////////////////

// Constructors//
//-------------//
ForceManagerComposite::ForceManagerComposite() {
  setLambda(1.0);
  // numAtoms = 0;
}

ForceManagerComposite::ForceManagerComposite(
    std::vector<std::shared_ptr<ForceManager>> fmList) {
  for (int i = 0; i < fmList.size(); i++) {
    addForceManager(fmList[i]);
  }
}

bool ForceManagerComposite::isComposite() const { return true; }

void ForceManagerComposite::addForceManager(std::shared_ptr<ForceManager> fm) {
  if (children.size() == 0)
    numAtoms = fm->getNumAtoms();
  else
    assert(numAtoms == fm->getNumAtoms());
  children.push_back(fm);

  // add xyzq as well
  // we only need the charges but it is available only along with xyz
  std::vector<float4> coords;
  auto tmpxyzq = std::make_shared<XYZQ>();
  auto charges = fm->getPSF()->getAtomCharges();
  for (int i = 0; i < numAtoms; ++i) {
    float4 f4 = {0.0f, 0.0f, 0.0f, (float)charges[i]};
    coords.push_back(f4);
  }
  tmpxyzq->set_ncoord(numAtoms);
  tmpxyzq->set_xyzq(numAtoms, coords.data(), 0);
  xyzqs.push_back(tmpxyzq);
}

void ForceManagerComposite::initialize() {
  for (auto child : children) {
    child->initialize();
  }

  totalForceValues = std::make_shared<Force<double>>();
  totalForceValues->realloc(numAtoms, 1.5f);
  // TODO move this to MBAR FM

  // TODO :: create a stream for each FM
  // it will be used while copying xyz to their respective xyzq s
  compositeStream = std::make_shared<cudaStream_t>();
  cudaStreamCreate(compositeStream.get());

  totalPotentialEnergy.allocate(children.size());

  childrenPotentialEnergy.allocate(children.size());
}

std::vector<Bond> ForceManagerComposite::getBonds() {
  return children[0]->getBonds();
}

float ForceManagerComposite::getLambda() const { return lambda; }

void ForceManagerComposite::setLambda(float lambdaIn) {
  // if (children.size() != 2)
  //  std::cerr << ""
  lambda = lambdaIn;
  setSelectorVec({1 - lambda, lambda});
}

void ForceManagerComposite::setSelectorVec(std::vector<float> lambdaIn) {
  lambdai = lambdaIn;
}

void ForceManagerComposite::setBoxDimensions(const std::vector<double> &size) {
  for (auto child : children) {
    child->setBoxDimensions(size);
  }
}

void ForceManagerComposite::setKappa(float kappaIn) {
  for (auto child : children) {
    child->setKappa(kappaIn);
  }
}

void ForceManagerComposite::setCutoff(float cutoffIn) {
  for (auto child : children) {
    child->setCutoff(cutoffIn);
  }
}
void ForceManagerComposite::setCtonnb(float ctonnbIn) {
  for (auto child : children) {
    child->setCtonnb(ctonnbIn);
  }
}
void ForceManagerComposite::setCtofnb(float ctofnbIn) {
  for (auto child : children) {
    child->setCtofnb(ctofnbIn);
  }
}
void ForceManagerComposite::setPmeSplineOrder(int order) {
  for (auto child : children) {
    child->setPmeSplineOrder(order);
  }
}

void ForceManagerComposite::setFFTGrid(int nx, int ny, int nz) {
  for (auto child : children) {
    child->setFFTGrid(nx, ny, nz);
  }
}

void ForceManagerComposite::resetNeighborList(const float4 *xyzq) {
  for (int i = 0; i < children.size(); ++i) {
    children[i]->resetNeighborList(xyzq);
  }
}

// Given two XYZ pointers, copies coordinates from "main" to "child"
__global__ void copyXYZ(int numAtoms, const float4 *__restrict__ main,
                        float4 *__restrict__ child) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    child[index].x = main[index].x;
    child[index].y = main[index].y;
    child[index].z = main[index].z;
  }
}

float ForceManagerComposite::calc_force(const float4 *xyzq, bool reset,
                                        bool calcEnergy, bool calcVirial) {
  float pe = 0.0;
  for (int i = 0; i < children.size(); ++i) {
    auto child = children[i];
    int numThreads = 128;
    int numBlocks = (numAtoms - 1) / numThreads + 1;

    // Copy xyz part of XYZQ to Device's XYZQ
    copyXYZ<<<numBlocks, numThreads, 0, *compositeStream>>>(
        numAtoms, xyzq, xyzqs[i]->getDeviceXYZQ());
  }
  cudaStreamSynchronize(*compositeStream);

  for (int i = 0; i < children.size(); ++i) {
    auto child = children[i];
    child->calc_force_part1(xyzqs[i]->getDeviceXYZQ(), reset, calcEnergy,
                            calcVirial);
  }
  for (int i = 0; i < children.size(); ++i) {
    auto child = children[i];
    child->calc_force_part2(xyzqs[i]->getDeviceXYZQ(), reset, calcEnergy,
                            calcVirial);
  }
  for (int i = 0; i < children.size(); ++i) {
    auto child = children[i];
    child->calc_force_part3(xyzqs[i]->getDeviceXYZQ(), reset, calcEnergy,
                            calcVirial);
  }

  // What is happening here ?
  for (int i = 0; i < children.size(); ++i) {
    auto child = children[i];

    auto childPotentialEnergy = child->getPotentialEnergy();
    // TODO : this is not the best way. Profile it to see the impact
    cudaMemcpyAsync(totalPotentialEnergy.getDeviceArray().data() + i,
                    childPotentialEnergy.getDeviceArray().data(),
                    sizeof(double), cudaMemcpyDeviceToDevice, *compositeStream);
  }
  return pe;
}

// combine the two end states
__global__ void combine(float lambda, int numAtoms, int stride,
                        const float *__restrict__ force1,
                        const float *__restrict__ force2,
                        float *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    force[index] = (1.0 - lambda) * force1[index] + lambda * force2[index];
    force[index + stride] = (1.0 - lambda) * force1[index + stride] +
                            lambda * force2[index + stride];
    force[index + 2 * stride] = (1.0 - lambda) * force1[index + 2 * stride] +
                                lambda * force2[index + 2 * stride];
  }
}

// combine multiple end states
__global__ void combine(float lambda, int numAtoms, int stride,
                        const double *__restrict__ forceChild,
                        double *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    force[index] += forceChild[index];
    force[index + stride] += lambda * forceChild[index + stride];
    force[index + 2 * stride] += lambda * forceChild[index + 2 * stride];
  }
}

std::shared_ptr<Force<double>> ForceManagerComposite::getForces() {

  std::cout
      << "Don't call me. Instead call getForcesInChild with <int> childId";
  int numThreads = 512;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  totalForceValues->clear();

  for (int i = 0; i < children.size(); ++i) {
    if (lambdai[i] != 0.0) {
      combine<<<numBlocks, numThreads, 0, *compositeStream>>>(
          lambdai[i], numAtoms, getForceStride(),
          children[i]->getForces()->xyz(), totalForceValues->xyz());
    }
  }

  cudaStreamSynchronize(*compositeStream);

  return totalForceValues;
}

CudaContainer<double> ForceManagerComposite::getVirial() {
  std::cout
      << "Don't call me. Instead call getVirialInChild with <int> childId\n"
      << "For now, returns child[0]'s virial.";
  return children[0]->getVirial();
}

std::shared_ptr<Force<double>>
ForceManagerComposite::getForcesInChild(int childId) {
  assert(childId < children.size());
  return children[childId]->getForces();
}

std::vector<float> ForceManagerComposite::getPotentialEnergies() {
  std::vector<float> dummy;
  return dummy;
}

CudaContainer<double> ForceManagerComposite::getPotentialEnergy() {
  return totalPotentialEnergy;
}
std::vector<float> ForceManager::getPotentialEnergies() {
  // TODO : Don't do this
  // Pb : this should not be done on the Host side
  std::vector<float> out;
  // Copy every energy-virial to host
  directEnergyVirial.copyToHost();
  bondedEnergyVirial.copyToHost();
  reciprocalEnergyVirial.copyToHost();
  cudaCheck(cudaStreamSynchronize(*forceManagerStream));

  // Add every energy component
  float totalBondedEnergy = bondedEnergyVirial.getEnergy("bond") +
                            bondedEnergyVirial.getEnergy("angle") +
                            bondedEnergyVirial.getEnergy("ureyb") +
                            bondedEnergyVirial.getEnergy("dihe") +
                            bondedEnergyVirial.getEnergy("imdihe");

  float totalNonBondedEnergy = directEnergyVirial.getEnergy("ewex") +
                               directEnergyVirial.getEnergy("elec") +
                               directEnergyVirial.getEnergy("vdw") +
                               reciprocalEnergyVirial.getEnergy("ewks") +
                               reciprocalEnergyVirial.getEnergy("ewse");

  // Put in "out" , return "out"
  out.push_back(totalBondedEnergy + totalNonBondedEnergy);
  return out;
}

const std::vector<double> &ForceManagerComposite::getBoxDimensions() {
  return children[0]->getBoxDimensions();
}

int ForceManagerComposite::getCompositeSize() { return children.size(); }

CudaContainer<double> ForceManager::getVirial() {
  directEnergyVirial.getVirial(directVirial);
  bondedEnergyVirial.getVirial(bondedVirial);
  reciprocalEnergyVirial.getVirial(reciprocalVirial);

  directVirial.transferFromDevice();
  bondedVirial.transferFromDevice();
  reciprocalVirial.transferFromDevice();

  if (pbc == PBC::P21) {
    for (int i = 0; i < 9; ++i) {
      reciprocalVirial[i] /= 2.0;
    }
    reciprocalVirial.transferToDevice();
  }

  for (int i = 0; i < 9; i++) {
    virial[i] = directVirial[i] + bondedVirial[i] + reciprocalVirial[i];
  }
  virial.transferToDevice();
  // return directVirial;
  return virial;
}

void ForceManager::setCharmmContext(std::shared_ptr<CharmmContext> ctx) {
  context = ctx;
}

bool ForceManager::hasCharmmContext() {
  if (context == nullptr) {
    return false;
  }
  return true;
}

CudaContainer<double> ForceManager::getPotentialEnergy() {
  return totalPotentialEnergy;
}

std::vector<std::shared_ptr<ForceManager>> ForceManager::getChildren() {
  return children;
}

std::vector<std::shared_ptr<ForceManager>>
ForceManagerComposite::getChildren() {
  return this->children;
}

void ForceManager::setPrintEnergyDecomposition(bool bIn) {
  printEnergyDecomposition = bIn;
}

std::map<std::string, double> ForceManager::getEnergyComponents() {
  std::map<std::string, double> energyDecompositionMap;
  energyDecompositionMap["bond"] = bondedEnergyVirial.getEnergy("bond");
  energyDecompositionMap["angle"] = bondedEnergyVirial.getEnergy("angle");
  energyDecompositionMap["ureyb"] = bondedEnergyVirial.getEnergy("ureyb");
  energyDecompositionMap["dihe"] = bondedEnergyVirial.getEnergy("dihe");
  energyDecompositionMap["imdihe"] = bondedEnergyVirial.getEnergy("imdihe");
  energyDecompositionMap["ewks"] = reciprocalEnergyVirial.getEnergy("ewks");
  energyDecompositionMap["ewse"] = reciprocalEnergyVirial.getEnergy("ewse");
  energyDecompositionMap["ewex"] = directEnergyVirial.getEnergy("ewex");
  energyDecompositionMap["elec"] = directEnergyVirial.getEnergy("elec");
  energyDecompositionMap["vdw"] = directEnergyVirial.getEnergy("vdw");

  return energyDecompositionMap;
}

bool ForceManagerComposite::isInitialized() const {
  for (auto child : children) {
    if (!child->isInitialized()) {
      return false;
    }
  }
  return true;
}

CudaContainer<double>
ForceManagerComposite::computeAllChildrenPotentialEnergy(const float4 *xyzqIn) {
  // Compute potential energy for each child
  for (int i = 0; i < children.size(); ++i) {
    auto child = children[i];
    child->calc_force(xyzqIn, false, true, false);
    CudaContainer<double> pecc = child->getPotentialEnergy();
    pecc.transferFromDevice();
    childrenPotentialEnergy[i] = pecc.getHostArray()[0];
    // std::cout << "Child " << i << " potential energy : " <<
    // childrenPotentialEnergy[i] << std::endl;
  }
  // TODO: this is kind of a hotfix. Not sure this is ideal.
  childrenPotentialEnergy.transferToDevice();
  return childrenPotentialEnergy;
}
