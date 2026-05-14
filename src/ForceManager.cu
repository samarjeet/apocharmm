// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CharmmContext.h"
#include "ForceManager.h"
#include "gpu_utils.h"
#include <chrono>
#include <iostream>
#include <string>

ForceManager::ForceManager(void) {
  m_Context = nullptr;
  m_Psf = nullptr;
  m_Prm = nullptr;

  m_IsInitialized = false;

  m_BondedStream = nullptr;
  m_ReciprocalStream = nullptr;
  m_DirectStream = nullptr;
  m_ForceManagerStream = nullptr;

  m_BondedForceValues = nullptr;
  m_ReciprocalForceValues = nullptr;
  m_DirectForceValues = nullptr;
  m_TotalForceValues = nullptr;

  m_BoxX = -9999.9999f;
  m_BoxY = -9999.9999f;
  m_BoxZ = -9999.9999f;
  m_BoxDimensions = {-9999.9999, -9999.9999, -9999.9999};

  m_Kappa = 0.34;
  m_Cutoff = 14.0;
  m_Ctonnb = 12.0;
  m_Ctofnb = 10.0;

  m_NfftX = -1;
  m_NfftY = -1;
  m_NfftZ = -1;

  m_PmeSplineOrder = 4;

  m_Pbc = PBC::P1;

  m_BondedForcePtr = nullptr;
  m_ReciprocalForcePtr = nullptr;
  m_DirectForcePtr = nullptr;

  m_BondedVirial.resize(9);
  m_ReciprocalVirial.resize(9);
  m_DirectVirial.resize(9);
  m_TotalVirial.resize(9);

  m_BondedVirial.setToValue(0.0);
  m_ReciprocalVirial.setToValue(0.0);
  m_DirectVirial.setToValue(0.0);
  m_TotalVirial.setToValue(0.0);

  m_ClearGraphCreated = false;

  m_ComputeDirectSpaceForces = true;

  m_VdwType = VDW_VFSW;

  m_PrintEnergyDecomposition = false;
}

ForceManager::ForceManager(std::shared_ptr<CharmmPSF> psf,
                           std::shared_ptr<CharmmParameters> prm)
    : ForceManager() {
  m_Psf = psf;
  m_Prm = prm;
}

ForceManager::ForceManager(const ForceManager &other) : ForceManager() {
  m_Psf = std::make_shared<CharmmPSF>(*other.m_Psf);
  m_Prm = std::make_shared<CharmmParameters>(*other.m_Prm);

  m_BoxX = other.m_BoxX;
  m_BoxY = other.m_BoxY;
  m_BoxZ = other.m_BoxZ;
  m_BoxDimensions = other.m_BoxDimensions;

  m_Kappa = other.m_Kappa;
  m_Cutoff = other.m_Cutoff;
  m_Ctonnb = other.m_Ctonnb;
  m_Ctofnb = other.m_Ctofnb;

  m_NfftX = other.m_NfftX;
  m_NfftY = other.m_NfftY;
  m_NfftZ = other.m_NfftZ;

  m_PmeSplineOrder = other.m_PmeSplineOrder;

  m_Pbc = other.m_Pbc;

  m_VdwType = other.m_VdwType;

  // TODO unittest this (check that copied fm has same attributes, check that
  // acting on copied fm does not change original fm attributes)
}

ForceManager::~ForceManager(void) { this->dealloc(); }

void ForceManager::setCharmmContext(std::shared_ptr<CharmmContext> ctx) {
  m_Context = ctx;
  return;
}

void ForceManager::setPsf(std::shared_ptr<CharmmPSF> psf) {
  m_Psf = psf;
  // If changing the CharmmPSF, set "initialized" flag to FALSE
  m_IsInitialized = false;
  return;
}

void ForceManager::addPsf(const std::string &psfFile) {
  m_Psf = std::make_shared<CharmmPSF>(psfFile);
  // If changing the CharmmPSF, set "initialized" flag to FALSE
  m_IsInitialized = false;
  return;
}

void ForceManager::addPrm(const std::string &prmFile) {
  m_Prm = std::make_unique<CharmmParameters>(prmFile);
  // If changing the CharmmParameters, set "initialized" flag to FALSE
  m_IsInitialized = false;
  return;
}

void ForceManager::addPrm(const std::vector<std::string> &prmList) {
  m_Prm = std::make_unique<CharmmParameters>(prmList);
  m_IsInitialized = false;
  return;
}

void ForceManager::setBoxDimensions(const std::vector<double> &boxDimensions) {
  this->checkBoxDimensions(boxDimensions);
  m_BoxDimensions = boxDimensions;

  m_BoxX = static_cast<float>(boxDimensions[0]);
  m_BoxY = static_cast<float>(boxDimensions[1]);
  m_BoxZ = static_cast<float>(boxDimensions[2]);

  if (m_BondedForcePtr != nullptr)
    m_BondedForcePtr->setBoxDimensions(boxDimensions);
  if (m_ReciprocalForcePtr != nullptr)
    m_ReciprocalForcePtr->setBoxDimensions(boxDimensions);
  if (m_DirectForcePtr != nullptr)
    m_DirectForcePtr->setBoxDimensions(boxDimensions);

  for (ForceView &forceView : m_ForceViews)
    forceView.setBoxDimensions(boxDimensions);

  return;
}

void ForceManager::setKappa(const float kappa) {
  m_Kappa = kappa;
  return;
}

void ForceManager::setCutoff(const float cutoff) {
  m_Cutoff = cutoff;
  return;
}

void ForceManager::setCtonnb(const float ctonnb) {
  m_Ctonnb = ctonnb;
  return;
}

void ForceManager::setCtofnb(const float ctofnb) {
  m_Ctofnb = ctofnb;
  return;
}

void ForceManager::setFFTGrid(const int nfftx, const int nffty,
                              const int nfftz) {
  m_NfftX = nfftx;
  m_NfftY = nffty;
  m_NfftZ = nfftz;
  return;
}

void ForceManager::setPmeSplineOrder(const int pmeSplineOrder) {
  m_PmeSplineOrder = pmeSplineOrder;
  return;
}

void ForceManager::setPeriodicBoundaryCondition(const PBC pbc) {
  m_Pbc = pbc;
  // If changing the PBC, set "initialized" flag to FALSE
  m_IsInitialized = false;
  return;
}

void ForceManager::setVdwType(const int vdwType) {
  m_VdwType = vdwType;
  return;
}

void ForceManager::setPrintEnergyDecomposition(
    const bool printEnergyDecomposition) {
  m_PrintEnergyDecomposition = printEnergyDecomposition;
  return;
}

void ForceManager::addForceManager(std::shared_ptr<ForceManager> fm) {
  throw std::invalid_argument("ERROR: Cannot add ForceManager to ForceManager");
  return;
}

std::shared_ptr<CharmmContext> ForceManager::getContext(void) {
  return m_Context;
}

bool ForceManager::hasCharmmContext(void) const {
  return (m_Context != nullptr);
}

std::shared_ptr<CharmmPSF> ForceManager::getPsf(void) { return m_Psf; }

int ForceManager::getNumAtoms(void) const { return m_Psf->getNumAtoms(); }

std::shared_ptr<CharmmParameters> ForceManager::getPrm(void) { return m_Prm; }

bool ForceManager::isInitialized(void) const { return m_IsInitialized; }

const CudaContainer<int4> &ForceManager::getShakeAtoms(void) const {
  return m_ShakeAtoms;
}

CudaContainer<int4> &ForceManager::getShakeAtoms(void) { return m_ShakeAtoms; }

const CudaContainer<float4> &ForceManager::getShakeParams(void) const {
  return m_ShakeParams;
}

CudaContainer<float4> &ForceManager::getShakeParams(void) {
  return m_ShakeParams;
}

const CudaEnergyVirial &ForceManager::getBondedEnergyVirial(void) const {
  return m_BondedEnergyVirial;
}

CudaEnergyVirial &ForceManager::getBondedEnergyVirial(void) {
  return m_BondedEnergyVirial;
}

const CudaEnergyVirial &ForceManager::getReciprocalEnergyVirial(void) const {
  return m_ReciprocalEnergyVirial;
}

CudaEnergyVirial &ForceManager::getReciprocalEnergyVirial(void) {
  return m_ReciprocalEnergyVirial;
}

const CudaEnergyVirial &ForceManager::getDirectEnergyVirial(void) const {
  return m_DirectEnergyVirial;
}

CudaEnergyVirial &ForceManager::getDirectEnergyVirial(void) {
  return m_DirectEnergyVirial;
}

std::map<std::string, double> ForceManager::getEnergyComponents(void) {
  std::map<std::string, double> energyDecompositionMap;

  energyDecompositionMap["bond"] = m_BondedEnergyVirial.getEnergy("bond");
  energyDecompositionMap["angle"] = m_BondedEnergyVirial.getEnergy("angle");
  energyDecompositionMap["ureyb"] = m_BondedEnergyVirial.getEnergy("ureyb");
  energyDecompositionMap["dihe"] = m_BondedEnergyVirial.getEnergy("dihe");
  energyDecompositionMap["imdihe"] = m_BondedEnergyVirial.getEnergy("imdihe");

  energyDecompositionMap["ewks"] = m_ReciprocalEnergyVirial.getEnergy("ewks");
  energyDecompositionMap["ewse"] = m_ReciprocalEnergyVirial.getEnergy("ewse");

  energyDecompositionMap["ewex"] = m_DirectEnergyVirial.getEnergy("ewex");
  energyDecompositionMap["elec"] = m_DirectEnergyVirial.getEnergy("elec");
  energyDecompositionMap["vdw"] = m_DirectEnergyVirial.getEnergy("vdw");

  return energyDecompositionMap;
}

std::shared_ptr<cudaStream_t> ForceManager::getBondedStream(void) {
  return m_BondedStream;
}

std::shared_ptr<cudaStream_t> ForceManager::getReciprocalStream(void) {
  return m_ReciprocalStream;
}

std::shared_ptr<cudaStream_t> ForceManager::getDirectStream(void) {
  return m_DirectStream;
}

std::shared_ptr<cudaStream_t> ForceManager::getForceManagerStream(void) {
  return m_ForceManagerStream;
}

std::shared_ptr<Force<long long int>> ForceManager::getBondedForcevalues(void) {
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));
  return m_BondedForceValues;
}
std::shared_ptr<Force<long long int>>
ForceManager::getReciprocalForcevalues(void) {
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));
  return m_ReciprocalForceValues;
}
std::shared_ptr<Force<long long int>> ForceManager::getDirectForcevalues(void) {
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));
  return m_DirectForceValues;
}
std::shared_ptr<Force<double>> ForceManager::getTotalForcevalues(void) {
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));
  return m_TotalForceValues;
}

std::shared_ptr<Force<double>> ForceManager::getForces(void) {
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));
  return m_TotalForceValues;
}

int ForceManager::getForceStride(void) const {
  return m_TotalForceValues->stride();
}

const std::vector<double> &ForceManager::getBoxDimensions(void) const {
  return m_BoxDimensions;
}

std::vector<double> &ForceManager::getBoxDimensions(void) {
  return m_BoxDimensions;
}

float ForceManager::getKappa(void) const { return m_Kappa; }

float ForceManager::getCutoff(void) const { return m_Cutoff; }

float ForceManager::getCtonnb(void) const { return m_Ctonnb; }

float ForceManager::getCtofnb(void) const { return m_Ctofnb; }

std::vector<int> ForceManager::getFFTGrid(void) const {
  return {m_NfftX, m_NfftY, m_NfftZ};
}

PBC ForceManager::getPeriodicBoundaryCondition(void) const { return m_Pbc; }

CudaContainer<double> &ForceManager::getPotentialEnergy(void) {
  return m_TotalPotentialEnergy;
}

float ForceManager::getPotentialEnergies(void) {
  // TODO : Don't do this
  // Pb : this should not be done on the Host side
  // Copy every energy-virial to host
  m_DirectEnergyVirial.copyToHost();
  m_BondedEnergyVirial.copyToHost();
  m_ReciprocalEnergyVirial.copyToHost();
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));

  // Add every energy component
  float totalBondedEnergy =
      static_cast<float>(m_BondedEnergyVirial.getEnergy("bond")) +
      static_cast<float>(m_BondedEnergyVirial.getEnergy("angle")) +
      static_cast<float>(m_BondedEnergyVirial.getEnergy("ureyb")) +
      static_cast<float>(m_BondedEnergyVirial.getEnergy("dihe")) +
      static_cast<float>(m_BondedEnergyVirial.getEnergy("imdihe"));

  float totalNonBondedEnergy =
      static_cast<float>(m_ReciprocalEnergyVirial.getEnergy("ewks")) +
      static_cast<float>(m_ReciprocalEnergyVirial.getEnergy("ewse")) +
      static_cast<float>(m_DirectEnergyVirial.getEnergy("ewex")) +
      static_cast<float>(m_DirectEnergyVirial.getEnergy("elec")) +
      static_cast<float>(m_DirectEnergyVirial.getEnergy("vdw"));

  return (totalBondedEnergy + totalNonBondedEnergy);
}

CudaContainer<double> &ForceManager::getVirial(void) {
  m_BondedEnergyVirial.getVirial(m_BondedVirial);
  m_ReciprocalEnergyVirial.getVirial(m_ReciprocalVirial);
  m_DirectEnergyVirial.getVirial(m_DirectVirial);

  m_BondedVirial.transferFromDevice();
  m_ReciprocalVirial.transferFromDevice();
  m_DirectVirial.transferFromDevice();

  if (m_Pbc == PBC::P21) {
    for (int i = 0; i < 9; i++)
      m_ReciprocalVirial[i] /= 2.0;
    m_ReciprocalVirial.transferToDevice();
  }

  for (int i = 0; i < 9; i++) {
    m_TotalVirial[i] =
        m_BondedVirial[i] + m_ReciprocalVirial[i] + m_DirectVirial[i];
  }
  m_TotalVirial.transferToDevice();

  return m_TotalVirial;
}

int ForceManager::getVdwType(void) const { return m_VdwType; }

// const std::vector<Bond> &ForceManager::getBonds(void) const {
//   return m_Psf->getBonds();
// }

// std::vector<Bond> &ForceManager::getBonds(void) { return m_Psf->getBonds(); }

bool ForceManager::isComposite(void) const { return false; }

const std::vector<std::shared_ptr<ForceManager>> &
ForceManager::getChildren(void) const {
  return m_Children;
}

std::vector<std::shared_ptr<ForceManager>> &ForceManager::getChildren(void) {
  return m_Children;
}

void ForceManager::initialize(void) {
  // Some sanity checks before starting
  if ((m_BoxX == -9999.9999f) || (m_BoxY == -9999.9999f) ||
      (m_BoxZ == -9999.9999f)) {
    throw std::invalid_argument("Error: Box dimension was not set");
  }

  if ((m_Cutoff <= 0.0f) || (m_Cutoff > m_BoxX / 2.0f)) {
    throw std::invalid_argument(
        "Error: Cutoff value (" + std::to_string(m_Cutoff) +
        ") not valid (boxx: " + std::to_string(m_BoxX) + ")");
  }

  // If nfft not given, use values via truncating
  if ((m_NfftX <= 0) || (m_NfftY <= 0) || (m_NfftZ <= 0)) {
    std::vector<int> nfft = this->computeFFTGridSize();
    if ((nfft[0] <= 0) || (nfft[1] <= 0) || (nfft[2] <= 0))
      throw std::runtime_error("Error: Computed FFT grid size not valid");
    this->setFFTGrid(nfft[0], nfft[1], nfft[2]);
  }

  const int numAtoms = m_Psf->getNumAtoms();

  // Bonded
  m_BondedStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_BondedStream.get()));

  auto bondedParamsAndList = m_Prm->getBondedParamsAndLists(m_Psf);
  m_BondedForceValues = std::make_shared<Force<long long int>>();
  m_BondedForceValues->realloc(numAtoms, 1.5f);

  m_BondedForcePtr = std::make_unique<CudaBondedForce<long long int, float>>(
      m_BondedEnergyVirial, "bond", "ureyb", "angle", "dihe", "imdihe", "cmap");
  m_BondedForcePtr->setup_list(bondedParamsAndList.listsSize,
                               bondedParamsAndList.listVal, *m_BondedStream);
  m_BondedForcePtr->setup_coef(bondedParamsAndList.paramsSize,
                               bondedParamsAndList.paramsVal);
  m_BondedForcePtr->setBoxDimensions(m_BoxDimensions);
  m_BondedForcePtr->setForce(m_BondedForceValues);
  m_BondedForcePtr->setStream(m_BondedStream);

  // Reciprocal
  m_ReciprocalStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_ReciprocalStream.get()));

  m_ReciprocalForceValues = std::make_shared<Force<long long int>>();
  m_ReciprocalForceValues->realloc(numAtoms, 1.5f);

  m_ReciprocalForcePtr =
      std::make_unique<CudaPMEReciprocalForce>(m_ReciprocalEnergyVirial);
  m_ReciprocalForcePtr->setPBC(m_Pbc);
  m_ReciprocalForcePtr->setParameters(m_NfftX, m_NfftY, m_NfftZ,
                                      m_PmeSplineOrder, m_Kappa,
                                      *m_ReciprocalStream);
  m_ReciprocalForcePtr->setNumAtoms(numAtoms);
  m_ReciprocalForcePtr->setBoxDimensions(m_BoxDimensions);
  m_ReciprocalForcePtr->setForce(m_ReciprocalForceValues);
  m_ReciprocalForcePtr->setStream(m_ReciprocalStream);

  // Direct
  m_DirectStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_DirectStream.get()));

  m_DirectForceValues = std::make_shared<Force<long long int>>();
  m_DirectForceValues->realloc(numAtoms, 1.5f);

  auto iblo14 = m_Psf->getIblo14();
  auto inb14 = m_Psf->getInb14();
  auto vdwParamsAndTypes = m_Prm->getVdwParamsAndTypes(m_Psf);
  auto inExLists = m_Psf->getInclusionExclusionLists();

  m_DirectForcePtr = std::make_unique<CudaPMEDirectForce<long long int, float>>(
      m_DirectEnergyVirial, "vdw", "elec", "ewex");
  const bool q_p21 = (m_Pbc == PBC::P21);
  // TODO this seems to do the job twice ?
  // 1. "directForcePtr->setup(boxx, kappa, ctofnb, (...) );"
  // 2. "directForcePtr->setBoxDimensions({boxx...});"
  m_DirectForcePtr->setup(m_BoxX, m_BoxY, m_BoxZ, m_Kappa, m_Ctofnb, m_Ctonnb,
                          1.0, m_VdwType, EWALD, q_p21);
  m_DirectForcePtr->setBoxDimensions(m_BoxDimensions);
  m_DirectForcePtr->setStream(m_DirectStream);
  m_DirectForcePtr->setForce(m_DirectForceValues);
  m_DirectForcePtr->setNumAtoms(numAtoms);
  m_DirectForcePtr->setCutoff(m_Cutoff);
  m_DirectForcePtr->setupSorted(numAtoms);
  m_DirectForcePtr->setupTopologicalExclusions(numAtoms, iblo14, inb14);
  m_DirectForcePtr->setupNeighborList(numAtoms);
  m_DirectForcePtr->set_vdwparam(vdwParamsAndTypes.vdwParams);
  m_DirectForcePtr->set_vdwparam14(vdwParamsAndTypes.vdw14Params);
  m_DirectForcePtr->set_vdwtype(vdwParamsAndTypes.vdwTypes);
  m_DirectForcePtr->set_vdwtype14(vdwParamsAndTypes.vdw14Types);
  m_DirectForcePtr->set_14_list(inExLists.sizes, inExLists.in14_ex14);

  // Initialize any forces that are already subscribed
  for (ForceView &forceView : m_ForceViews) {
    forceView.initialize(numAtoms, {static_cast<double>(m_BoxX),
                                    static_cast<double>(m_BoxY),
                                    static_cast<double>(m_BoxZ)});
  }

  m_ForceManagerStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_ForceManagerStream.get()));

  m_TotalForceValues = std::make_shared<Force<double>>();
  m_TotalForceValues->realloc(numAtoms, 1.5f);

  cudaCheck(cudaDeviceSynchronize());
  m_TotalPotentialEnergy.resize(1); // doing it for diffave and difflc; for Now

  this->initializeHolonomicConstraintsVariables();

  m_IsInitialized = true;

  return;
}

void ForceManager::resetNeighborList(const float4 *xyzq) {
  m_DirectForcePtr->resetNeighborList(xyzq, m_Psf->getNumAtoms());
  return;
}

void ForceManager::calcForcePart1(const float4 *xyzq, const bool reset,
                                  const bool calcEnergy,
                                  const bool calcVirial) {

  if (reset) {
    /* FOR FUTURE USE
    // forces_[1].resetNeighborList(xyzq, numAtoms, directStream);
    // throw std::invalid_argument("calc_force reset option not implemented\n");
    */
  } else {
    // updateSortedKernel();
  }

  if (!m_ClearGraphCreated) {
    cudaCheck(cudaStreamBeginCapture(*m_ForceManagerStream,
                                     cudaStreamCaptureModeGlobal));
    m_BondedForceValues->clear(*m_ForceManagerStream);
    m_ReciprocalForceValues->clear(*m_ForceManagerStream);
    m_DirectForceValues->clear(*m_ForceManagerStream);
    cudaCheck(cudaStreamEndCapture(*m_ForceManagerStream, &m_ClearGraph));
    cudaCheck(cudaGraphInstantiate(&m_CleargraphInstance, m_ClearGraph, NULL,
                                   NULL, 0));
    m_ClearGraphCreated = true;
  }
  cudaCheck(cudaGraphLaunch(m_CleargraphInstance, *m_ForceManagerStream));

  // Clear the virials and energy
  if ((calcEnergy == true) && (calcVirial == true)) {
    m_BondedEnergyVirial.clear(*m_BondedStream);
    m_ReciprocalEnergyVirial.clear(*m_ReciprocalStream);
    m_DirectEnergyVirial.clear(*m_DirectStream);
  } else if (calcEnergy == true) {
    m_BondedEnergyVirial.clearEnergy(*m_BondedStream);
    m_ReciprocalEnergyVirial.clearEnergy(*m_ReciprocalStream);
    m_DirectEnergyVirial.clearEnergy(*m_DirectStream);
  } else if (calcVirial == true) {
    m_BondedEnergyVirial.clearVirial(*m_BondedStream);
    m_ReciprocalEnergyVirial.clearVirial(*m_ReciprocalStream);
    m_DirectEnergyVirial.clearVirial(*m_DirectStream);
  }

  for (ForceView &forceView : m_ForceViews)
    forceView.clear();

  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));

  return;
}

void ForceManager::calcForcePart2(const float4 *xyzq, const bool reset,
                                  const bool calcEnergy,
                                  const bool calcVirial) {
  gpu_range_start("bonded");
  m_BondedForcePtr->calc_force(xyzq, calcEnergy, calcVirial);
  gpu_range_stop();

  gpu_range_start("reciprocal");
  m_ReciprocalForcePtr->calc_force(xyzq, calcEnergy, calcVirial);
  gpu_range_stop();

  gpu_range_start("direct");
  m_DirectForcePtr->calc_force(xyzq, calcEnergy, calcVirial);
  gpu_range_stop();

  for (ForceView &forceView : m_ForceViews)
    forceView.calcForce(xyzq, calcEnergy, calcVirial);

  return;
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
__global__ void UpdatePotentialEnergyKernel(
    double *__restrict__ pe, const double *__restrict__ e0,
    const double *__restrict__ e1, const double *__restrict__ e2,
    const double *__restrict__ e3, const double *__restrict__ e4,
    const double *__restrict__ e5, const double *__restrict__ e6,
    const double *__restrict__ e7, const double *__restrict__ e8,
    const double *__restrict__ e9) {
  if (threadIdx.x == 0) {
    pe[0] = e0[0] + e1[0] + e2[0] + e3[0] + e4[0] + e5[0] + e6[0] + e7[0] +
            e8[0] + e9[0];
  }
  return;
}

__global__ static void
UpdatePotentialEnergyKernel2(double *__restrict__ pe,
                             const double *__restrict__ en) {
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
      (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
    *pe += *en;
  return;
}

void ForceManager::calcForcePart3(const float4 *xyzq, const bool reset,
                                  const bool calcEnergy,
                                  const bool calcVirial) {
  m_TotalForceValues->clear(*m_ForceManagerStream);

  cudaCheck(cudaStreamSynchronize(*m_BondedStream));
  m_TotalForceValues->add<double>(*m_BondedForceValues, *m_ForceManagerStream);

  cudaCheck(cudaStreamSynchronize(*m_ReciprocalStream));
  m_TotalForceValues->add<double>(*m_ReciprocalForceValues,
                                  *m_ForceManagerStream);

  cudaCheck(cudaStreamSynchronize(*m_DirectStream));
  m_TotalForceValues->add<double>(*m_DirectForceValues, *m_ForceManagerStream);

  for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
    cudaCheck(cudaStreamSynchronize(*m_ForceStreams[i]));
    m_TotalForceValues->add<double>(*m_ForceViews[i].getForce(),
                                    *m_ForceManagerStream);
  }

  // TODO : find a better way.
  // For now, as virial requires forces to be double
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));

  if (calcVirial) {
    // Are we not computing virial twice ? I thought
    // directForcePtr->calc_force(xyzq, calcEnergy, calcVirial) would already
    // do it
    m_BondedForceValues->convert<double>(*m_BondedStream);
    m_ReciprocalForceValues->convert<double>(*m_ReciprocalStream);
    m_DirectForceValues->convert<double>(*m_DirectStream);
    for (std::size_t i = 0; i < m_ForceViews.size(); i++)
      m_ForceValues[i]->convert<double>(*m_ForceStreams[i]);

    cudaCheck(cudaStreamSynchronize(*m_BondedStream));
    cudaCheck(cudaStreamSynchronize(*m_ReciprocalStream));
    cudaCheck(cudaStreamSynchronize(*m_DirectStream));
    for (std::size_t i = 0; i < m_ForceViews.size(); i++)
      cudaCheck(cudaStreamSynchronize(*m_ForceStreams[i]));

    const int numAtoms = m_Psf->getNumAtoms();

    m_BondedEnergyVirial.calcVirial(
        numAtoms, xyzq, m_BoxDimensions[0], m_BoxDimensions[1],
        m_BoxDimensions[2], this->getForceStride(),
        reinterpret_cast<double *>(m_BondedForceValues->xyz()),
        *m_BondedStream);
    // Reciprocal space virial has already been calculated in the scalar_sum
    // m_ReciprocalEnergyVirial.calcVirial(
    //     numAtoms, xyzq, m_BoxDimensions[0], m_BoxDimensions[1],
    //     m_BoxDimensions[2], this->getForceStride(),
    //     reinterpret_cast<double *>(m_ReciprocalForceValues->xyz()),
    //     *m_ReciprocalStream);
    m_DirectEnergyVirial.calcVirial(
        numAtoms, xyzq, m_BoxDimensions[0], m_BoxDimensions[1],
        m_BoxDimensions[2], this->getForceStride(),
        reinterpret_cast<double *>(m_DirectForceValues->xyz()),
        *m_DirectStream);
    for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
      m_EnergyVirials[i]->calcVirial(
          numAtoms, xyzq, m_BoxDimensions[0], m_BoxDimensions[1],
          m_BoxDimensions[2], this->getForceStride(),
          reinterpret_cast<double *>(m_ForceValues[i]->xyz()),
          *m_ForceStreams[i]);
    }

    cudaCheck(cudaStreamSynchronize(*m_BondedStream));
    cudaCheck(cudaStreamSynchronize(*m_ReciprocalStream));
    cudaCheck(cudaStreamSynchronize(*m_DirectStream));
    for (std::size_t i = 0; i < m_ForceViews.size(); i++)
      cudaCheck(cudaStreamSynchronize(*m_ForceStreams[i]));
  }

  // Copy everything (all EnergyVirials) to Host, add together
  if (calcEnergy) {
    m_BondedEnergyVirial.copyToHost();
    m_ReciprocalEnergyVirial.copyToHost();
    m_DirectEnergyVirial.copyToHost();
    for (std::size_t i = 0; i < m_ForceViews.size(); i++)
      m_EnergyVirials[i]->copyToHost();
    cudaCheck(cudaDeviceSynchronize());

    // totalBondedEnergy = bondedEnergyVirial.getEnergy("bond") +
    //                     bondedEnergyVirial.getEnergy("angle") +
    //                     bondedEnergyVirial.getEnergy("ureyb") +
    //                     bondedEnergyVirial.getEnergy("dihe") +
    //                     bondedEnergyVirial.getEnergy("imdihe");

    // totalNonBondedEnergy = directEnergyVirial.getEnergy("ewex") +
    //                        directEnergyVirial.getEnergy("elec") +
    //                        directEnergyVirial.getEnergy("vdw") +
    //                        reciprocalEnergyVirial.getEnergy("ewks") +
    //                        reciprocalEnergyVirial.getEnergy("ewse");

    UpdatePotentialEnergyKernel<<<1, 32, 0, *m_ForceManagerStream>>>(
        m_TotalPotentialEnergy.getDeviceArray().data(),
        m_BondedEnergyVirial.getEnergyPointer("bond"),
        m_BondedEnergyVirial.getEnergyPointer("angle"),
        m_BondedEnergyVirial.getEnergyPointer("ureyb"),
        m_BondedEnergyVirial.getEnergyPointer("dihe"),
        m_BondedEnergyVirial.getEnergyPointer("imdihe"),
        m_ReciprocalEnergyVirial.getEnergyPointer("ewks"),
        m_ReciprocalEnergyVirial.getEnergyPointer("ewse"),
        m_DirectEnergyVirial.getEnergyPointer("ewex"),
        m_DirectEnergyVirial.getEnergyPointer("elec"),
        m_DirectEnergyVirial.getEnergyPointer("vdw"));

    // JEG260514: For now, we are assuming that all added forces only have a
    // single energy component. This CudaEnergyVirial interface is not very
    // flexibile. Should be overhauled at some point.
    for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
      UpdatePotentialEnergyKernel2<<<1, 32, 0, *m_ForceManagerStream>>>(
          m_EnergyVirials[i]->getEnergyPointer(),
          m_TotalPotentialEnergy.getDeviceArray().data());
    }

    cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));

    if (m_PrintEnergyDecomposition) {
      std::cout << "bond energy         : "
                << m_BondedEnergyVirial.getEnergy("bond") << "\n";
      std::cout << "angle energy        : "
                << m_BondedEnergyVirial.getEnergy("angle") << "\n";
      std::cout << "ureyb energy        : "
                << m_BondedEnergyVirial.getEnergy("ureyb") << "\n";
      std::cout << "dihe energy         : "
                << m_BondedEnergyVirial.getEnergy("dihe") << "\n";
      std::cout << "imdihe energy       : "
                << m_BondedEnergyVirial.getEnergy("imdihe") << "\n";

      std::cout << "recip kspace energy : "
                << m_ReciprocalEnergyVirial.getEnergy("ewks") << "\n";
      std::cout << "recip  self energy  : "
                << m_ReciprocalEnergyVirial.getEnergy("ewse") << "\n";

      std::cout << "ewex energy         : "
                << m_DirectEnergyVirial.getEnergy("ewex") << "\n";
      std::cout << "elec energy         : "
                << m_DirectEnergyVirial.getEnergy("elec") << "\n";
      std::cout << "vdw energy          : "
                << m_DirectEnergyVirial.getEnergy("vdw") << "\n";

      for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
        std::cout << m_ForceTags[i] << ": " << m_EnergyVirials[i]->getEnergy()
                  << "\n";
      }

      m_TotalPotentialEnergy.transferToHost();

      std::cout << "Total potential energy : " << m_TotalPotentialEnergy[0]
                << std::endl;
    }
  }

  return;
}

void ForceManager::calcForce(const float4 *xyzq, bool reset, bool calcEnergy,
                             bool calcVirial) {
  this->calcForcePart1(xyzq, reset, calcEnergy, calcVirial);
  this->calcForcePart2(xyzq, reset, calcEnergy, calcVirial);
  this->calcForcePart3(xyzq, reset, calcEnergy, calcVirial);
  return;
}

CudaContainer<double>
ForceManager::computeAllChildrenPotentialEnergy(const float4 *xyzq) {
  throw std::invalid_argument(
      "ERROR: computeAllChildrenPotential cannot be called from ForceManager");
  return;
}

inline bool isHydrogen(const std::string &atomType) {
  return (atomType[0] == 'H');
}

void ForceManager::initializeHolonomicConstraintsVariables(void) {
  const int numAtoms = m_Psf->getNumAtoms();

  const auto &bonds = m_Psf->getBonds();
  const auto &atomNames = m_Psf->getAtomNames();
  const auto &atomTypes = m_Psf->getAtomTypes();
  const auto &atomMasses = m_Psf->getMasses();

  std::vector<int> numBondsH(numAtoms, 0);
  std::vector<std::vector<int>> hydrogenBonds(numAtoms);

  std::vector<int4> shakeAtoms;
  std::vector<float4> shakeParams;
  auto bondParams = m_Prm->getBondParams();

  for (const auto &bond : bonds) {
    // TODO : refine these selection criteria
    if (isHydrogen(atomTypes[bond.iatom]) ||
        isHydrogen(atomTypes[bond.jatom])) {
      if (!((atomTypes[bond.iatom] == "OT" and atomTypes[bond.jatom] == "HT") ||
            (atomTypes[bond.iatom] == "HT" and atomTypes[bond.jatom] == "OT") ||
            (atomTypes[bond.iatom][0] == 'H' and
             atomTypes[bond.jatom][0] == 'H'))) {
        int heavyAtom = -1, hydrogenAtom = -1;
        if (isHydrogen(atomTypes[bond.iatom])) {
          heavyAtom = bond.jatom;
          hydrogenAtom = bond.iatom;
        } else {
          heavyAtom = bond.iatom;
          hydrogenAtom = bond.jatom;
        }
        numBondsH[heavyAtom]++;
        hydrogenBonds[heavyAtom].push_back(hydrogenAtom);
      }
    }
  }

  for (int i = 0; i < numAtoms; i++) {
    if (numBondsH[i]) {
      std::vector<int> group;
      group = {i, -1, -1, -1};
      float totalMass = static_cast<float>(atomMasses[i]);
      float hydrogenMass = 0.0f;
      BondValues bondValue;
      for (std::size_t j = 0; j < hydrogenBonds[i].size(); j++) {
        int hyd = hydrogenBonds[i][j];
        group[j + 1] = hyd;
        totalMass += atomMasses[j];
        hydrogenMass = atomMasses[hyd];

        std::string atomType0 = "", atomType1 = "";
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

      float avgMass =
          totalMass / static_cast<float>(hydrogenBonds[i].size() + 1);
      shakeAtoms.push_back({group[0], group[1], group[2], group[3]});

      float4 p = make_float4(1.0f / static_cast<float>(atomMasses[i]), avgMass,
                             bondValue.b0 * bondValue.b0, 1.0f / hydrogenMass);
      shakeParams.push_back(p);
    }
  }

  m_ShakeAtoms = shakeAtoms;
  m_ShakeParams = shakeParams;

  return;
}

std::vector<int> ForceManager::computeFFTGridSize(void) {
  this->checkBoxDimensions(m_BoxDimensions);
  int fx = 2 * (static_cast<int>(m_BoxX) / 2);
  int fy = 2 * (static_cast<int>(m_BoxY) / 2);
  int fz = 2 * (static_cast<int>(m_BoxZ) / 2);
  if (fx < 2) {
    fx = 2;
    std::cout << "Warning: boxx seems very small (" << m_BoxX
              << "), setting associated fft grid size to2\n";
  }
  if (fy < 2) {
    fy = 2;
    std::cout << "Warning: boxy seems very small (" << m_BoxY
              << "), setting associated fft grid size to2\n";
  }
  if (fz < 2) {
    fz = 2;
    std::cout << "Warning: boxz seems very small (" << m_BoxZ
              << "), setting associated fft grid size to2\n";
  }
  return {fx, fy, fz};
}

void ForceManager::checkBoxDimensions(const std::vector<double> &size) {
  for (auto dim : size) {
    if (dim < 0) {
      throw std::invalid_argument("Box dimensions: " + std::to_string(size[0]) +
                                  " x " + std::to_string(size[1]) + " x " +
                                  std::to_string(size[2]) + " are NOT valid");
    }
  }
  return;
}

void ForceManager::dealloc(void) {
  if (m_BondedStream != nullptr) {
    cudaCheck(cudaStreamDestroy(*m_BondedStream));
    m_BondedStream.reset();
  }
  if (m_ReciprocalStream != nullptr) {
    cudaCheck(cudaStreamDestroy(*m_ReciprocalStream));
    m_ReciprocalStream.reset();
  }
  if (m_DirectStream != nullptr) {
    cudaCheck(cudaStreamDestroy(*m_DirectStream));
    m_DirectStream.reset();
  }
  if (m_ForceManagerStream != nullptr) {
    cudaCheck(cudaStreamDestroy(*m_ForceManagerStream));
    m_ForceManagerStream.reset();
  }
  return;
}
