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
#include "CharmmCrd.h"
#include "Checkpoint.h"
#include "Constants.h"
#include "PBC.h"
#include "gpu_utils.h"
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

CharmmContext::CharmmContext(std::shared_ptr<ForceManager> fmIn)
    : forceManager(fmIn) {

  std::vector<int> devices = {0, 1, 2, 3};
  start_gpu(1, 1, 0, devices);

  useHolonomicConstraints(true);

  kineticEnergy.resize(1);
  virialKineticEnergyTensor.resize(9);

  if (!forceManager->isInitialized()) {
    forceManager->initialize();
  }
  pressure.resize(9);

  // linkBackForceManager();

  hasLogger = false;
  std::random_device rd{};
  seed = rd();
}

// Copy Constructor . Does not copy CharmmContext.
CharmmContext::CharmmContext(const CharmmContext &ctxIn)
    : numAtoms(ctxIn.numAtoms), xyzq(ctxIn.xyzq),
      coordsCharge(ctxIn.coordsCharge), velocityMass(ctxIn.velocityMass),
      kineticEnergy(ctxIn.kineticEnergy), pressure(ctxIn.pressure),
      temperature(ctxIn.temperature),
      numDegreesOfFreedom(ctxIn.numDegreesOfFreedom),
      usingHolonomicConstraints(ctxIn.usingHolonomicConstraints),
      hasLogger(ctxIn.hasLogger), seed(ctxIn.seed) {}

void CharmmContext::setupFromCheckpoint(
    std::shared_ptr<Checkpoint> checkpoint) {
  numAtoms = checkpoint->get<int>("numAtoms");
  // coordsCharge = checkpoint->get<CudaContainer<double4>>("coordsCharge");

  return;
}

void CharmmContext::setMasses(const std::vector<double> &masses) {
  if (masses.size() != numAtoms) {
    std::stringstream tmpexc;
    tmpexc << "Masses vector size does not match numAtoms (" << masses.size()
           << " != " << numAtoms << ")\n";
    throw std::invalid_argument(tmpexc.str());
  }
  assert(velocityMass.size() == numAtoms);
  for (int i = 0; i < numAtoms; i++) {
    velocityMass[i].w = 1.0 / masses[i];
  }
  velocityMass.transferToDevice();
}

void CharmmContext::setNumAtoms(const int num) { numAtoms = num; }

void CharmmContext::setCoordinates(const std::shared_ptr<Coordinates> crd) {
  // auto coords = crd->getCoordinates();
  this->setCoordinates(crd->getCoordinates());
  /* *
  if (!forceManager->isComposite()) {
    assert(coords.size() == forceManager->getPSF()->getNumAtoms());
  }
  if (!forceManager->hasCharmmContext()) {
    linkBackForceManager();
  }
  setNumAtoms(coords.size());

  velocityMass.allocate(numAtoms);
  setMasses(forceManager->getPSF()->getAtomMasses());

  useHolonomicConstraints(usingHolonomicConstraints);
  auto charges =
      forceManager->getPSF()->getAtomCharges(); // this is a std::vector<double>

  // 4N-sized vector to contain spatial coords + atomic charges
  std::vector<double4> crdCharges(coords.size());
  int i = 0;
  for (auto &&coord : coords) {
    coord.w = (float)charges[i]; // put the charges in coords ith element
    crdCharges[i] = {(double)coord.x, (double)coord.y, (double)coord.z,
                     (double)coord.w};
    i++;
  }

  // xyzq gets initialized with coords (crd->getCoordinates()) + the charges
  // extracted from the PSF
  xyzq.set_ncoord(coords.size());
  xyzq.set_xyzq(coords.size(), coords.data(), 0);

  coordsCharge.allocate(coords.size());
  coordsCharge.set(crdCharges);
  resetNeighborList();
  * */
}

void CharmmContext::setCoordinates(
    const std::vector<std::vector<double>> coords) {
  // Constructor for CharmmCrd casts to float4 which loses precision when
  // reading from restart file
  // 1. Create a charmmCrd using this vec{vec{double}}
  // 2. Call setCoordinates(charmmCrd) with that newly created crd
  // auto charmmCrd = std::make_shared<CharmmCrd>(crd);
  // setCoordinates(charmmCrd);
  if (!forceManager->isComposite()) {
    assert(coords.size() == forceManager->getPSF()->getNumAtoms());
  }
  if (!forceManager->hasCharmmContext()) {
    linkBackForceManager();
  }
  setNumAtoms(coords.size());

  velocityMass.resize(numAtoms);
  setMasses(forceManager->getPSF()->getAtomMasses());

  useHolonomicConstraints(usingHolonomicConstraints);
  auto charges =
      forceManager->getPSF()->getAtomCharges(); // this is a std::vector<double>

  // 4N-sized vector to contain spatial coords + atomic charges
  std::vector<double4> crdCharges(coords.size());
  for (int i = 0; i < numAtoms; i++)
    crdCharges[i] = {coords[i][0], coords[i][1], coords[i][2],
                     static_cast<double>(static_cast<float>(charges[i]))};

  // xyzq gets initialized with coords (crd->getCoordinates()) + the charges
  // extracted from the PSF
  xyzq.set_ncoord(coords.size());
  std::vector<float4> fcrds(numAtoms);
  for (int i = 0; i < numAtoms; i++)
    fcrds[i] = make_float4(crdCharges[i].x, crdCharges[i].y, crdCharges[i].z,
                           crdCharges[i].w);
  xyzq.set_xyzq(coords.size(), fcrds.data(), 0);

  coordsCharge.resize(coords.size());
  coordsCharge.set(crdCharges);
  resetNeighborList();
}

std::vector<std::vector<double>> CharmmContext::getCoordinates() {
  // auto coords = xyzq.get_xyz();
  coordsCharge.transferFromDevice();
  std::vector<std::vector<double>> coordsVec;
  for (int i = 0; i < numAtoms; i++) {
    std::vector<double> tmp = {coordsCharge[i].x, coordsCharge[i].y,
                               coordsCharge[i].z};
    coordsVec.push_back(tmp);
  }
  return coordsVec;
}

void CharmmContext::setCoordinatesNumpy(pybind11::array_t<double> input_array) {
  // Get shape and data type of array
  auto shape = input_array.shape();
  throw std::invalid_argument("setCoordinatesNumpy not implemented fully");
  /*std::string dtype = input_array.dtype().name();

  // Print shape and data type
  std::cout << "Array shape: (";
  for (size_t i = 0; i < shape.size(); i++) {
    std::cout << shape[i];
    if (i < shape.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << ")" << std::endl;
  std::cout << "Array data type: " << dtype << std::endl;
  */
}

void CharmmContext::setCoords(const std::vector<float> &coords) {
  assert(coords.size() == numAtoms * 3);

  xyzq.set_xyz(coords);
  resetNeighborList();
}

std::vector<float> CharmmContext::getCoords() { return xyzq.get_xyz(); }

int CharmmContext::getNumAtoms() const { return numAtoms; }

__global__ static void
imageCenterKernel(PBC pbc, float3 boxSize, int stride, int numGroups,
                  const int2 *__restrict__ groups, double4 *__restrict__ xyzq,
                  float4 *__restrict__ xyzqf, double4 *__restrict__ velMass,
                  double *__restrict__ force) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index < numGroups) {
    int2 group = groups[index];
    float gx = 0.0f;
    float gy = 0.0f;
    float gz = 0.0f;

    for (int i = group.x; i <= group.y; ++i) {
      gx += xyzqf[i].x;
      gy += xyzqf[i].y;
      gz += xyzqf[i].z;
    }

    gx /= (group.y - group.x + 1);
    gy /= (group.y - group.x + 1);
    gz /= (group.y - group.x + 1);

    if (gx < -0.5 * boxSize.x) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].x += boxSize.x;
        xyzqf[i].x += boxSize.x;
        if (pbc == PBC::P21) {
          xyzq[i].y = -xyzq[i].y;
          xyzqf[i].y = -xyzqf[i].y;
          xyzq[i].z = -xyzq[i].z;
          xyzqf[i].z = -xyzqf[i].z;

          velMass[i].y = -velMass[i].y;
          velMass[i].z = -velMass[i].z;

          force[stride + i] = -force[stride + i];
          force[2 * stride + i] = -force[2 * stride + i];

          gy = -gy;
          gz = -gz;
        }
      }
    }

    if (gx > 0.5 * boxSize.x) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].x -= boxSize.x;
        xyzqf[i].x -= boxSize.x;
        if (pbc == PBC::P21) {
          xyzq[i].y = -xyzq[i].y;
          xyzqf[i].y = -xyzqf[i].y;
          xyzq[i].z = -xyzq[i].z;
          xyzqf[i].z = -xyzqf[i].z;

          velMass[i].y = -velMass[i].y;
          velMass[i].z = -velMass[i].z;

          force[stride + i] = -force[stride + i];
          force[2 * stride + i] = -force[2 * stride + i];

          gy = -gy;
          gz = -gz;
        }
      }
    }

    if (gy < -0.5 * boxSize.y) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].y += boxSize.y;
        xyzqf[i].y += boxSize.y;
      }
    }

    if (gy > 0.5 * boxSize.y) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].y -= boxSize.y;
        xyzqf[i].y -= boxSize.y;
      }
    }

    if (gz < -0.5 * boxSize.z) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].z += boxSize.z;
        xyzqf[i].z += boxSize.z;
      }
    }

    if (gz > 0.5 * boxSize.z) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].z -= boxSize.z;
        xyzqf[i].z -= boxSize.z;
      }
    }
  }
}

void CharmmContext::imageCentering() {
  auto boxSize = forceManager->getBoxDimensions();
  double boxx = boxSize[0];
  double boxy = boxSize[1];
  double boxz = boxSize[2];
  auto pbc = forceManager->getPeriodicBoundaryCondition();

  auto groups = forceManager->getPSF()->getGroups();

  auto force = getForces();
  auto forceStride = getForceStride();

  // find a better place for this
  int numGroups = groups.size();
  int numThreads = 128;
  int numBlocks = (numGroups - 1) / numThreads + 1;

  float3 box = {(float)boxSize[0], (float)boxSize[1], (float)boxSize[2]};

  // forces do not necessarily need to be inverted for the case pf o21 since
  // they will be updated soon during energy/force calculation

  imageCenterKernel<<<numBlocks, numThreads>>>(
      pbc, box, forceStride, numGroups, groups.getDeviceArray().data(),
      coordsCharge.getDeviceArray().data(), xyzq.xyzq,
      velocityMass.getDeviceArray().data(), force->xyz());
  cudaCheck(cudaDeviceSynchronize());
}

void CharmmContext::resetNeighborList() {
  imageCentering();
  forceManager->resetNeighborList(xyzq.getDeviceXYZQ());
}

float CharmmContext::calculateForces(bool reset, bool calcEnergy,
                                     bool calcVirial) {
  return forceManager->calc_force(xyzq.getDeviceXYZQ(), reset, calcEnergy,
                                  calcVirial);
}

std::vector<float> CharmmContext::getPotentialEnergies() {
  return forceManager->getPotentialEnergies();
}

std::shared_ptr<Force<double>> CharmmContext::getForces() {
  return forceManager->getForces();
}

float CharmmContext::getTemperature() { return temperature; }

void CharmmContext::setTemperature(const float temp) { temperature = temp; }

__global__ void
calculateCenterOfMassMomemtumKernel(int numAtoms,
                                    double4 *__restrict__ velmass) {

  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // float4 com;
  // for (int i = idx; i < numAtoms; i += blockDim.x * gridDim.x) {
  //   auto mass = 1.0 / velmass[i].w;
  //   com.x = velmass[i].x * mass;
  //   com.y = velmass[i].y * mass;
  //   com.z = velmass[i].z * mass;
  // }
  return;
}

void CharmmContext::removeCenterOfMassMotion() {
  /*
  // TODO : do this in the kernel rather than the host side
  velocityMass.transferFromDevice();

  // Remove the center of mass velocity
  float3 com = {0.0, 0.0, 0.0};
  float totalMass = 0.0;
  for (int i = 0; i < numAtoms; ++i) {
    com.x += velocityMass[i].x / velocityMass[i].w;
    com.y += velocityMass[i].y / velocityMass[i].w;
    com.z += velocityMass[i].z / velocityMass[i].w;
    totalMass += 1 / velocityMass[i].w;
  }
  com.x /= totalMass;
  com.y /= totalMass;
  com.z /= totalMass;

  for (int i = 0; i < numAtoms; ++i) {
    velocityMass[i].x -= com.x;
    velocityMass[i].y -= com.y;
    velocityMass[i].z -= com.z;
  }

  velocityMass.transferToDevice();
  */
}

void CharmmContext::assignVelocitiesAtTemperature(float temp) {
  if (numAtoms == -1) {
    throw std::invalid_argument(
        "numAtoms = -1 in CharmmContext::assignVelocitiesAtTemperature -- This "
        "Context object was not initialized properly (no Coordinate given "
        "?).\n Make sure you used setCoordinates before trying to "
        "assignVelocities.");
  }
  setTemperature(temp);
  double boltz = charmm::constants::kBoltz * temperature;

  // std::random_device rd{};
  //  std::mt19937 gen{rd()};
  // std::cout << "seed = " << seed << std::endl;
  std::mt19937 gen{seed};

  for (int i = 0; i < numAtoms; i++) {
    double sd = boltz * velocityMass[i].w;
    sd = sqrt(sd);

    std::normal_distribution<> d(0, sd);

    velocityMass[i].x = d(gen);
    velocityMass[i].y = d(gen);
    velocityMass[i].z = d(gen);
  }

  velocityMass.transferToDevice();
  removeCenterOfMassMotion();

  float kineticEnergy = 0.0;
  for (int i = 0; i < numAtoms; ++i) {
    kineticEnergy += 1.0 / velocityMass[i].w *
                     (pow(velocityMass[i].x, 2) + pow(velocityMass[i].y, 2) +
                      pow(velocityMass[i].z, 2));
  }
  kineticEnergy *= 0.5;
  int ndegf = getDegreesOfFreedom();

  // std::cout << "Velocities assigned at temperature " << temp << "\n";

  // std::cout << "dof : " << ndegf << "\n";
  float backTemp =
      kineticEnergy / (1 / 2.0 * ndegf * charmm::constants::kBoltz);
  // std::cout << "calculated temp from ke (host) : " << backTemp << "\n";

  velocityMass.transferToDevice();
  // std::cout << "calculated temp from ke : " << computeTemperature() <<
  // "\n";
}

static std::vector<std::string> split(std::string line) {
  std::stringstream ss(line);
  std::string atomId, resId, resName, atom, x, y, z;
  ss >> atomId >> resId >> resName >> atom >> x >> y >> z;
  std::vector<std::string> content = {atomId, resId, resName, atom, x, y, z};

  return content;
}

void CharmmContext::assignVelocitiesFromCHARMMVelocityFile(
    std::string fileName) {
  std::ifstream fin(fileName);

  if (!fin.is_open()) {
    throw std::invalid_argument("Could not open CHARMM velocity file ");
    exit(0);
  }

  std::string line;
  //  comment lines
  while (1) {
    std::getline(fin, line);
    if (line[0] != '*')
      break;
  }

  int nAtoms = std::stoul(line);
  assert(nAtoms == numAtoms);

  int i = 0;

  std::getline(fin, line);
  while (i < numAtoms) {
    if (line.size() == 0) {
      // std::cerr << "ERROR: Blank line read in " << fileName << "\n.
      // Exiting\n";
      throw std::invalid_argument("ERROR: Blank line read in " + fileName +
                                  "\n. Exiting\n");
      exit(0);
    }
    auto content = split(line);
    int atomId, resId, resIdInSeg;
    std::string resName, atomName, segName;
    float x, y, z, bFactor;

    std::stringstream ss(line);
    ss >> atomId >> resId >> resName >> atomName >> x >> y >> z >> segName >>
        resIdInSeg >> bFactor;

    velocityMass[i].x = x;
    velocityMass[i].y = y;
    velocityMass[i].z = z;

    std::getline(fin, line);
    ++i;
  }

  float kineticEnergy = 0.0;
  for (int i = 0; i < numAtoms; ++i) {
    kineticEnergy += 1.0 / velocityMass[i].w *
                     (pow(velocityMass[i].x, 2) + pow(velocityMass[i].y, 2) +
                      pow(velocityMass[i].z, 2));
  }
  kineticEnergy *= 0.5;
  // int ndegf = numAtoms; // This is only for water molecules
  int ndegf = getDegreesOfFreedom();

  std::cout << "dof : " << ndegf << "\n";
  float backTemp =
      kineticEnergy / (1 / 2.0 * ndegf * charmm::constants::kBoltz);
  std::cout << "calculated temp from ke (host) : " << backTemp << "\n";

  velocityMass.transferToDevice();
  std::cout << "calculated temp from ke : " << computeTemperature() << "\n";
}

void CharmmContext::assignVelocities(const std::vector<double> velIn) {
  assert(velIn.size() == numAtoms * 3);
  for (int i = 0; i < numAtoms; ++i) {
    velocityMass[i].x = velIn[i * 3];
    velocityMass[i].y = velIn[i * 3 + 1];
    velocityMass[i].z = velIn[i * 3 + 2];
  }
  velocityMass.transferToDevice();
}

void CharmmContext::assignVelocities(
    const std::vector<std::vector<double>> velIn) {
  assert(velIn.size() == numAtoms);
  for (int i = 0; i < numAtoms; ++i) {
    assert(velIn[i].size() == 3);
    velocityMass[i].x = velIn[i][0];
    velocityMass[i].y = velIn[i][1];
    velocityMass[i].z = velIn[i][2];
  }
  velocityMass.transferToDevice();
}

CudaContainer<double4> &CharmmContext::getVelocityMass() {
  return velocityMass;
}

CudaContainer<double4> &CharmmContext::getCoordinatesCharges() {
  return coordsCharge;
}

XYZQ *CharmmContext::getXYZQ() { return &xyzq; }

int CharmmContext::getForceStride() const {
  return forceManager->getForceStride();
}

// Cuda Kernel to compute kinetic energy
__global__ void
calculateKineticEnergyKernel(int numAtoms, const double4 *__restrict__ velMass,
                             double *__restrict__ kineticEnergy) {
  constexpr int blockSize = 128;
  __shared__ double sdata[blockSize];
  int threadId = threadIdx.x;
  sdata[threadId] = 0.0;
  size_t index = threadIdx.x + blockDim.x * blockIdx.x;

  if (index == 0)
    kineticEnergy[0] = 0.0;

  while (index < numAtoms) {
    sdata[threadId] += 0.5 *
                       ((velMass[index].x * velMass[index].x) +
                        (velMass[index].y * velMass[index].y) +
                        (velMass[index].z * velMass[index].z)) /
                       velMass[index].w;
    index += gridDim.x * blockDim.x; // grid stride to load data
  }

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (threadId < s) // parallel sweep reduction
      sdata[threadId] += sdata[threadId + s];
  }

  if (threadId == 0)
    atomicAdd(kineticEnergy, sdata[0]);
}

void CharmmContext::calculateKineticEnergy() {
  int numThreads = 128; // this is the blockSize
  int numBlocks = 64;

  calculateKineticEnergyKernel<<<numBlocks, numThreads>>>(
      numAtoms, velocityMass.getDeviceArray().data(),
      kineticEnergy.getDeviceArray().data());

  cudaCheck(cudaDeviceSynchronize()); // TODO fix this
}

double CharmmContext::getKineticEnergy() {
  calculateKineticEnergy();
  kineticEnergy.transferFromDevice();
  return kineticEnergy.getHostArray()[0];
}

float CharmmContext::computeTemperature() {
  // auto kineticEnergy = calculateKineticEnergy();
  if (numAtoms == -1 || velocityMass.size() == 0) {
    throw std::invalid_argument(
        "No atoms in the system -- coordinates have not been loaded and/or "
        "velocities not assigned.");
  }
  auto ke = getKineticEnergy();
  int numDegreesOfFreedom = getDegreesOfFreedom();
  return ke / (0.5 * numDegreesOfFreedom * charmm::constants::kBoltz);
}

void CharmmContext::setPeriodicBoundaryCondition(const PBC _pbc) {
  forceManager->setPeriodicBoundaryCondition(_pbc);
  resetNeighborList();
}

PBC CharmmContext::getPeriodicBoundaryCondition() {
  return forceManager->getPeriodicBoundaryCondition();
}

const std::vector<double> &CharmmContext::getBoxDimensions(void) const {
  return forceManager->getBoxDimensions();
}

std::vector<double> &CharmmContext::getBoxDimensions(void) {
  return forceManager->getBoxDimensions();
}

void CharmmContext::setBoxDimensions(const std::vector<double> &boxDimensions) {
  forceManager->setBoxDimensions(boxDimensions);
}

std::vector<Bond> CharmmContext::getBonds() { return forceManager->getBonds(); }

int CharmmContext::getDegreesOfFreedom() { return numDegreesOfFreedom; }

float CharmmContext::calculatePotentialEnergy(bool reset, bool print) {
  return forceManager->calc_force(xyzq.getDeviceXYZQ(), reset, true, true);
}
CudaContainer<double> &CharmmContext::getPotentialEnergy() {
  return forceManager->getPotentialEnergy();
}

double CharmmContext::getVolume() const {
  auto boxSize = forceManager->getBoxDimensions();
  double boxx = boxSize[0];
  double boxy = boxSize[1];
  double boxz = boxSize[2];

  return boxx * boxy * boxz;
}

// calculate Kinetic component to the pressure
static __global__ void
calculateKineticKernel(int numAtoms, const double4 *__restrict__ velMass,
                       double *accumulant) {
  // TODO : convert this to a logn summation like KE in charmmcontext
  constexpr int blockSize = 128 * 9;
  __shared__ double sdata[blockSize];
  int threadId = threadIdx.x;

  for (int i = 0; i < 9; ++i)
    sdata[9 * threadId + i] = 0.0;

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = index; i < numAtoms; i += blockDim.x)
    if (index < numAtoms) {
      double rvc = 0.5 / velMass[index].w;
      sdata[threadId] = rvc * velMass[i].x * velMass[i].x;
      sdata[threadId + 1] = rvc * velMass[i].x * velMass[i].y;
      sdata[threadId + 2] = rvc * velMass[i].x * velMass[i].z;
      sdata[threadId + 3] = rvc * velMass[i].y * velMass[i].x;
      sdata[threadId + 4] = rvc * velMass[i].y * velMass[i].y;
      sdata[threadId + 5] = rvc * velMass[i].y * velMass[i].z;
      sdata[threadId + 6] = rvc * velMass[i].z * velMass[i].x;
      sdata[threadId + 7] = rvc * velMass[i].z * velMass[i].y;
      sdata[threadId + 8] = rvc * velMass[i].z * velMass[i].z;
    }

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (threadId < s) {
      sdata[threadId] += sdata[threadId + s * 9];
      sdata[threadId + 1] += sdata[threadId + s * 9 + 1];
      sdata[threadId + 2] += sdata[threadId + s * 9 + 2];
      sdata[threadId + 3] += sdata[threadId + s * 9 + 3];
      sdata[threadId + 4] += sdata[threadId + s * 9 + 4];
      sdata[threadId + 5] += sdata[threadId + s * 9 + 5];
      sdata[threadId + 6] += sdata[threadId + s * 9 + 6];
      sdata[threadId + 7] += sdata[threadId + s * 9 + 7];
      sdata[threadId + 8] += sdata[threadId + s * 9 + 8];
    }
  }

  if (threadId == 0) {
    atomicAdd(&accumulant[0], sdata[0]);
    atomicAdd(&accumulant[1], sdata[1]);
    atomicAdd(&accumulant[2], sdata[2]);
    atomicAdd(&accumulant[3], sdata[3]);
    atomicAdd(&accumulant[4], sdata[4]);
    atomicAdd(&accumulant[5], sdata[5]);
    atomicAdd(&accumulant[6], sdata[6]);
    atomicAdd(&accumulant[7], sdata[7]);
    atomicAdd(&accumulant[8], sdata[8]);
  }
}

void CharmmContext::computePressure() {
  auto ke = getKineticEnergy();
  int numBlocks = 64;
  int numThreads = 128;

  // TODO : put this a separate stream
  // TODO: the kinetic component computed here is wrong (accumulates over
  // time.)
  calculateKineticKernel<<<numBlocks, numThreads>>>(
      numAtoms, velocityMass.getDeviceArray().data(),
      virialKineticEnergyTensor.getDeviceArray().data());

  cudaCheck(cudaDeviceSynchronize());

  auto vcell = charmm::constants::patmos / getVolume();
  // std::cout << "[CTX] vcell : " << vcell << "\n";

  // TODO : calculate the non-diagonal entries

  // get the virial from force manager
  auto virial = forceManager->getVirial();
  virial.transferFromDevice();

  virialKineticEnergyTensor.transferFromDevice();

  for (int i = 0; i < 9; ++i) {
    pressure[i] = (2.0 * virialKineticEnergyTensor.getHostArray()[i] +
                   virial.getHostArray()[i]) /
                  vcell;
  }

  /* Debug printing
    std::cout << "-------------------------------" << std::endl;
    std::cout << "[CTX] Pressure" << std::endl;
    int k = 0;
    for (int i = 0; i < 3; i++) {
      std::cout << "[CTX] ";
      for (int j = 0; j < 3; j++)
        std::cout <<  pressure[k++] << " ";
      std::cout << "\n";
    }
    std::cout << "[CTX] virialKineticEnergyTensor " << std::endl;
    k=0;
    for (int i = 0; i < 3; i++) {
      std::cout << "[CTX] ";
      for (int j = 0; j < 3; j++)
        std::cout <<  virialKineticEnergyTensor.getHostArray()[k++] << " ";
      std::cout << "\n";
    }

    std::cout << "[CTX] virial " << std::endl;
    k=0;
    for (int i = 0; i < 3; i++) {
      std::cout << "[CTX] ";
      for (int j = 0; j < 3; j++)
        std::cout <<  virial.getHostArray()[k++] << " ";
      std::cout << "\n";
    }
    std::cout << "[CTX] vcell: " << vcell << "\n";
  */

  pressure.transferToDevice();
}

CudaContainer<double> CharmmContext::getVirial() {
  return forceManager->getVirial();
}

CudaContainer<int4> CharmmContext::getWaterMolecules() {
  auto waterMolecules = forceManager->getPSF()->getWaterMolecules();
  return waterMolecules;
}

CudaContainer<int4> CharmmContext::getShakeAtoms() {
  return forceManager->getShakeAtoms();
}

CudaContainer<float4> CharmmContext::getShakeParams() {
  return forceManager->getShakeParams();
}

void CharmmContext::useHolonomicConstraints(bool set) {
  usingHolonomicConstraints = set;
  int ndegf = numAtoms * 3;
  auto pbc = forceManager->getPeriodicBoundaryCondition();
  if (pbc == PBC::P1) {
    ndegf -= 3;
  } else if (pbc == PBC::P21) {
    ndegf -= 1;
  } else {
  }

  if (usingHolonomicConstraints) {
    ndegf -= getWaterMolecules().size() * 3;
    int numShakeConstraints = 0;
    auto shakeAtoms = forceManager->getShakeAtoms().getHostArray();
    for (int i = 0; i < shakeAtoms.size(); ++i) {
      ++numShakeConstraints;
      if (shakeAtoms[i].z != -1)
        ++numShakeConstraints;
      if (shakeAtoms[i].w != -1)
        ++numShakeConstraints;
    }
    ndegf -= numShakeConstraints;
  }

  // TODO : other restraints need to be accounted for as well
  numDegreesOfFreedom = ndegf;
}

void CharmmContext::orient() {}

void CharmmContext::setForceManager(std::shared_ptr<ForceManager> fm) {
  forceManager = fm;
}

void CharmmContext::linkBackForceManager() {
  forceManager->setCharmmContext(shared_from_this());
}

void CharmmContext::writeCrd(std::string fileName) {

  std::ofstream fout(fileName);

  if (!fout.is_open()) {
    throw std::invalid_argument("ERROR! Can't open the crd file to write \n");
  }
}

void CharmmContext::readRestart(std::string fileName) {
  std::ifstream restartFile(fileName);
  std::string line, sectionString = "!VX";

  if (!restartFile.is_open()) {
    throw std::invalid_argument("ERROR: Cannot open the file " + fileName +
                                "\nExiting\n");
    exit(0);
  }

  // find numAtoms
  int lineCount = 0;
  while (std::getline(restartFile, line)) {
    lineCount++;
    if (line.find(sectionString) != std::string::npos) {
      break;
    }
  }
  numAtoms = (lineCount - 3);
  restartFile.clear();
  restartFile.seekg(0);

  if (numAtoms == -1) {
    throw std::invalid_argument("ERROR: numAtoms is still -1.\n");
  }

  // First line SHOULD BE a comment saying !XOLD, YOLD, ZOLD
  std::getline(restartFile, line);

  // Extract the positions to a float3 vector, then create a CharmmCrd file
  // from it
  std::vector<float3> inpCrd;
  float x, y, z;
  float3 crd;

  for (int count = 0; count < numAtoms; count++) {
    std::getline(restartFile, line);
    std::stringstream ss(line);
    ss >> x >> y >> z;
    // Save somewhere !
    crd.x = x;
    crd.y = y;
    crd.z = z;
    inpCrd.push_back(crd);
  }
  auto charmmCrd = std::make_shared<CharmmCrd>(inpCrd);
  // use setCoordinates to setup ctx.numAtoms as well as charges (from PSF)
  // and xyzq
  setCoordinates(charmmCrd);

  // Pass the blank line & the comment-title line
  std::getline(restartFile, line);
  std::getline(restartFile, line);
  double vx, vy, vz;
  // Extract the velocities
  for (int count = 0; count < numAtoms; count++) {
    std::getline(restartFile, line);
    std::stringstream ss(line);
    ss >> vx >> vy >> vz;
    // Save somewhere !
    double4 vm = velocityMass[count];
    vm.x = vx;
    vm.y = vy;
    vm.z = vz;
    velocityMass[count] = vm;
  }
  velocityMass.transferToDevice();

  // Find the box dimension line
  while (std::getline(restartFile, line)) {
    if (line.find("!BOXX") != std::string::npos) {
      break;
    }
  }
  if (std::getline(restartFile, line)) {
    std::stringstream ss(line);
    ss >> x >> y >> z;
    setBoxDimensions({x, y, z});
  }
}

int CharmmContext::getNumDegreesOfFreedom() { return numDegreesOfFreedom; }

void CharmmContext::setLogger() {
  logger = std::make_shared<Logger>(shared_from_this());
  hasLogger = true;
}
void CharmmContext::setLogger(std::string logFileName) {
  logger = std::make_shared<Logger>(shared_from_this(), logFileName);
  hasLogger = true;
}

std::shared_ptr<Logger> CharmmContext::getLogger() {
  if (not hasLogger) {
    throw std::invalid_argument("ERROR: Logger not set. \n");
  }
  return logger;
}

bool CharmmContext::hasLoggerSet() const { return hasLogger; }
