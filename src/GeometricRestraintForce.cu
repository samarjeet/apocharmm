// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "GeometricRestraintForce.h"
#include "gpu_utils.h"

__device__ float3 operator+(const float3 &a, const float3 &b) {

  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 operator-(const float3 &a, const float3 &b) {

  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 operator*(const float3 &a, const float3 &b) {

  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float dot(const float3 &a, const float3 &b) {

  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 operator*(const float &a, const float3 &b) {

  return make_float3(a * b.x, a * b.y, a * b.z);
}

template <typename AT, typename CT>
GeometricRestraintForce<AT, CT>::GeometricRestraintForce(
    CudaEnergyVirial &_energyVirial)
    : energyVirial(_energyVirial) {

  energyVirial.insert("geo");
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::setForce(
    std::shared_ptr<Force<long long int>> &forceValIn) {
  forceVal = forceValIn;
}

template <typename AT, typename CT>
__forceinline__ __device__ float
harmonicRestraintForce(const float4 *__restrict__ xyzqi, AT *__restrict__ force,
                       float3 origin, float3 orientation, AT *energy) {

  // distance from origin
  // float3 r = make_float3(xyzqi->x, xyzqi->y, xyzqi->z) - origin;
}

// Should we use a single kernel for all the restraints or a kernel for each
// restraint type or over all atoms?
template <typename AT, typename CT>
__global__ void restraintKernel(size_t numRestraints, Restraint *restraints,
                                // bool calcEnergy, bool calcVirial,
                                const float4 *__restrict__ xyzq,
                                const int stride, const CT boxx, const CT boxy,
                                const CT boxz, AT *__restrict__ force,
                                double *__restrict__ energy_geo) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  float magnitude = 0.0;
  int restraintIndex = index / 32;

  if (restraintIndex < numRestraints) {
    auto r = restraints[restraintIndex];
    float3 origin = r.origin;
    if (r.relativeToBox) {
      origin.x *= boxx;
      origin.y *= boxy;
      origin.z *= boxz;
    }
    if (r.potential == PotentialFunction::HARMONIC) {
      if (r.shape == RestraintShape::PLANE) {

        float dotProduct = 0.0;
        // float val = 0.0;
        //  scratch[threadIdx.x % warpSize] = 0.0;
        //  One warp per restraint
        for (int i = index % 32; i < r.size; i += warpSize) {
          if (i < r.size) {
            auto xyzqi = xyzq[r.atoms[i]];
            float3 vec = make_float3(xyzqi.x, xyzqi.y, xyzqi.z) - r.origin;
            dotProduct += vec.x * r.orientation.x + vec.y * r.orientation.y +
                          vec.z * r.orientation.z;
            if (dotProduct > 0) {
              magnitude = dotProduct - r.offsetDistance;
            } else {
              magnitude = dotProduct + r.offsetDistance;
            }
            if (r.insideOnly && magnitude < 0) {
              magnitude = 0;
            } else if (not r.insideOnly && magnitude > 0) {
              magnitude = 0;
            }
            magnitude *= r.forceConstant;

            AT fx, fy, fz;
            // calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);
            calc_component_force<AT, CT>(magnitude, r.orientation.x,
                                         r.orientation.y, r.orientation.z, fx,
                                         fy, fz);

            write_force(fx, fy, fz, r.atoms[i], stride, force);
          }
        }
      }
    }
  }
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::addRestraint(
    RestraintShape shape, PotentialFunction potential, bool isCenterOfMass,
    float3 origin, bool relativeToBox, float3 orientation, bool insideOnly,
    float forceConstant, float offsetDistance, std::vector<int> atoms) {

  Restraint r;
  r.shape = shape;
  r.potential = potential;
  r.isCenterOfMass = isCenterOfMass;
  r.origin = origin;
  r.relativeToBox = relativeToBox;
  auto norm =
      sqrt(orientation.x * orientation.x + orientation.y * orientation.y +
           orientation.z * orientation.z);
  orientation.x /= norm;
  orientation.y /= norm;
  orientation.z /= norm;
  r.orientation = orientation;
  r.insideOnly = insideOnly;
  r.forceConstant = forceConstant;
  r.offsetDistance = offsetDistance;
  int pos = 0;
  for (auto a : atoms) {
    r.atoms[pos++] = a;
  }
  r.size = atoms.size();
  // r.atoms = atoms;

  // int idx = restraints.size();
  // restraints.resize(restraints.size() + 1);
  // restraints[idx] = r;

  restraints.push_back(r);
  thrust::device_vector<int> atoms_d(atoms);
  // // thrust::device_vector<thrust::device_vector<int>> restraintAtoms;
  // // restraintAtoms.push_back(atoms_d);
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::initialize() {}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::calc_force(const float4 *xyzq,
                                                 bool calcEnergy,
                                                 bool calcVirial) {
  std::cout << "Calculating geometric restraint force\n";
  std::cout << "There are " << restraints.size() << " restraints\n";

  // Each warp will calculate the force for one restraint
  // TODO : clean up the number of threads and blocks
  int numThreads = 128;
  int numBlocks = (restraints.size() - 1) / numThreads + 1;

  float3 box = {50.0, 50.0, 50.0};

  // restraintKernel<<<numBlocks, numThreads>>>(
  //   restraints.size(), restraints.getDeviceData(),
  //   // calcEnergy, calcVirial,
  //   xyzq, forceVal->stride(), box.x, box.y, box.z, forceVal->xyz(),
  //   energyVirial.getEnergyPointer("geo"));
  restraintKernel<<<numBlocks, numThreads>>>(
      restraints.size(), thrust::raw_pointer_cast(restraints.data()),
      // calcEnergy, calcVirial,
      xyzq, forceVal->stride(), box.x, box.y, box.z, forceVal->xyz(),
      energyVirial.getEnergyPointer("geo"));
}

template class GeometricRestraintForce<long long int, float>;
