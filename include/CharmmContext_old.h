// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

/**\file*/

#pragma once
#include <memory>
#include <vector>

#include "CudaContainer.h"
#include "CudaBondedForce.h"
#include "CudaEnergyVirial.h"
#include "CudaPMEDirectForce.h"
#include "CudaTopExcl.h"
#include "CudaNeighborListBuild.h"
#include "CudaNeighborList.h"
#include "cuda_utils.h"
#include "Force.h"
//#include "ForceType.h"
#include "XYZQ.h"
#include "CudaDomdecRecip.h"
#include "CharmmPSF.h"
#include "CharmmCrd.h"
#include "ForceManager.h"
#include <VolumePiston.h>
#include <random_utils.h>

template <typename T>
struct H_DVector {
    T* h = 0;
    T* d = 0;
    // const size_t len;
    size_t len;
    H_DVector() { len = 1000; }
    H_DVector(size_t len) : len(len) {
        h = (T*)malloc(len * sizeof(T));
        cudaCheck(cudaMalloc((void**)&d, len * sizeof(T)));
    }
    void c2d() { cudaCheck(cudaMemcpy(d, h, len * sizeof(T), cudaMemcpyHostToDevice)); }
    void c2h() { cudaCheck(cudaMemcpy(h, d, len * sizeof(T), cudaMemcpyDeviceToHost)); }
    ~H_DVector() {
        free(h);
        cudaFree(d);
    }
};

typedef float CT;
typedef float4 CT4;

enum class PBC { P1, P21 };

class CharmmContext {
   public:
    CharmmContext();
    CharmmContext(const int numAtoms);
    ~CharmmContext();
    CharmmContext(int numAtoms, double boxx_in, double boxy_in, double boxz_in, uint64_t seed);
    void setTemperature(const float temp);
    float getTemperature() const;

    void setPeriodicBoundaryCondition(const PBC p);
    PBC getPeriodicBoundaryCondition() const;
    void setCoords(std::vector<float>& coords);
    void setCharges(std::vector<float>& charges);
    void setPeriodicBoxSizes(double x, double y, double z);
    void setPiston(double pressure, double pmass);
    void setMasses(const char* fileName);
    void assignVelocities();

    double calculatePotentialEnergy(bool reset);
    double calculatePotentialEnergyPrint(bool reset);
    void calculateKineticEnergy();
    double* getForces();
    /** moves the calculated forces into force_invmass array.*/
    void moveForcesToDynamics();
    /** moves xyz into xyzq array.*/
    void movePositionsToEnergy();
    int getNumAtoms() const;
    XYZQ* getXYZQ();

    int* get_loc2glo() const;
    int getForceStride() const;
    double* d_ke;
    float boxx, boxy, boxz;

    /**Seed for random number generator.
     * This is used as the key for philox.*/
    uint64_t getSeed() const { return seed; }
    /**random number step.
     * Increment after a random kernel.
     * Used for 64 msbs of the counter for philox.*/
    H_DVector<uint64_t> randstep;
    /**Volume piston used fro integrating the dynamcis of the box size.*/
    VolumePiston piston;
    /**Force and inverse of atom masses, used in dynamics to update velocities.*/
    H_DVector<double4> force_invmass;
    /**Positions of atoms, used in dynamics.*/
    H_DVector<double4> xyz;
    // CudaContainer<float3> xyz;
    /**Velocities and masses of atoms, used in dynamics.*/
    H_DVector<double4> vel_mass;
    /**Box dimentions, used in box piston dynamics.*/
    H_DVector<double3> box;
    /**Box dimensions time derivative, used in box piston dynamics.*/
    H_DVector<double3> box_dot;
    /**Potential energy.*/
    H_DVector<double> potential_energy;
    /**Pressure group corrected virial, related to the change in energy given a small change in box size.*/
    H_DVector<double3> virial;

    // Getters and setters
    void setNumAtoms(const int num);
    void setTopologicalExclusions(const std::vector<int>& iblo14, const std::vector<int>& inb14);
    void setReciprocalSpaceParameters(int nfft1, int nfft2, int nfft3, int pmeSplineOrderIn, float kappaIn);
    void setDirectSpaceParameters(float cutoffIn);
    void set14InclusionExclusion(const std::vector<int>& inExSize, const std::vector<int>& inEx);
    void setVdwType(const std::vector<int>& vdwType);
    void setVdwParam(const std::vector<float>& vdwParam);
    void setBondedParams(const std::vector<int>& bondedParamsSize,
                         const std::vector<std::vector<float>>& bondedParamsVal);
    void setBondedLists(const std::vector<int>& bondedListSize, const std::vector<std::vector<int>>& bondedListVal);
    void setCoordsCharges(const std::vector<float4>& coordsChargesIn);
    void setCoordsCharges(const std::vector<std::vector<float>>& coordsChargesIn);
    void setCoordsCharges(const float4* coordsChargesIn);
    void setCoordinates( CharmmCrd& crd);
    void setMasses(const std::vector<double>& masses);
    void assignVelocitiesAtTemperature(float temp);
    CudaContainer<double4> getVelocityMass();
    void addPSF(const CharmmPSF& psf);
    void addCrd(const CharmmCrd& crd);
    void readCharmmParameterFile(const std::string fileName);
    //void addForce(std::shared_ptr<ForceType> force);
    // TODO : remove this
    void calculateReciprocalSpace();
    void calculateForces();

   private:
    uint64_t seed;
    int numAtoms;
    float temperature;

    int dummy;
    int zone_patom[9];
    float kappa;
    float cutoff;
    int pmeSplineOrder;

    int nfftx, nffty, nfftz;
    PBC pbc;
    CudaContainer<float> charges;

    CudaEnergyVirial energyVirial;
    CudaBondedForce<long long int, float> bonded;
    Force<long long int> bondedForce;
    cudaStream_t bondedStream;

    CudaEnergyVirial directEnergyVirial;
    // CudaPMEDirectForceBase<long long int, float> *direct;
    CudaPMEDirectForce<long long int, float>* direct = 0;
    cudaStream_t directStream;

    CudaTopExcl topExcl;
    CudaNeighborList<32> neighborList;
    CudaNeighborListBuild<32>* nlist[2];
    int* loc2glo = 0;
    void set_glo_vdwtype(std::string fileName);
    void set_glo_vdwtype(const std::vector<int>& vdwType);
    int* glo_vdwtype = 0;
    Force<long long int> directForce;
    Force<long long int> directForceSorted;

    CudaEnergyVirial recipEnergyVirial;
    CudaDomdecRecip recip;
    Force<long long int> recipForce;
    cudaStream_t recipStream;
    cudaEvent_t recipForce_done_event;

    XYZQ xyzq, xyzq_sorted;
    CudaContainer<float4> coordsCharge;

    //std::vector<std::shared_ptr<ForceType>> forces;

    // TODO : have a separate class/CudaContainer for this
    float4 *h_velMass, *d_velMass;

    CudaContainer<double4> velocityMass;

    ForceManager forceManager;
};
