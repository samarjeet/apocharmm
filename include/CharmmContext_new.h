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
#include <VolumePiston.h>
#include <random_utils.h>


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
    void setMasses(const char* fileName);

    double calculatePotentialEnergy(bool reset);
    double calculatePotentialEnergyPrint(bool reset);
    void calculateKineticEnergy();
    double* getForces();
    int getNumAtoms() const;
    XYZQ* getXYZQ();

    int* get_loc2glo() const;
    int getForceStride() const;
    double* d_ke;
    float boxx, boxy, boxz;


    // Getters and setters
    void setNumAtoms(const int num);
    void setCoordsCharges(const std::vector<float4>& coordsChargesIn);
    void setCoordsCharges(const std::vector<std::vector<float>>& coordsChargesIn);
    void setCoordinates( CharmmCrd& crd);
    void setMasses(const std::vector<double>& masses);
    void assignVelocitiesAtTemperature(float temp);
    CudaContainer<double4> getVelocityMass();
    void addPSF(const CharmmPSF& psf);
    void readCharmmParameterFile(const std::string fileName);
    void calculateForces();

   private:
    uint64_t seed;
    int numAtoms;

    PBC pbc;
    CudaContainer<float> charges;

    XYZQ xyzq, xyzq_sorted;
    CudaContainer<float4> coordsCharge;

    // TODO : have a separate class/CudaContainer for this
    float4 *h_velMass, *d_velMass;

    CudaContainer<double4> velocityMass;
};
