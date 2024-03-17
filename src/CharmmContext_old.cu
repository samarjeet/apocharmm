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
#include "gpu_utils.h"
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
//#include "xyzq_sorted.h"

void setBondedCoeff(CudaBondedForce<long long int, float> *bonded,
                    const char *sizeFile, const char *valFile) {
    int nbondcoef, nureybcoef, nanglecoef, ndihecoef, nimdihecoef, ncmapcoef;
    float2 *bondcoef, *ureybcoef, *anglecoef, *cmapcoef;
    float4 *dihecoef, *imdihecoef;

    std::ifstream inSizeFile(sizeFile);
    std::ifstream inValFile(valFile);
    inSizeFile >> nbondcoef >> nureybcoef >> nanglecoef >> ndihecoef >>
        nimdihecoef >> ncmapcoef;

    bondcoef = (float2 *)malloc(nbondcoef * sizeof(float2));
    ureybcoef = (float2 *)malloc(nureybcoef * sizeof(float2));
    anglecoef = (float2 *)malloc(nanglecoef * sizeof(float2));
    dihecoef = (float4 *)malloc(ndihecoef * sizeof(float4));
    imdihecoef = (float4 *)malloc(nimdihecoef * sizeof(float4));
    cmapcoef = (float2 *)malloc(ncmapcoef * sizeof(float2));

    for (int i = 0; i < nbondcoef; ++i) {
        inValFile >> bondcoef[i].x >> bondcoef[i].y;
    }

    for (int i = 0; i < nureybcoef; ++i) {
        inValFile >> ureybcoef[i].x >> ureybcoef[i].y;
    }

    for (int i = 0; i < nanglecoef; ++i) {
        inValFile >> anglecoef[i].x >> anglecoef[i].y;
    }

    for (int i = 0; i < ndihecoef; ++i) {
        inValFile >> dihecoef[i].x >> dihecoef[i].y >> dihecoef[i].z >>
            dihecoef[i].w;
    }

    for (int i = 0; i < nimdihecoef; ++i) {
        inValFile >> imdihecoef[i].x >> imdihecoef[i].y >> imdihecoef[i].z >>
            imdihecoef[i].w;
    }

    for (int i = 0; i < ncmapcoef; ++i) {
        inValFile >> cmapcoef[i].x >> cmapcoef[i].y;
    }

    inSizeFile.close();
    inValFile.close();

    bonded->setup_coef(nbondcoef, bondcoef, nureybcoef, ureybcoef, nanglecoef,
                       anglecoef, ndihecoef, dihecoef, nimdihecoef, imdihecoef,
                       nureybcoef, ureybcoef);

    free(bondcoef);
    free(ureybcoef);
    free(anglecoef);
    free(dihecoef);
    free(imdihecoef);
    free(cmapcoef);
}

void setBondedList(CudaBondedForce<long long int, float> *bonded,
                   const char *sizeFile, const char *valFile) {
    int nbondlist, nureyblist, nanglelist, ndihelist, nimdihelist, ncmaplist;
    bondlist_t *h_bondlist;
    bondlist_t *h_ureyblist;
    anglelist_t *h_anglelist;
    dihelist_t *h_dihelist;
    dihelist_t *h_imdihelist;
    cmaplist_t *h_cmaplist;

    std::ifstream inSizeFile(sizeFile);
    std::ifstream inValFile(valFile);
    inSizeFile >> nbondlist >> nureyblist >> nanglelist >> ndihelist >>
        nimdihelist >> ncmaplist;

    h_bondlist = (bondlist_t *)malloc(sizeof(bondlist_t) * nbondlist);
    h_ureyblist = (bondlist_t *)malloc(sizeof(bondlist_t) * nbondlist);
    h_anglelist = (anglelist_t *)malloc(sizeof(anglelist_t) * nanglelist);
    h_dihelist = (dihelist_t *)malloc(sizeof(dihelist_t) * ndihelist);
    h_imdihelist = (dihelist_t *)malloc(sizeof(dihelist_t) * ndihelist);
    h_cmaplist = (cmaplist_t *)malloc(sizeof(cmaplist_t) * ncmaplist);

    for (int i = 0; i < nbondlist; ++i) {
        inValFile >> h_bondlist[i].i >> h_bondlist[i].j >>
            h_bondlist[i].itype >> h_bondlist[i].ishift;
    }

    for (int i = 0; i < nureyblist; ++i) {
        inValFile >> h_ureyblist[i].i >> h_ureyblist[i].j >>
            h_ureyblist[i].itype >> h_ureyblist[i].ishift;
    }

    for (int i = 0; i < nanglelist; ++i) {
        inValFile >> h_anglelist[i].i >> h_anglelist[i].j >> h_anglelist[i].k >>
            h_anglelist[i].itype >> h_anglelist[i].ishift1 >>
            h_anglelist[i].ishift2;
    }

    for (int i = 0; i < ndihelist; ++i) {
        inValFile >> h_dihelist[i].i >> h_dihelist[i].j >> h_dihelist[i].k >>
            h_dihelist[i].l >> h_dihelist[i].itype >> h_dihelist[i].ishift1 >>
            h_dihelist[i].ishift2 >> h_dihelist[i].ishift3;
    }

    for (int i = 0; i < nimdihelist; ++i) {
        inValFile >> h_imdihelist[i].i >> h_imdihelist[i].j >>
            h_imdihelist[i].k >> h_imdihelist[i].l >> h_imdihelist[i].itype >>
            h_imdihelist[i].ishift1 >> h_imdihelist[i].ishift2 >>
            h_imdihelist[i].ishift3;
    }
    bonded->setup_list(nbondlist, h_bondlist, nureyblist, h_ureyblist,
                       nanglelist, h_anglelist, ndihelist, h_dihelist,
                       nimdihelist, h_imdihelist, ncmaplist, h_cmaplist);

    inSizeFile.close();
    inValFile.close();
}

void CharmmContext::set_glo_vdwtype(std::string fileName) {
    int *h_vdwtype;
    int ncoord = numAtoms;
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        // Open file
        file.open(fileName);

        h_vdwtype = new int[ncoord];

        for (int i = 0; i < ncoord; i++) {
            file >> h_vdwtype[i];
        }

        file.close();
    } catch (std::ifstream::failure e) {
        std::cerr << "Error opening/reading/closing file " << fileName
                  << std::endl;
        exit(1);
    }

    // Align ncoord to warpsize
    int ncoord_aligned = ((ncoord - 1) / warpsize + 1) * warpsize;
    int vdwtype_len = ncoord;
    reallocate<int>(&glo_vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);
    copy_HtoD_sync<int>(h_vdwtype, glo_vdwtype, ncoord);
    delete[] h_vdwtype;
}

void CharmmContext::set_glo_vdwtype(const std::vector<int> &vdwType) {
    // Align ncoord to warpsize
    int ncoord_aligned = ((numAtoms - 1) / warpsize + 1) * warpsize;
    int vdwtype_len = numAtoms;
    reallocate<int>(&glo_vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);
    copy_HtoD_sync<int>(vdwType.data(), glo_vdwtype, numAtoms);
}

CharmmContext::CharmmContext()
    : recip(recipEnergyVirial),
      bonded(energyVirial, "bond", "ureyb", "angle", "dihe", "imdihe", "cmap"),
      topExcl(),
      neighborList(topExcl, 1, 1, 1) {
    numAtoms = -1;
    boxx = 0;
    boxy = 0;
    boxz = 0;
    std::vector<int> devices = {0, 1, 2, 3};
    start_gpu(1, 1, 0, devices);
    cudaCheck(cudaStreamCreate(&directStream));
    cudaCheck(cudaStreamCreate(&bondedStream));
    cudaCheck(cudaStreamCreate(&recipStream));
    cudaCheck(cudaEventCreate(&recipForce_done_event));
    nlist[0] = nullptr;
    nlist[1] = nullptr;
    // box = H_DVector<double3>(1);
    cudaMalloc(&d_ke, 1 * sizeof(double));
}

CharmmContext::CharmmContext(int numAtoms)
    : numAtoms(numAtoms),
      recip(recipEnergyVirial),
      bonded(energyVirial, "bond", "ureyb", "angle", "dihe", "imdihe", "cmap"),
      topExcl(),
      neighborList(topExcl, 1, 1, 1) {
    boxx = 0;
    boxy = 0;
    boxz = 0;
    std::vector<int> devices = {0, 1, 2, 3};
    start_gpu(1, 1, 0, devices);
    cudaCheck(cudaStreamCreate(&directStream));
    cudaCheck(cudaStreamCreate(&bondedStream));
    cudaCheck(cudaStreamCreate(&recipStream));
    cudaCheck(cudaEventCreate(&recipForce_done_event));
    nlist[0] = nullptr;
    nlist[1] = nullptr;
    // box = H_DVector<double3>(1);
    cudaMalloc(&d_ke, 1 * sizeof(double));

    setNumAtoms(numAtoms);

    // By default, it adds Bonded forces, Direct space and recip forces
    // Other ForceType can be added
    // Any element can be removed as well
    // forces.push_back(std::make_unique<CudaBondedForce<long long int, float>>(
    //        energyVirial, "bond", "ureyb", "angle", "dihe", "imdihe",
    //        "cmap"));
    //addForce(std::make_shared<CudaBondedForce<long long int, float>>(
    //    energyVirial, "bond", "ureyb", "angle", "dihe", "imdihe", "cmap"));
}

// Destructor

CharmmContext::~CharmmContext() {
    // cudaDeviceSynchronize();
    try {
        if (loc2glo) {
            cudaCheck(cudaFree(loc2glo));
        }
        if (glo_vdwtype) {
            cudaCheck(cudaFree(glo_vdwtype));
        }
        if (nlist[0]) {
            free(nlist[0]);
            free(nlist[1]);
        }
        // free(direct);
    } catch (const char *msg) {
        std::cerr << msg << std::endl;
    }
}

void CharmmContext::setNumAtoms(const int num) {
    numAtoms = num;
    zone_patom[0] = 0;
    for (int i = 1; i < 9; i++) zone_patom[i] = numAtoms;

    cudaCheck(cudaMalloc(&loc2glo, numAtoms * sizeof(int)));

    recipForce.realloc(numAtoms, 1.5f);
    directForce.realloc(numAtoms, 1.5f);
    directForceSorted.realloc(numAtoms, 1.5f);
    bondedForce.realloc(numAtoms, 1.5f);

    charges.allocate(numAtoms);
    velocityMass.allocate(numAtoms);
}

void CharmmContext::setTopologicalExclusions(const std::vector<int> &iblo14,
                                             const std::vector<int> &inb14) {
    assert(numAtoms == iblo14.size());
    topExcl.setFromVector(numAtoms, iblo14, inb14);
}

void CharmmContext::setReciprocalSpaceParameters(int nfft1, int nfft2,
                                                 int nfft3,
                                                 int pmeSplineOrderIn,
                                                 float kappaIn) {
    nfftx = nfft1;
    nffty = nfft2;
    nfftz = nfft3;
    pmeSplineOrder = pmeSplineOrderIn;
    kappa = kappaIn;
    recip.setParameters(nfftx, nffty, nfftz, pmeSplineOrder, kappa,
                        recipStream);
}

void CharmmContext::setDirectSpaceParameters(float cutoffIn) {
    cutoff = cutoffIn;
    assert(cutoff <= boxx / 2.0 and cutoff <= boxy / 2.0 and
           cutoff <= boxz / 2.0);
    std::vector<int> numIntZones(8, 0);
    std::vector<std::vector<int>> intZones(8, std::vector<int>());
    numIntZones.at(0) = 1;
    intZones.at(0).push_back(0);
    neighborList.registerList(numIntZones, intZones);

    nlist[0] = new CudaNeighborListBuild<32>(0, 0, 0);
    nlist[1] = new CudaNeighborListBuild<32>(0, 0, 0);

    direct = new CudaPMEDirectForce<long long int, float>(
        directEnergyVirial, "vdw", "elec", "ewex");
    direct->setup(boxx, boxy, boxz, kappa, cutoff, cutoff - 2.0, 1.0, 3, 101);
}

void CharmmContext::set14InclusionExclusion(const std::vector<int> &inExSize,
                                            const std::vector<int> &inEx) {
    direct->set_14_list(inExSize, inEx);
}

void CharmmContext::setVdwType(const std::vector<int> &vdwType) {
    assert(numAtoms == vdwType.size());
    direct->set_vdwtype(vdwType);
    direct->set_vdwtype14(vdwType);
    set_glo_vdwtype(vdwType);
}

void CharmmContext::setVdwParam(const std::vector<float> &vdwParam) {
    direct->set_vdwparam(vdwParam);
    direct->set_vdwparam14(vdwParam);
}

void CharmmContext::setCoordsCharges(
    const std::vector<float4> &coordsChargesIn) {
    assert(coordsChargesIn.size() == numAtoms);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    xyzq.set_ncoord(numAtoms);
    xyzq_sorted.set_ncoord(numAtoms);
    xyzq.set_xyzq(numAtoms, coordsChargesIn.data(), 0, stream);
}

void CharmmContext::setCoordsCharges(
    const std::vector<std::vector<float>> &coordsChargesIn) {
    std::vector<float4> coordsCharges;
    for (int i = 0; i < numAtoms; ++i) {
        float4 elem;
        elem.x = coordsChargesIn[i][0];
        elem.y = coordsChargesIn[i][1];
        elem.z = coordsChargesIn[i][2];
        elem.w = coordsChargesIn[i][3];
        coordsCharges.push_back(elem);
    }
    setCoordsCharges(coordsCharges);
}

void CharmmContext::setCoordsCharges(
    const float4* coordsChargesIn) {
    std::vector<float4> coordsCharges;
    for (int i = 0; i < numAtoms; ++i) {
        float4 elem;
        elem.x = coordsChargesIn[i].x;
        elem.y = coordsChargesIn[i].y;
        elem.z = coordsChargesIn[i].z;
        elem.w = coordsChargesIn[i].w;
        coordsCharges.push_back(elem);
    }
    setCoordsCharges(coordsCharges);
}

void CharmmContext::setCoordinates( CharmmCrd& crd){
    // use the coordinates from the crd 
    auto coords = crd.getCoordinates();
}

void CharmmContext::setBondedParams(
    const std::vector<int> &bondedParamsSize,
    const std::vector<std::vector<float>> &bondedParamsVal) {
    bonded.setup_coef(bondedParamsSize, bondedParamsVal);
}

void CharmmContext::setBondedLists(
    const std::vector<int> &bondedListSize,
    const std::vector<std::vector<int>> &bondedListVal) {
    bonded.setup_list(bondedListSize, bondedListVal, bondedStream);
}

void CharmmContext::setMasses(const std::vector<double> &masses) {
    assert(masses.size() == numAtoms);
    assert(velocityMass.size() == numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        velocityMass[i].w = 1.0 / masses[i];
    }
    velocityMass.transferToDevice();
}

void CharmmContext::assignVelocitiesAtTemperature(float temp) {
    assert(numAtoms != -1);
    setTemperature(temp);
    double kboltz = 1.987191E-03;
    double boltz = kboltz * temperature;

    // TODO : make it random
    // randstep = H_DVector<uint64_t>(1);
    for (int i = 0; i < numAtoms; i++) {
        double sd = boltz * velocityMass[i].w;
        sd = sqrt(sd);
        double2 r;

        // r = randnormal(seed, *randstep.h, i * 2);
        // float2 r;
        r.x = 1.1;
        r.y = 1.2;
        velocityMass[i].x = r.x * sd;
        velocityMass[i].y = r.y * sd;
        // r = randnormal(seed, *randstep.h, i * 2 + 1);
        velocityMass[i].z = r.x * sd;
    }
    velocityMass.transferToDevice();
}

void CharmmContext::addPSF(const CharmmPSF& psf) {
    /*
    add the following :
    atom types
    charges of the atoms
    bonds
    angles
    dihedrals
    impropers
    */

    // psf

}

void CharmmContext::readCharmmParameterFile(const std::string fileName) {}

//void CharmmContext::addForce(std::shared_ptr<ForceType> force) {
//    forces.push_back(std::move(force));
//}

void CharmmContext::calculateForces() {
    // ----------------------Bonded-------------
    bondedForce.clear(bondedStream);
    //std::static_pointer_cast<CudaBondedForce<long long int, float>>(forces[0])
    //    ->calc_force(xyzq.xyzq, boxx, boxy, boxz, false, false,
    //                 bondedForce.stride(), bondedForce.xyz(), true, true, true,
    //                 true, true, false, bondedStream);
    cudaStreamSynchronize(bondedStream);
    cudaDeviceSynchronize();
    energyVirial.copyToHost();
    cudaDeviceSynchronize();

    std::cout << "Bond energy : " << energyVirial.getEnergy("bond") << "\n";
    std::cout << "Urey  energy : " << energyVirial.getEnergy("ureyb") << "\n";
    std::cout << "Angle energy : " << energyVirial.getEnergy("angle") << "\n";
    std::cout << "Dihedral energy : " << energyVirial.getEnergy("dihe") << "\n";
    std::cout << "Imdihedral energy : " << energyVirial.getEnergy("imdihe")
              << "\n";
}

CharmmContext::CharmmContext(int nAtoms, double boxx_in, double boxy_in,
                             double boxz_in, uint64_t seed)
    : numAtoms(nAtoms),
      bonded(energyVirial, "bond", "ureyb", "angle", "dihe", "imdihe", "cmap"),
      topExcl(numAtoms, "../test/iblo14.txt", "../test/inb14.txt"),
      kappa(0.34),
      cutoff(8),
      pmeSplineOrder(4),
      neighborList(topExcl, 1, 1, 1),
      nfftx(48),
      nffty(48),
      nfftz(48),
      recip(nfftx, nffty, nfftz, pmeSplineOrder, kappa, recipEnergyVirial,
            "ewks", "ewse"),
      randstep(1),
      seed(seed),
      force_invmass(nAtoms),
      xyz(nAtoms),
      vel_mass(nAtoms),
      box(1),
      box_dot(1),
      potential_energy(1),
      virial(1),
      xyzq(nAtoms),
      xyzq_sorted(nAtoms) {
    std::vector<int> devices = {0, 1, 2, 3};
    start_gpu(1, 1, 0, devices);
    pbc = PBC::P1;
    // Bonded force stuff
    setBondedCoeff(&bonded, "../test/bonded_size_coef.txt",
                   "../test/bonded_val_coef.txt");
    cudaDeviceSynchronize();
    setBondedList(&bonded, "../test/bonded_size_list.txt",
                  "../test/bonded_val_list.txt");
    // setBondedList(&bonded, "../test/bonded_size_list.txt",
    // "../test/bond_orig_val_list.txt");
    cudaDeviceSynchronize();
    *randstep.h = 0;  // set rand step to zero
    randstep.c2d();   // copy to device

    // bonded.print();

    // Non-bonded stuff
    cudaCheck(cudaMalloc(&loc2glo, numAtoms * sizeof(int)));

    std::vector<int> numIntZones(8, 0);
    std::vector<std::vector<int>> intZones(8, std::vector<int>());
    numIntZones.at(0) = 1;
    intZones.at(0).push_back(0);
    neighborList.registerList(numIntZones, intZones);

    nlist[0] = new CudaNeighborListBuild<32>(0, 0, 0);
    nlist[1] = new CudaNeighborListBuild<32>(0, 0, 0);

    direct = new CudaPMEDirectForce<long long int, float>(
        directEnergyVirial, "vdw", "elec", "ewex");
    setPeriodicBoxSizes(boxx_in, boxy_in,
                        boxz_in);  // default placeholder for now
    cudaCheck(cudaStreamCreate(&directStream));

    // Direct space : 14 inclusion and exclusion
    direct->set_14_list("../test/in14_ex14_size.txt",
                        "../test/in14_ex14_val.txt", directStream);

    // set the vdw params
    int h_nvdwparam = 0;
    std::ifstream infile("../test/vdwparams.txt");
    float val;
    while (infile >> val) {
        h_nvdwparam++;
    }
    std::cout << "h_nvdwparam : " << h_nvdwparam << "\n";
    direct->set_vdwtype(numAtoms, "../test/global_vdwtype.txt");
    direct->set_vdwtype14(numAtoms, "../test/global_vdwtype.txt");
    glo_vdwtype = NULL;
    set_glo_vdwtype("../test/global_vdwtype.txt");
    cudaDeviceSynchronize();
    // direct->set_vdwparam(h_nvdwparam, h_vdwparam);
    direct->set_vdwparam(h_nvdwparam, "../test/vdwparams.txt");
    direct->set_vdwparam14(h_nvdwparam, "../test/vdwparams14.txt");

    dummy = 0;
    cudaMalloc(&d_ke, 1 * sizeof(double));
    zone_patom[0] = 0;
    for (int i = 1; i < 9; i++) zone_patom[i] = numAtoms;
    // cudaStreamCreate(&directStream);
    cudaCheck(cudaStreamCreate(&bondedStream));
    cudaCheck(cudaStreamCreate(&recipStream));
    cudaCheck(cudaEventCreate(&recipForce_done_event));
    recip.set_stream(recipStream);
    recipForce.realloc(numAtoms, 1.5f);

    directForce.realloc(numAtoms, 1.5f);
    directForceSorted.realloc(numAtoms, 1.5f);
    bondedForce.realloc(numAtoms, 1.5f);
    charges.allocate(numAtoms);
}

void CharmmContext::setTemperature(const float temp) { temperature = temp; }

float CharmmContext::getTemperature() const { return temperature; }

void CharmmContext::setPeriodicBoundaryCondition(const PBC p) { pbc = p; }

PBC CharmmContext::getPeriodicBoundaryCondition() const { return pbc; }

void CharmmContext::setCoords(std::vector<float> &coordsVec) {
    // std::cout << "Size of coodsVec " << coordsVec.size() << "\n";
    /*assert(coordsVec.size() == numAtoms * 3);
    std::vector<float4> coordsCharge;
    for ( const auto& c: coordsVec){
      float4 elem ;
      elem.x = c.x;
      elem.y = c.y;
      elem.z = c.z;
      elem.w = 0.0;
      coordsCharge.push_back(elem);
    }
    //float4 *h_xyzq_sorted = (float4 *)malloc(numAtoms * sizeof(float4));
    //float *coords = coordsVec.data();
    //const float *chargeContainer = charges.getHostArray().data();
    //for (int i = 0; i < numAtoms; ++i) {
    //    h_xyzq_sorted[i].x = coords[3 * i];
    //    h_xyzq_sorted[i].y = coords[3 * i + 1];
    //    h_xyzq_sorted[i].z = coords[3 * i + 2];
    //    h_xyzq_sorted[i].w = chargeContainer[i];
    //    xyz.h[i].x = coords[3 * i];
    //    xyz.h[i].y = coords[3 * i + 1];
    //    xyz.h[i].z = coords[3 * i + 2];
    //}
    //xyz.c2d();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    xyzq.set_xyzq(numAtoms, coordsCharge.data(), 0, stream);
    */
}

void CharmmContext::setCharges(std::vector<float> &chargesVec) {
    // std::cout << "Size of chargesVec : " << chargesVec.size() << "\n";
    /*charges.set(chargesVec);
    assert(chargesVec.size() == numAtoms);
    // std::cout << "Size of coodsVec " << coordsVec.size() << "\n";
    float4 *h_xyzq_sorted = (float4 *)malloc(numAtoms * sizeof(float4));
    float *chargeContainer = chargesVec.data();
    for (int i = 0; i < numAtoms; ++i) {
        h_xyzq_sorted[i].x = (CT)xyz.h[i].x;
        h_xyzq_sorted[i].y = (CT)xyz.h[i].y;
        h_xyzq_sorted[i].z = (CT)xyz.h[i].z;
        h_xyzq_sorted[i].w = chargeContainer[i];
        xyz.h[i].w = chargeContainer[i];
    }
    xyz.c2d();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    xyzq.set_xyzq(numAtoms, h_xyzq_sorted, 0, stream);
    // set_xyzq()
    free(h_xyzq_sorted);
    */
}

void CharmmContext::setPeriodicBoxSizes(double x, double y, double z) {
    boxx = (CT)x;
    boxy = (CT)y;
    boxz = (CT)z;
    // direct->setup(boxx, boxy, boxz, kappa, cutoff, cutoff - 2.0, 1.0, 3,
    // 101);  // DHFR box.h->x = x; box.h->y = y; box.h->z = z; box.c2d();
}

void CharmmContext::setPiston(double pressure, double pmass) {
    piston.pressure = pressure;
    piston.piston_mass = pmass;
}

__global__ void set_loc2glo(int *array, int size) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) array[id] = id;
}

__global__ void printForcesKernel(int numAtoms, long long int *force) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numAtoms) {
        printf("tid: %d force: %.6f\n", tid, force[tid]);
    }
}
void printForces(Force<long long int> force) {
    int numAtoms = force.size() / 3;
    int nthreads = 128;
    int nblocks = (numAtoms - 1) / nthreads + 1;
    printForcesKernel<<<nthreads, nblocks>>>(numAtoms, force.xyz());
    cudaDeviceSynchronize();
}

__global__ void mapBackdirectsKernel(long long int *unsorted,
                                     long long int *sorted, int stride,
                                     int numAtoms, int *loc2glo) {
    // Add to bondedForce after transformation
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numAtoms) {
        int ind = loc2glo[tid];
        unsorted[ind] += sorted[tid];
        unsorted[ind + stride] += sorted[tid + stride];
        unsorted[ind + 2 * stride] += sorted[tid + 2 * stride];
    }
}

double CharmmContext::calculatePotentialEnergyPrint(bool reset) {
    // double energy = 0.0;

    // ----------------------Direct-------------

    if (reset) {
        int numThreads = 512;
        int numBlocks = (numAtoms - 1) / numThreads + 1;

        // int loc2glo_len = numAtoms;
        set_loc2glo<<<numBlocks, numThreads>>>(loc2glo, numAtoms);
        cudaDeviceSynchronize();
        // sort coord
        neighborList.sort(0, zone_patom, xyzq.xyzq, xyzq_sorted.xyzq, loc2glo,
                          directStream);
        cudaStreamSynchronize(directStream);

        direct->set_vdwtype(numAtoms, glo_vdwtype,
                            neighborList.get_ind_sorted(), directStream);
        // build neighborList
        neighborList.build(0, zone_patom, boxx, boxy, boxz, cutoff + 1.0,
                           xyzq_sorted.xyzq, loc2glo, directStream);
        cudaStreamSynchronize(directStream);
    }

    directForce.clear();
    directForceSorted.clear();
    directEnergyVirial.clear(0);
    direct->calc_force(0, xyzq_sorted.xyzq, (&neighborList)->getBuilder(0),
                       true, true, directForceSorted.stride(),
                       directForceSorted.xyz(), directStream);
    cudaStreamSynchronize(directStream);

    direct->calc_14_force(xyzq.xyzq, true, true, directForce.stride(),
                          directForce.xyz(), directStream);
    cudaStreamSynchronize(directStream);
    cudaDeviceSynchronize();
    directEnergyVirial.copyToHost(0);
    cudaDeviceSynchronize();

    std::cout << "elec energy : " << std::setprecision(10)
              << directEnergyVirial.getEnergy("elec") << "\n";
    std::cout << "ewex energy : " << directEnergyVirial.getEnergy("ewex")
              << "\n";
    std::cout << "vdw  energy : " << directEnergyVirial.getEnergy("vdw")
              << "\n";

    bondedForce.clear(bondedStream);
    energyVirial.clear(bondedStream);
    bonded.calc_force(xyzq.xyzq, boxx, boxy, boxz, true, true,
                      bondedForce.stride(), bondedForce.xyz(), true, true, true,
                      true, true, false, bondedStream);

    // bonded.print();
    cudaStreamSynchronize(bondedStream);
    cudaDeviceSynchronize();
    energyVirial.copyToHost();
    cudaDeviceSynchronize();

    std::cout << "Bond energy : " << energyVirial.getEnergy("bond") << "\n";
    std::cout << "Urey  energy : " << energyVirial.getEnergy("ureyb") << "\n";
    std::cout << "Angle energy : " << energyVirial.getEnergy("angle") << "\n";
    std::cout << "Dihedral energy : " << energyVirial.getEnergy("dihe") << "\n";
    std::cout << "Imdihedral energy : " << energyVirial.getEnergy("imdihe")
              << "\n";

    cudaDeviceSynchronize();
    // ----------------------Recip-------------
    recipForce.clear(recipStream);
    recipEnergyVirial.clear();

    recip.calc(1.0 / boxx, 1.0 / boxy, 1.0 / boxz, xyzq.xyzq, xyzq.ncoord, true,
               true, recipForce);
    cudaCheck(cudaEventRecord(recipForce_done_event, recipStream));
    cudaStreamSynchronize(recipStream);

    // Adding forces
    recipForce.convert<double>(recipStream);
    recipForce.add<double>(bondedForce, recipStream);

    // Mapping forces back from sorted to actual coordinates
    int nThreads = 128;
    int nBlocks = (numAtoms - 1) / nThreads + 1;
    mapBackdirectsKernel<<<nBlocks, nThreads, 0, directStream>>>(
        directForce.xyz(), directForceSorted.xyz(), directForce.stride(),
        numAtoms, loc2glo);
    cudaStreamSynchronize(directStream);
    recipForce.add<double>(directForce, recipStream);
    cudaStreamSynchronize(recipStream);

    recipEnergyVirial.copyToHost(0);
    cudaDeviceSynchronize();
    printf("ewse : %f\n", recipEnergyVirial.getEnergy("ewse"));
    printf("ewks : %f\n", recipEnergyVirial.getEnergy("ewks"));

    cudaStreamSynchronize(directStream);
    cudaStreamSynchronize(bondedStream);
    cudaStreamSynchronize(recipStream);
    return 0.0;
}

__global__ void updateSortedKernel(float4 *__restrict__ xyzq_sorted,
                                   const float4 *__restrict__ xyzq,
                                   const int *loc2glo, const int numAtoms) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < numAtoms) {
        int j = loc2glo[tid];
        xyzq_sorted[tid] = xyzq[j];
    }
}

double CharmmContext::calculatePotentialEnergy(bool reset) {
    // double energy = 0.0;
    cudaCheck(cudaDeviceSynchronize());
    // bondedForce.clear();
    // movePositionsToEnergy();
    // cudaCheck(cudaDeviceSynchronize());

    // ----------------------Direct-------------
    if (reset) {
        int numThreads = 512;
        int numBlocks = (numAtoms - 1) / numThreads + 1;

        // int loc2glo_len = numAtoms;
        set_loc2glo<<<numBlocks, numThreads>>>(loc2glo, numAtoms);
        cudaDeviceSynchronize();
        // sort coord
        neighborList.sort(0, zone_patom, xyzq.xyzq, xyzq_sorted.xyzq, loc2glo,
                          directStream);
        cudaStreamSynchronize(directStream);

        direct->set_vdwtype(numAtoms, glo_vdwtype,
                            neighborList.get_ind_sorted(), directStream);
        // build neighborList
        neighborList.build(0, zone_patom, boxx, boxy, boxz, cutoff + 1.0,
                           xyzq_sorted.xyzq, loc2glo, directStream);
        cudaStreamSynchronize(directStream);
    } else {
        int numThreads = 512;
        int numBlocks = (numAtoms - 1) / numThreads + 1;

        updateSortedKernel<<<numBlocks, numThreads, 0, directStream>>>(
            xyzq_sorted.xyzq, xyzq.xyzq, loc2glo, numAtoms);
    }

    directForce.clear();
    directForceSorted.clear();
    directEnergyVirial.clear(0);
    direct->calc_force(0, xyzq_sorted.xyzq, (&neighborList)->getBuilder(0),
                       true, true, directForceSorted.stride(),
                       directForceSorted.xyz(), directStream);
    // cudaStreamSynchronize(directStream);
    direct->calc_14_force(xyzq.xyzq, true, true, directForce.stride(),
                          directForce.xyz(), directStream);
    cudaStreamSynchronize(directStream);

    // ----------------------Bonded-------------
    bondedForce.clear(bondedStream);
    bonded.calc_force(xyzq.xyzq, boxx, boxy, boxz, false, false,
                      bondedForce.stride(), bondedForce.xyz(), true, true, true,
                      true, true, false, bondedStream);
    cudaStreamSynchronize(bondedStream);

    // ----------------------Recip-------------
    recipForce.clear(recipStream);
    // recip.calc(1.0 / boxx, 1.0 / boxy, 1.0 / boxz, xyzq.xyzq, xyzq.ncoord,
    // false, false, recipForce);
    cudaStreamSynchronize(recipStream);

    // Adding forces
    recipForce.convert<double>(recipStream);
    recipForce.add<double>(bondedForce, recipStream);

    // Mapping forces back from sorted to actual coordinates
    int nThreads = 512;
    int nBlocks = (numAtoms - 1) / nThreads + 1;
    mapBackdirectsKernel<<<nBlocks, nThreads, 0, directStream>>>(
        directForce.xyz(), directForceSorted.xyz(), directForce.stride(),
        numAtoms, loc2glo);
    cudaStreamSynchronize(directStream);
    cudaStreamSynchronize(recipStream);
    recipForce.add<double>(directForce, recipStream);
    cudaStreamSynchronize(recipStream);
    // printForces(recipForce);
    cudaCheck(cudaDeviceSynchronize());
    // moveForcesToDynamics();
    cudaCheck(cudaDeviceSynchronize());
    return 0.0;
}

void CharmmContext::calculateReciprocalSpace() {
    cudaCheck(cudaDeviceSynchronize());
    // ----------------------Recip-------------
    recipForce.clear(recipStream);
    recip.calc(1.0 / boxx, 1.0 / boxy, 1.0 / boxz, xyzq.xyzq, xyzq.ncoord,
               false, false, recipForce);
    cudaStreamSynchronize(recipStream);
}
void CharmmContext::setMasses(const char *fileName) {
    // Read from the file and set the h_velmass and d_velmass
    std::ifstream infile(fileName);
    for (int i = 0; i < numAtoms; ++i) {
        double mass;
        infile >> mass;
        vel_mass.h[i].w = mass;
        force_invmass.h[i].w = 1.0 / mass;
    }
    vel_mass.c2d();       // copy to device
    force_invmass.c2d();  // copy to device
}

void CharmmContext::assignVelocities() {
    std::cout << "Assigning velocities at temperature " << temperature << " \n";
    double kboltz = 1.987191E-03;
    double boltz = kboltz * temperature;
    for (int i = 0; i < numAtoms; i++) {
        double sd = boltz * force_invmass.h[i].w;
        sd = sqrt(sd);
        double2 r;

        r = randnormal(seed, *randstep.h, i * 2);
        vel_mass.h[i].x = r.x * sd;
        vel_mass.h[i].y = r.y * sd;
        r = randnormal(seed, *randstep.h, i * 2 + 1);
        vel_mass.h[i].z = r.x * sd;
    }
    vel_mass.c2d();  // copy velocites back to device
    // temporarily assign box_dot to zero here, should be normal distributed
    box_dot.h->x = 0.0;
    box_dot.h->y = 0.0;
    box_dot.h->z = 0.0;
    box_dot.c2d();
    *randstep.h = *randstep.h + 1;
    randstep.c2d();  // increment randstep by one, to avoid repeats
}

CudaContainer<double4> CharmmContext::getVelocityMass() { return velocityMass; }

double *CharmmContext::getForces() {
    // bondedForce.convert<double>(0);
    // cudaDeviceSynchronize();

    // bondedForce.add(recipForce.xyz(), numAtoms);
    // recipForce.add(all)
    // recipForce.add<double>(bondedForce,0 );
    return (double *)recipForce.xyz();
}

__global__ void moveForcesToDynamicsKernel(
    const double *__restrict__ forces, int stride,
    double4 *__restrict__ force_invmass, const double *__restrict__ pe_ener,
    double *__restrict__ pe_dyn, const Virial_t *__restrict__ virial_ener,
    double3 *__restrict__ virial_dyn, int numAtoms) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < numAtoms) {
        force_invmass[tid].x = -forces[tid];
        force_invmass[tid].y = -forces[tid + stride];
        force_invmass[tid].z = -forces[tid + 2 * stride];
    }
    if (tid == 0) {
        *pe_dyn = *pe_ener;
        virial_dyn->x = -virial_ener->virmat[0];
        virial_dyn->y = -virial_ener->virmat[4];
        virial_dyn->z = -virial_ener->virmat[8];
        printf("\nvirial is %f,%f,%f\n", virial_dyn->x, virial_dyn->y,
               virial_dyn->z);
    }
}

void CharmmContext::moveForcesToDynamics() {
    int numThreads = 512;
    int numBlocks = (numAtoms - 1) / numThreads + 1;
    auto force = getForces();
    auto stride = directForceSorted.stride();
    std::string vdwString = "vdw";
    box.c2h();
    directEnergyVirial.calcVirial(numAtoms, (getXYZQ())->getDeviceXYZQ(),
                                  box.h->x, box.h->y, box.h->z, stride, force,
                                  0);
    cudaCheck(cudaDeviceSynchronize());
    auto pe_ener = directEnergyVirial.getEnergyPointer(vdwString);
    auto virial_ener = directEnergyVirial.getVirialPointer();
    moveForcesToDynamicsKernel<<<numBlocks, numThreads>>>(
        force, stride, force_invmass.d, pe_ener, potential_energy.d,
        virial_ener, virial.d, numAtoms);
}

__global__ void movePositionsToEnergyKernel(const double4 *__restrict__ xyz,
                                            CT4 *__restrict__ xyzq,
                                            int numAtoms) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < numAtoms) {
        xyzq[tid].x = (CT)xyz[tid].x;
        xyzq[tid].y = (CT)xyz[tid].y;
        xyzq[tid].z = (CT)xyz[tid].z;
    }
}

void CharmmContext::movePositionsToEnergy() {
    box.c2h();  // temporary fix, copy box lens back to host
    boxx = box.h->x;
    boxy = box.h->y;
    boxz = box.h->z;
    direct->set_box_size(box.d);
    int numThreads = 512;
    int numBlocks = (numAtoms - 1) / numThreads + 1;
    CT4 *xyzq = (getXYZQ())->getDeviceXYZQ();
    movePositionsToEnergyKernel<<<numBlocks, numThreads>>>(xyz.d, xyzq,
                                                           numAtoms);
}

int CharmmContext::getNumAtoms() const { return numAtoms; }

XYZQ *CharmmContext::getXYZQ() {
    // return &xyzq_sorted;
    // XYZQ *ret = &xyzq;
    // return ret;
    return &xyzq;
}

__global__ void calculateKineticEnergyKernel1(const double4 *d_velMass,
                                              int numAtoms, double *d_ke) {
    extern __shared__ double sdata[];

    unsigned int id = threadIdx.x;
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < numAtoms) {
        sdata[id] = (d_velMass[tid].x * d_velMass[tid].x +
                     d_velMass[tid].y * d_velMass[tid].y +
                     d_velMass[tid].z * d_velMass[tid].z) *
                    d_velMass[tid].w * 0.5;
    } else {
        sdata[id] = 0.0f;
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (id % (s * 2) == 0) {
            sdata[id] += sdata[id + s];
        }
        __syncthreads();

        if (id == 0) {
            d_ke[blockIdx.x] = (double)sdata[0];
        }
    }
}

__global__ void calculateKineticEnergyKernel2(double *pre, int size,
                                              double *ke) {
    // unsigned int id = threadIdx.x;
    *ke = 0.0;
    for (int i = 0; i < size; ++i) {
        *ke += pre[i];
    }
    // printf("Kinetic Energy : %.6f\n", *ke);
}

void CharmmContext::calculateKineticEnergy() {
    int nThreads = 128;
    int nBlocks = (numAtoms - 1) / nThreads + 1;
    double *d_ke_pre;  //*d_ke ;
    cudaMalloc(&d_ke_pre, nBlocks * sizeof(double));
    // cudaMalloc(&d_ke,  sizeof(double));
    // calculateKineticEnergyKernel1<<<nBlocks, nThreads, nThreads *
    // sizeof(double)>>>(vel_mass.d, numAtoms, d_ke_pre);
    auto da = velocityMass.getDeviceArray();
    auto data = da.data();
    // calculateKineticEnergyKernel1<<<nBlocks, nThreads, nThreads *
    // sizeof(double)>>>(data, numAtoms, d_ke_pre);
    calculateKineticEnergyKernel1<<<nBlocks, nThreads,
                                    nThreads * sizeof(double)>>>(
        (velocityMass.getDeviceArray()).data(), numAtoms, d_ke_pre);
    cudaDeviceSynchronize();
    calculateKineticEnergyKernel2<<<1, 1>>>(d_ke_pre, nBlocks, d_ke);
    cudaDeviceSynchronize();
    // cudaFree(d_ke);
    cudaFree(d_ke_pre);
}

int *CharmmContext::get_loc2glo() const { return loc2glo; }

int CharmmContext::getForceStride() const { return bondedForce.stride(); }
