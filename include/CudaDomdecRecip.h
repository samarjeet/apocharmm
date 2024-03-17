// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#ifndef NOCUDAC
#ifndef CUDADOMDECRECIP_H
#define CUDADOMDECRECIP_H

#include "DomdecRecip.h"
#include <vector>
#include "Force.h"
#include "XYZQ.h"
#include "CudaPMERecip.h"

// class CudaDomdecRecip : public DomdecRecip {
class CudaDomdecRecip  {
   private:
    CudaPMERecip<int, float, float2> PMErecip;
    int nfftx, nffty, nfftz;
    int order;
    double kappa;
    int numAtoms;

    std::vector<double> boxDimensions;

    std::shared_ptr<Force<long long int>> forceVal;
    std::shared_ptr<cudaStream_t> recipStream;


    // Energy and virial
    CudaEnergyVirial& energyVirial;

    void solvePoisson(const double inv_boxx, const double inv_boxy, const double inv_boxz, const float4* coord,
                      const int ncoord, const bool calc_energy, const bool calc_virial, double* recip) {
        for (int i = 0; i < 9; i++) recip[i] = 0.0;
        recip[0] = inv_boxx;
        recip[4] = inv_boxy;
        recip[8] = inv_boxz;
        PMErecip.spread_charge(coord, ncoord, recip);
        PMErecip.r2c_fft();
        PMErecip.scalar_sum(recip, kappa, calc_energy, calc_virial);
        PMErecip.c2r_fft();
    }

    void solvePoissonBlock(const double inv_boxx, const double inv_boxy, const double inv_boxz, const float4* coord,
                           const int ncoord, const bool calc_energy, const bool calc_virial, const float* bixlam,
                           const int* blockIndexes, double* recip) {
        for (int i = 0; i < 9; i++) recip[i] = 0.0;
        recip[0] = inv_boxx;
        recip[4] = inv_boxy;
        recip[8] = inv_boxz;
        PMErecip.spread_charge_block(coord, ncoord, bixlam, blockIndexes, recip);
        PMErecip.r2c_fft();
        PMErecip.scalar_sum(recip, kappa, calc_energy, calc_virial);
        PMErecip.c2r_fft();
    }

   public:
    // move constructor
    CudaDomdecRecip(CudaDomdecRecip && other) : energyVirial(other.energyVirial), PMErecip(std::move(other.PMErecip)), nfftx(other.nfftx), nffty(other.nffty), nfftz(other.nfftz), order(other.order), kappa(other.kappa), forceVal(other.forceVal), recipStream(other.recipStream) {}

    CudaDomdecRecip(CudaEnergyVirial& energyVirial) : energyVirial(energyVirial), PMErecip(energyVirial) {}
    CudaDomdecRecip(const int nfftx, const int nffty, const int nfftz, const int order, const double kappa,
                    CudaEnergyVirial& energyVirial, const char* nameRecip, const char* nameSelf)
        :  // DomdecRecip(nfftx, nffty, nfftz, order, kappa),
          energyVirial(energyVirial),
          PMErecip(nfftx, nffty, nfftz, order, BOX, 1, 0, energyVirial, nameRecip, nameSelf) {}

    //~CudaDomdecRecip() {std::cout << "In CudaDomdecRecip destructor\n";}

    void setParameters(int nfft1, int nfft2, int nfft3, int sp, float k, cudaStream_t stream) {
        PMErecip.setParams(nfft1, nfft2, nfft3, sp, stream);
        // DomdecRecip::setParameters(nfft1, nfft2, nfft3, sp, k );
        nfftx = nfft1;
        nffty = nfft2;
        nfftz = nfft3;
        order = sp;
        kappa = k;
    }
    void set_stream(cudaStream_t stream) { PMErecip.set_stream(stream); }

    // void clear_energy_virial() {grid.clear_energy_virial();}

    // void get_energy_virial(const bool calc_energy, const bool calc_virial,
    //			 double& energy, double& energy_self, double *virial) {
    // grid.get_energy_virial(kappa, calc_energy, calc_virial, energy, energy_self, virial);
    //}

    //
    // Strided add into Force<long long int>
    //
    void calc(const double inv_boxx, const double inv_boxy, const double inv_boxz, const float4* coord,
              const int ncoord, const bool calc_energy, const bool calc_virial, Force<long long int>& force) {
        double recip[9];
        solvePoisson(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, recip);
        PMErecip.gather_force(coord, ncoord, recip, force.stride(), force.xyz());
        if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
    }

    //
    // Strided store into Force<float>
    //
    void calc(const double inv_boxx, const double inv_boxy, const double inv_boxz, const float4* coord,
              const int ncoord, const bool calc_energy, const bool calc_virial, Force<float>& force) {
        double recip[9];
        solvePoisson(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, recip);
        PMErecip.gather_force(coord, ncoord, recip, force.stride(), force.xyz());
        cudaCheck(cudaDeviceSynchronize());
        if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
    }

    //
    // Non-strided store info XYZQ
    //
    void calc(const double inv_boxx, const double inv_boxy, const double inv_boxz, const float4* coord,
              const int ncoord, const bool calc_energy, const bool calc_virial, float3* force) {
        double recip[9];
        solvePoisson(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, recip);
        PMErecip.gather_force(coord, ncoord, recip, 1, force);
        if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
    }

    // begin block calc

    // Strided add into Force<long long int>
    void calc_block(const double inv_boxx, const double inv_boxy, const double inv_boxz, const float4* coord,
                    const int ncoord, const float* bixlam, const bool calc_energy, const bool calc_virial,
                    Force<long long int>& force, double* biflam,  // outputs
                    const int* blockIndexes) {
        double recip[9];
        solvePoissonBlock(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, bixlam, blockIndexes,
                          recip);
        PMErecip.gather_force_block(coord, ncoord, recip, force.stride(), force.xyz(), bixlam, biflam, blockIndexes);
        // if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
        // MSLDPME ->
        PMErecip.calc_self_energy_block(coord, ncoord, this->kappa, bixlam, biflam, blockIndexes);
    }

    // Strided store into Force<float>
    void calc_block(const double inv_boxx, const double inv_boxy, const double inv_boxz, const float4* coord,
                    const int ncoord, const float* bixlam, const bool calc_energy, const bool calc_virial,
                    Force<float>& force, double* biflam,  // outputs
                    const int* blockIndexes) {
        double recip[9];
        solvePoissonBlock(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, bixlam, blockIndexes,
                          recip);
        PMErecip.gather_force_block(coord, ncoord, recip, force.stride(), force.xyz(), bixlam, biflam, blockIndexes);
        // if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
        // MSLDPME ->
        PMErecip.calc_self_energy_block(coord, ncoord, this->kappa, bixlam, biflam, blockIndexes);
    }

    // Non-strided store info XYZQ
    void calc_block(const double inv_boxx, const double inv_boxy, const double inv_boxz, const float4* coord,
                    const int ncoord, const float* bixlam, const bool calc_energy, const bool calc_virial,
                    float3* force, double* biflam,  // outputs
                    const int* blockIndexes) {
        double recip[9];
        solvePoissonBlock(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, bixlam, blockIndexes,
                          recip);
        PMErecip.gather_force_block(coord, ncoord, recip, 1, force, bixlam, biflam, blockIndexes);
        // if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
        // MSLDPME ->
        PMErecip.calc_self_energy_block(coord, ncoord, this->kappa, bixlam, biflam, blockIndexes);
    }

    // end block calc

    // TODO : remove this getter - just for test
    int getFFTX() { return nfftx; }

    void setForce(std::shared_ptr<Force<long long int>> & forceValIn){
      forceVal = forceValIn;
    }

    void setStream(std::shared_ptr<cudaStream_t> streamIn){
      recipStream = streamIn;
    }

    void setNumAtoms(int n){
      numAtoms = n;
    }

    void setBoxDimensions(std::vector<double> dim){
      boxDimensions = dim;
    }

    void calc_force(const float4* xyzq){
      set_stream(*recipStream);
      calc(1.0/boxDimensions[0] , 1.0/boxDimensions[1], 1.0/boxDimensions[2], xyzq, numAtoms, true, true, *forceVal );
      cudaCheck(cudaDeviceSynchronize()); 
      //std::cout << "calculating reciprocal force. nffty = " << nffty << "\n";

    }
};

#endif  // CUDADOMDECRECIP_H
#endif  // NOCUDAC
