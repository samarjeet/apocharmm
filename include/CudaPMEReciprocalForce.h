// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#ifndef NOCUDAC

#include "CudaPMERecip.h"
#include "DomdecRecip.h"
#include "Force.h"
#include "PBC.h"
#include "XYZQ.h"
#include <vector>

class CudaPMEReciprocalForce {
private:
  CudaPMERecip<int, float, float2> PMErecip; // PMErecip2;
  PBC pbc;
  int nfftx, nffty, nfftz;
  int order;
  double kappa;
  int numAtoms;

  std::vector<double> boxDimensions;

  CudaContainer<float4> p21FullCellCoords;

  std::shared_ptr<Force<long long int>> forceVal;
  std::shared_ptr<cudaStream_t> recipStream;

  // Energy and virial
  CudaEnergyVirial &energyVirial;

  void solvePoisson(const double inv_boxx, const double inv_boxy,
                    const double inv_boxz, const float4 *coord,
                    const int ncoord, const bool calc_energy,
                    const bool calc_virial, double *recip);

  void solvePoissonBlock(const double inv_boxx, const double inv_boxy,
                         const double inv_boxz, const float4 *coord,
                         const int ncoord, const bool calc_energy,
                         const bool calc_virial, const float *bixlam,
                         const int *blockIndexes, double *recip) {
    for (int i = 0; i < 9; i++)
      recip[i] = 0.0;
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
  CudaPMEReciprocalForce(CudaPMEReciprocalForce &&other)
      : energyVirial(other.energyVirial), PMErecip(std::move(other.PMErecip)),
        // PMErecip2(std::move(other.PMErecip2)),
        nfftx(other.nfftx), nffty(other.nffty), nfftz(other.nfftz),
        order(other.order), kappa(other.kappa), forceVal(other.forceVal),
        recipStream(other.recipStream) {}

  CudaPMEReciprocalForce(CudaEnergyVirial &energyVirial)
      : energyVirial(energyVirial),
        PMErecip(energyVirial) // ,PMErecip2(energyVirial)
  {}
  CudaPMEReciprocalForce(const int nfftx, const int nffty, const int nfftz,
                         const int order, const double kappa,
                         CudaEnergyVirial &energyVirial, const char *nameRecip,
                         const char *nameSelf)
      : // DomdecRecip(nfftx, nffty, nfftz, order, kappa),
        energyVirial(energyVirial),
        PMErecip(nfftx, nffty, nfftz, order, BOX, 1, 0, energyVirial, nameRecip,
                 nameSelf)
  // PMErecip2(nfftx, nffty, nfftz, order, BOX, 1, 0, energyVirial,
  //           nameRecip, nameSelf)
  {}

  //~CudaPMEReciprocalForce() {std::cout << "In CudaPMEReciprocalForce
  // destructor\n";}

  void setPBC(const PBC pbc_in) { pbc = pbc_in; }
  void setParameters(int nfft1, int nfft2, int nfft3, int sp, float k,
                     cudaStream_t stream) {

    if (pbc == PBC::P21) {
      nfft1 = 2 * nfft1;
    }
    PMErecip.setParams(nfft1, nfft2, nfft3, sp, stream);
    // PMErecip2.setParams(nfft1, nfft2, nfft3, sp, stream);
    //  DomdecRecip::setParameters(nfft1, nfft2, nfft3, sp, k );
    nfftx = nfft1;
    nffty = nfft2;
    nfftz = nfft3;
    order = sp;
    kappa = k;
  }
  // void set_stream(cudaStream_t stream) { PMErecip.set_stream(stream); }

  // void clear_energy_virial() {grid.clear_energy_virial();}

  // void get_energy_virial(const bool calc_energy, const bool calc_virial,
  //			 double& energy, double& energy_self, double *virial) {
  // grid.get_energy_virial(kappa, calc_energy, calc_virial, energy,
  // energy_self, virial);
  //}

  //
  // Strided add into Force<long long int>
  //
  void calc(const double inv_boxx, const double inv_boxy, const double inv_boxz,
            const float4 *coord, const int ncoord, const bool calc_energy,
            const bool calc_virial, Force<long long int> &force);

  //
  // Strided store into Force<float>
  //
  void calc(const double inv_boxx, const double inv_boxy, const double inv_boxz,
            const float4 *coord, const int ncoord, const bool calc_energy,
            const bool calc_virial, Force<float> &force) {
    double recip[9];
    solvePoisson(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy,
                 calc_virial, recip);
    PMErecip.gather_force(coord, ncoord, recip, force.stride(), force.xyz());
    cudaCheck(cudaDeviceSynchronize());
    if (calc_energy)
      PMErecip.calc_self_energy(coord, ncoord, this->kappa);
  }

  //
  // Non-strided store info XYZQ
  //
  void calc(const double inv_boxx, const double inv_boxy, const double inv_boxz,
            const float4 *coord, const int ncoord, const bool calc_energy,
            const bool calc_virial, float3 *force) {
    double recip[9];
    solvePoisson(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy,
                 calc_virial, recip);
    PMErecip.gather_force(coord, ncoord, recip, 1, force);
    if (calc_energy)
      PMErecip.calc_self_energy(coord, ncoord, this->kappa);
  }

  // begin block calc

  // Strided add into Force<long long int>
  void calc_block(const double inv_boxx, const double inv_boxy,
                  const double inv_boxz, const float4 *coord, const int ncoord,
                  const float *bixlam, const bool calc_energy,
                  const bool calc_virial, Force<long long int> &force,
                  double *biflam, // outputs
                  const int *blockIndexes) {
    double recip[9];
    solvePoissonBlock(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy,
                      calc_virial, bixlam, blockIndexes, recip);
    PMErecip.gather_force_block(coord, ncoord, recip, force.stride(),
                                force.xyz(), bixlam, biflam, blockIndexes);
    // if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
    // MSLDPME ->
    PMErecip.calc_self_energy_block(coord, ncoord, this->kappa, bixlam, biflam,
                                    blockIndexes);
  }

  // Strided store into Force<float>
  void calc_block(const double inv_boxx, const double inv_boxy,
                  const double inv_boxz, const float4 *coord, const int ncoord,
                  const float *bixlam, const bool calc_energy,
                  const bool calc_virial, Force<float> &force,
                  double *biflam, // outputs
                  const int *blockIndexes) {
    double recip[9];
    solvePoissonBlock(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy,
                      calc_virial, bixlam, blockIndexes, recip);
    PMErecip.gather_force_block(coord, ncoord, recip, force.stride(),
                                force.xyz(), bixlam, biflam, blockIndexes);
    // if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
    // MSLDPME ->
    PMErecip.calc_self_energy_block(coord, ncoord, this->kappa, bixlam, biflam,
                                    blockIndexes);
  }

  // Non-strided store info XYZQ
  void calc_block(const double inv_boxx, const double inv_boxy,
                  const double inv_boxz, const float4 *coord, const int ncoord,
                  const float *bixlam, const bool calc_energy,
                  const bool calc_virial, float3 *force,
                  double *biflam, // outputs
                  const int *blockIndexes) {
    double recip[9];
    solvePoissonBlock(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy,
                      calc_virial, bixlam, blockIndexes, recip);
    PMErecip.gather_force_block(coord, ncoord, recip, 1, force, bixlam, biflam,
                                blockIndexes);
    // if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
    // MSLDPME ->
    PMErecip.calc_self_energy_block(coord, ncoord, this->kappa, bixlam, biflam,
                                    blockIndexes);
  }

  // end block calc

  // TODO : remove this getter - just for test
  int getFFTX();

  void setForce(std::shared_ptr<Force<long long int>> &forceValIn) {
    forceVal = forceValIn;
  }

  void setStream(std::shared_ptr<cudaStream_t> streamIn) {
    recipStream = streamIn;
    PMErecip.set_stream(*recipStream);
  }

  void setNumAtoms(int n) {
    if (pbc == PBC::P21)
      p21FullCellCoords.resize(2 * n);
    numAtoms = n;
  }

  void setBoxDimensions(std::vector<double> dim) { boxDimensions = dim; }

  void clear(void);

  void calc_force(const float4 *xyzq, bool calcEnergy, bool calcVirial) {
    // set_stream(*recipStream);
    calc(1.0 / boxDimensions[0], 1.0 / boxDimensions[1], 1.0 / boxDimensions[2],
         xyzq, numAtoms, calcEnergy, calcVirial, *forceVal);
  }
};

#endif // NOCUDAC
