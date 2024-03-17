// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaPMEReciprocalForce.h"

__global__ void createFullUnitCellKernel(int numAtoms, float xdim,
                                         const float4 *__restrict__ coords,
                                         float4 *__restrict__ fullCoords) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numAtoms) {
    fullCoords[i] = coords[i];
    fullCoords[numAtoms + i].x = coords[i].x + xdim;
    fullCoords[numAtoms + i].y = -coords[i].y;
    fullCoords[numAtoms + i].z = -coords[i].z;
    fullCoords[numAtoms + i].w = coords[i].w;
  }
}

__global__ void updateEWKS(double *ewksEnergy) { *ewksEnergy /= 2.0; }

void CudaPMEReciprocalForce::solvePoisson(
    const double inv_boxx, const double inv_boxy, const double inv_boxz,
    const float4 *coord, const int ncoord, const bool calc_energy,
    const bool calc_virial, double *recip) {
  for (int i = 0; i < 9; i++)
    recip[i] = 0.0;
  recip[0] = inv_boxx;
  recip[4] = inv_boxy;
  recip[8] = inv_boxz;

  if (pbc == PBC::P1) {
    PMErecip.spread_charge(coord, ncoord, recip);
    PMErecip.r2c_fft();
    PMErecip.scalar_sum(recip, kappa, calc_energy, calc_virial);
    PMErecip.c2r_fft();
  }

  else if (pbc == PBC::P21) {
    recip[0] = 0.5 * inv_boxx;
    // PMErecip.spread_charge(coord, ncoord, recip);
    //  flip and fill the rest of the box for P21
    // PMErecip.fillP21Grid(ncoord);

    auto p21coords = p21FullCellCoords.getDeviceArray().data();

    int numThreads = 128;
    int numBlocks = (ncoord - 1) / numThreads + 1;

    createFullUnitCellKernel<<<numBlocks, numThreads, 0, *recipStream>>>(
        ncoord, 1.0f / inv_boxx, coord, p21coords);

    PMErecip.spread_charge(p21coords, ncoord * 2, recip);
    PMErecip.r2c_fft();
    PMErecip.scalar_sum(recip, kappa, calc_energy, calc_virial);

    // energyVirial
    auto ewksPointer = energyVirial.getEnergyPointer("ewks");
    updateEWKS<<<1, 1, 0, *recipStream>>>(ewksPointer);

    PMErecip.c2r_fft();
  }
}
int CudaPMEReciprocalForce::getFFTX() { return nfftx; }

//
// Strided add into Force<long long int>
//
void CudaPMEReciprocalForce::calc(const double inv_boxx, const double inv_boxy,
                                  const double inv_boxz, const float4 *coord,
                                  const int ncoord, const bool calc_energy,
                                  const bool calc_virial,
                                  Force<long long int> &force) {
  double recip[9];

  solvePoisson(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy,
               calc_virial, recip);
  // TODO : for P21, gather_force only on asymmetric unit
  if (pbc == PBC::P1)
    PMErecip.gather_force(coord, ncoord, recip, force.stride(), force.xyz());
  else if (pbc == PBC::P21) {
    PMErecip.gather_force(coord, ncoord, recip, force.stride(), force.xyz());
  }

  if (calc_energy)
    PMErecip.calc_self_energy(coord, ncoord, this->kappa);
}