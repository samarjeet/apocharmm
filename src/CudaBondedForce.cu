// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#ifndef NOCUDAC
#include "CudaBondedForce.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <numeric>

// Energy and virial in device memory
// static __device__ BondedEnergyVirial_t d_energy_virial;

//
// Reduces energy values
//
__forceinline__ __device__ void reduce_energy(const double epot,
                                              volatile double *sh_epot,
                                              double *global_epot) {
#if __CUDA_ARCH__ < 300
  sh_epot[threadIdx.x] = epot;
  __syncthreads();
  for (int i = 1; i < blockDim.x; i *= 2) {
    int t = threadIdx.x + i;
    double epot_val = (t < blockDim.x) ? sh_epot[t] : 0.0;
    __syncthreads();
    sh_epot[threadIdx.x] += epot_val;
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(global_epot, sh_epot[0]);
#else
  const int tid = threadIdx.x & (warpsize - 1);
  const int base = (threadIdx.x / warpsize);
  // Reduce within warps
  double energy = epot;
  for (int d = warpsize / 2; d >= 1; d /= 2) {
    energy += __hiloint2double(SHFL(__double2hiint(energy), tid + d),
                               SHFL(__double2loint(energy), tid + d));
  }
  // Reduce between warps
  // NOTE: this __syncthreads() is needed because we're using a single shared
  // memory buffer
  __syncthreads();
  if (tid == 0)
    sh_epot[base] = energy;
  __syncthreads();
  if (base == 0) {
    energy = (tid < blockDim.x / warpsize) ? sh_epot[tid] : 0.0;
    for (int d = warpsize / 2; d >= 1; d /= 2) {
      energy += __hiloint2double(SHFL(__double2hiint(energy), tid + d),
                                 SHFL(__double2loint(energy), tid + d));
    }
  }
  if (threadIdx.x == 0)
    atomicAdd(global_epot, energy);
#endif
}

//
// Templated sqrt() -function
//
template <typename T>
__forceinline__ __device__ double sqrt_template(const T x) {
  if (sizeof(T) == 4) {
    return sqrtf(x);
  } else {
    return sqrt(x);
  }
}

template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__device__ void calc_bond_force_device(
    const int pos, const bondlist_t *__restrict__ bondlist,
    const float2 *__restrict__ bondcoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double &epot, Virial_t *__restrict__ virial) {
  int ii = bondlist[pos].i;
  int jj = bondlist[pos].j;
  int ic = bondlist[pos].itype;
  int ish = bondlist[pos].ishift;

  // Calculate shift for i-atom
  CT shx, shy, shz;
  calc_box_shift<CT>(ish, boxx, boxy, boxz, shx, shy, shz);

  float4 xyzqi = xyzq[ii];
  float4 xyzqj = xyzq[jj];

  CT dx = xyzqi.x + shx - xyzqj.x;
  CT dy = xyzqi.y + shy - xyzqj.y;
  CT dz = xyzqi.z + shz - xyzqj.z;

  CT r = sqrt_template<CT>(dx * dx + dy * dy + dz * dz);

  float2 bondcoef_val = bondcoef[ic];
  CT db = r - (CT)bondcoef_val.x;
  CT fij = db * (CT)bondcoef_val.y;

  if (calc_energy) {
    epot += (double)(fij * db);
  }
  fij *= ((CT)2) / r;

  AT fxij, fyij, fzij;
  calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

  // Store forces
  write_force<AT>(fxij, fyij, fzij, ii, stride, force);
  write_force<AT>(-fxij, -fyij, -fzij, jj, stride, force);

  // Store virial
  if (calc_virial) {
#ifdef USE_DP_SFORCE
    if (ish != 13) {
      atomicAdd(&virial->sforce_dp[ish][0], (double)(fij * dx));
      atomicAdd(&virial->sforce_dp[ish][1], (double)(fij * dy));
      atomicAdd(&virial->sforce_dp[ish][2], (double)(fij * dz));
    }
#else
    if (ish != 13) {
      fxij /= CONVERT_TO_VIR;
      fyij /= CONVERT_TO_VIR;
      fzij /= CONVERT_TO_VIR;
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][0],
                llitoulli(fxij));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][1],
                llitoulli(fyij));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][2],
                llitoulli(fzij));
    }
#endif
  }
}

//
// bondcoef.x = cbb
// bondcoef.y = cbc
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_bond_force_kernel(
    const int nbondlist, const bondlist_t *__restrict__ bondlist,
    const float2 *__restrict__ bondcoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_bond,
    Virial_t *__restrict__ virial) {
  // Amount of shared memory required:
  // CUDA_ARCH <  300: blockDim.x*sizeof(double)
  // CUDA_ARCH >= 300: (blockDim.x/warpsize)*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy) {
    epot = 0.0;
  }

  while (pos < nbondlist) {
    calc_bond_force_device<AT, CT, calc_energy, calc_virial>(
        pos, bondlist, bondcoef, xyzq, stride, boxx, boxy, boxz, force, epot,
        virial);
    pos += blockDim.x * gridDim.x;
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_bond);
  }
  // if (threadIdx.x == 0 )
  //  printf("after pos : %d, energy: %f\n ", pos - blockDim.x * gridDim.x,
  //  *energy_bond);
}

//
// bondcoef.x = cbb
// bondcoef.y = cbc
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_ureyb_force_kernel(
    const int nureyblist, const bondlist_t *__restrict__ ureyblist,
    const float2 *__restrict__ ureybcoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_ureyb,
    Virial_t *__restrict__ virial) {
  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy) {
    epot = 0.0;
  }

  while (pos < nureyblist) {
    calc_bond_force_device<AT, CT, calc_energy, calc_virial>(
        pos, ureyblist, ureybcoef, xyzq, stride, boxx, boxy, boxz, force, epot,
        virial);
    pos += blockDim.x * gridDim.x;
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_ureyb);
  }
}

template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__device__ void calc_angle_force_device(
    const int pos, const anglelist_t *__restrict__ anglelist,
    const float2 *__restrict__ anglecoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double &epot, Virial_t *__restrict__ virial) {
  int ii = anglelist[pos].i;
  int jj = anglelist[pos].j;
  int kk = anglelist[pos].k;
  int ic = anglelist[pos].itype;
  int ish = anglelist[pos].ishift1;
  int ksh = anglelist[pos].ishift2;

  // Calculate shift for i-atom
  CT ishx, ishy, ishz;
  calc_box_shift<CT>(ish, boxx, boxy, boxz, ishx, ishy, ishz);

  // Calculate shift for k-atom
  CT kshx, kshy, kshz;
  calc_box_shift<CT>(ksh, boxx, boxy, boxz, kshx, kshy, kshz);

  CT dxij = xyzq[ii].x + ishx - xyzq[jj].x;
  CT dyij = xyzq[ii].y + ishy - xyzq[jj].y;
  CT dzij = xyzq[ii].z + ishz - xyzq[jj].z;

  CT dxkj = xyzq[kk].x + kshx - xyzq[jj].x;
  CT dykj = xyzq[kk].y + kshy - xyzq[jj].y;
  CT dzkj = xyzq[kk].z + kshz - xyzq[jj].z;

  CT rij = sqrtf(dxij * dxij + dyij * dyij + dzij * dzij);
  CT rkj = sqrtf(dxkj * dxkj + dykj * dykj + dzkj * dzkj);

  CT rij_inv = ((CT)1) / rij;
  CT rkj_inv = ((CT)1) / rkj;

  CT dxijr = dxij * rij_inv;
  CT dyijr = dyij * rij_inv;
  CT dzijr = dzij * rij_inv;
  CT dxkjr = dxkj * rkj_inv;
  CT dykjr = dykj * rkj_inv;
  CT dzkjr = dzkj * rkj_inv;
  CT cst = dxijr * dxkjr + dyijr * dykjr + dzijr * dzkjr;

  // anglecoef.x = ctb
  // anglecoef.y = ctc
  float2 anglecoef_val = anglecoef[ic];
  // printf("%d %d %d %d %f %f\n", pos, ii, jj, kk, anglecoef_val.x*57.295,
  // anglecoef_val.y);

  // Restrict values of cst to the interval [-0.999 ... 0.999]
  // NOTE: we are ignoring the fancy stuff that is done on the CPU version
  cst = min((CT)0.999, max(-(CT)0.999, cst));

  CT at = acosf(cst);
  CT da = at - (CT)anglecoef_val.x;
  CT df = ((CT)anglecoef_val.y) * da;
  if (calc_energy) {
    epot += (double)(df * da);
  }
  CT st2r = ((CT)1.0) / (((CT)1.0) - cst * cst);
  CT str = sqrtf(st2r);
  df = -((CT)2.0) * df * str;

  CT dtxi = rij_inv * (dxkjr - cst * dxijr);
  CT dtxj = rkj_inv * (dxijr - cst * dxkjr);
  CT dtyi = rij_inv * (dykjr - cst * dyijr);
  CT dtyj = rkj_inv * (dyijr - cst * dykjr);
  CT dtzi = rij_inv * (dzkjr - cst * dzijr);
  CT dtzj = rkj_inv * (dzijr - cst * dzkjr);

  AT AT_dtxi, AT_dtyi, AT_dtzi;
  AT AT_dtxj, AT_dtyj, AT_dtzj;
  calc_component_force<AT, CT>(df, dtxi, dtyi, dtzi, AT_dtxi, AT_dtyi, AT_dtzi);
  calc_component_force<AT, CT>(df, dtxj, dtyj, dtzj, AT_dtxj, AT_dtyj, AT_dtzj);

  write_force<AT>(AT_dtxi, AT_dtyi, AT_dtzi, ii, stride, force);
  write_force<AT>(AT_dtxj, AT_dtyj, AT_dtzj, kk, stride, force);
  write_force<AT>(-AT_dtxi - AT_dtxj, -AT_dtyi - AT_dtyj, -AT_dtzi - AT_dtzj,
                  jj, stride, force);

  // Store virial
  if (calc_virial) {
#ifdef USE_DP_SFORCE
    if (ish != 13) {
      atomicAdd(&virial->sforce_dp[ish][0], (double)(df * dtxi));
      atomicAdd(&virial->sforce_dp[ish][1], (double)(df * dtyi));
      atomicAdd(&virial->sforce_dp[ish][2], (double)(df * dtzi));
    }
    if (ksh != 13) {
      atomicAdd(&virial->sforce_dp[ksh][0], (double)(df * dtxj));
      atomicAdd(&virial->sforce_dp[ksh][1], (double)(df * dtyj));
      atomicAdd(&virial->sforce_dp[ksh][2], (double)(df * dtzj));
    }
#else
    if (ish != 13) {
      AT_dtxi /= CONVERT_TO_VIR;
      AT_dtyi /= CONVERT_TO_VIR;
      AT_dtzi /= CONVERT_TO_VIR;
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][0],
                llitoulli(AT_dtxi));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][1],
                llitoulli(AT_dtyi));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][2],
                llitoulli(AT_dtzi));
    }
    if (ksh != 13) {
      AT_dtxj /= CONVERT_TO_VIR;
      AT_dtyj /= CONVERT_TO_VIR;
      AT_dtzj /= CONVERT_TO_VIR;
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ksh][0],
                llitoulli(AT_dtxj));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ksh][1],
                llitoulli(AT_dtyj));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ksh][2],
                llitoulli(AT_dtzj));
    }
#endif
  }
}

//
// anglecoef.x = ctb
// anglecoef.y = ctc
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_angle_force_kernel(
    const int nanglelist, const anglelist_t *__restrict__ anglelist,
    const float2 *__restrict__ anglecoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_angle,
    Virial_t *__restrict__ virial) {
  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy)
    epot = 0.0;

  while (pos < nanglelist) {
    calc_angle_force_device<AT, CT, calc_energy, calc_virial>(
        pos, anglelist, anglecoef, xyzq, stride, boxx, boxy, boxz, (AT *)force,
        epot, virial);
    pos += blockDim.x * gridDim.x;
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_angle);
  }
}

//
// Dihedral potential
//
// dihecoef.x = cpd (integer)
// dihecoef.y = cpc
// dihecoef.z = cpsin
// dihecoef.w = cpcos
//
// Out: df, e
//
template <typename T, bool calc_energy>
__forceinline__ __device__ void dihe_pot(const float4 *dihecoef,
                                         const int ic_in, const T st,
                                         const T ct, T &df, double &e) {
  df = (T)0;
  if (calc_energy)
    e = 0.0;
  int ic = ic_in;

  bool lrep = true;
  while (lrep) {
    float4 dihecoef_val = dihecoef[ic];
    // printf("%d %d %.2f %.1f %.1f\n", threadIdx.x,
    // dihecoef_val.x,dihecoef_val.y ,dihecoef_val.z,   dihecoef_val.w );

    int iper = (int)dihecoef_val.x;
    lrep = (iper > 0) ? false : true;
    iper = abs(iper);

    T e1 = (T)1;
    T df1 = (T)0;
    T ddf1 = (T)0;

    // Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
    for (int nper = 1; nper <= iper; nper++) {
      ddf1 = e1 * ct - df1 * st;
      df1 = e1 * st + df1 * ct;
      e1 = ddf1;
    }
    if (calc_energy)
      e1 = e1 * dihecoef_val.w + df1 * dihecoef_val.z;
    df1 = df1 * dihecoef_val.w - ddf1 * dihecoef_val.z;
    df1 = -iper * df1;
    if (calc_energy) {
      e1 += (T)1;
      if (iper == 0)
        e1 = (T)1;
    }

    float arg = dihecoef_val.y;
    if (calc_energy)
      e += arg * e1;
    df += arg * df1;

    ic++;
  }
}

//
// Improper dihedral potential
//
// imdihecoef.x = cid (integer)
// imdihecoef.y = cic
// imdihecoef.z = cisin
// imdihecoef.w = cicos
//
// Out: df, e
//
template <typename T, bool calc_energy>
__forceinline__ __device__ void imdihe_pot(const float4 *dihecoef,
                                           const int ic_in, const T st,
                                           const T ct, T &df, double &e) {
  df = (T)0;
  if (calc_energy)
    e = 0.0;

  float4 dihecoef_val = dihecoef[ic_in];

  if ((int)dihecoef_val.x != 0) {
    int ic = ic_in;
    bool lrep = true;
    while (lrep) {
      int iper = (int)dihecoef_val.x;
      lrep = (iper > 0) ? false : true;
      iper = abs(iper);

      T e1 = (T)1;
      T df1 = (T)0;
      T ddf1 = (T)0;

      // Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
      for (int nper = 1; nper <= iper; nper++) {
        ddf1 = e1 * ct - df1 * st;
        df1 = e1 * st + df1 * ct;
        e1 = ddf1;
      }
      if (calc_energy)
        e1 = e1 * dihecoef_val.w + df1 * dihecoef_val.z;
      df1 = df1 * dihecoef_val.w - ddf1 * dihecoef_val.z;
      df1 = -iper * df1;
      if (calc_energy) {
        e1 += (T)1;
        if (iper == 0)
          e1 = (T)1;
      }

      float arg = dihecoef_val.y;
      if (calc_energy)
        e += arg * e1;
      df += arg * df1;

      ic++;
      if (lrep)
        dihecoef_val = dihecoef[ic];
    }
    // use harmonic potential
  } else {
    // calcul of cos(phi-phi0),sin(phi-phi0) and (phi-phi0).

    T ca = ct * dihecoef_val.w + st * dihecoef_val.z;
    T sa = st * dihecoef_val.w - ct * dihecoef_val.z;
    T ap;
    if (ca > (T)0.1) {
      ap = asinf(sa);
    } else {
      // ap = sign(acos(max(ca,-(T)1)),sa);
      ap = acosf(max(ca, -(T)1));
      ap = (sa > (T)0) ? ap : -ap;
      // warning is now triggered at deltaphi=84.26...deg (used to be 90).
      // nbent = nbent + 1;
    }

    df = dihecoef_val.y * ap;
    if (calc_energy)
      e = df * ap;
    df *= (T)2;
  }
}

template <typename AT, typename CT, bool q_dihe, bool calc_energy,
          bool calc_virial>
__device__ void calc_dihe_force_device(
    const int pos, const dihelist_t *__restrict__ dihelist,
    const float4 *__restrict__ dihecoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double &epot, Virial_t *__restrict__ virial) {
  int ii = dihelist[pos].i;
  int jj = dihelist[pos].j;
  int kk = dihelist[pos].k;
  int ll = dihelist[pos].l;
  int ic = dihelist[pos].itype;
  int ish = dihelist[pos].ishift1;
  int jsh = dihelist[pos].ishift2;
  int lsh = dihelist[pos].ishift3;

  // Calculate shift for i-atom
  CT ishx, ishy, ishz;
  calc_box_shift<CT>(ish, boxx, boxy, boxz, ishx, ishy, ishz);

  // Calculate shift for j-atom
  CT jshx, jshy, jshz;
  calc_box_shift<CT>(jsh, boxx, boxy, boxz, jshx, jshy, jshz);

  // Calculate shift for l-atom
  CT lshx, lshy, lshz;
  calc_box_shift<CT>(lsh, boxx, boxy, boxz, lshx, lshy, lshz);

  CT fx = (xyzq[ii].x + ishx) - (xyzq[jj].x + jshx);
  CT fy = (xyzq[ii].y + ishy) - (xyzq[jj].y + jshy);
  CT fz = (xyzq[ii].z + ishz) - (xyzq[jj].z + jshz);

  CT gx = xyzq[jj].x + jshx - xyzq[kk].x;
  CT gy = xyzq[jj].y + jshy - xyzq[kk].y;
  CT gz = xyzq[jj].z + jshz - xyzq[kk].z;

  CT hx = xyzq[ll].x + lshx - xyzq[kk].x;
  CT hy = xyzq[ll].y + lshy - xyzq[kk].y;
  CT hz = xyzq[ll].z + lshz - xyzq[kk].z;

  // A=F^G, B=H^G.
  CT ax = fy * gz - fz * gy;
  CT ay = fz * gx - fx * gz;
  CT az = fx * gy - fy * gx;
  CT bx = hy * gz - hz * gy;
  CT by = hz * gx - hx * gz;
  CT bz = hx * gy - hy * gx;

  CT ra2 = ax * ax + ay * ay + az * az;
  CT rb2 = bx * bx + by * by + bz * bz;
  CT rg = sqrtf(gx * gx + gy * gy + gz * gz);

  //    if((ra2 <= rxmin2) .or. (rb2 <= rxmin2) .or. (rg <= rxmin)) then
  //          nlinear = nlinear + 1
  //       endif

  CT rgr = 1.0f / rg;
  CT ra2r = 1.0f / ra2;
  CT rb2r = 1.0f / rb2;
  CT rabr = sqrtf(ra2r * rb2r);

  // ct=cos(phi)
  CT ct = (ax * bx + ay * by + az * bz) * rabr;
  //
  // ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
  // which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
  CT st = rg * rabr * (ax * hx + ay * hy + az * hz);
  //
  //     Energy and derivative contributions.

  CT df;
  double e;
  if (q_dihe) {
    dihe_pot<CT, calc_energy>(dihecoef, ic, st, ct, df, e);
  } else {
    imdihe_pot<CT, calc_energy>(dihecoef, ic, st, ct, df, e);
  }

  if (calc_energy)
    epot += e;

  //
  //     Compute derivatives wrt catesian coordinates.
  //
  // GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
  //  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)

  CT fg = fx * gx + fy * gy + fz * gz;
  CT hg = hx * gx + hy * gy + hz * gz;
  ra2r *= df;
  rb2r *= df;
  CT fga = fg * ra2r * rgr;
  CT hgb = hg * rb2r * rgr;
  CT gaa = ra2r * rg;
  CT gbb = rb2r * rg;
  // DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.

  // Store forces
  AT dfx, dfy, dfz;
  calc_component_force<AT, CT>(-gaa, ax, ay, az, dfx, dfy, dfz);
  write_force<AT>(dfx, dfy, dfz, ii, stride, force);

  AT dgx, dgy, dgz;
  calc_component_force<AT, CT>(fga, ax, ay, az, -hgb, bx, by, bz, dgx, dgy,
                               dgz);
  write_force<AT>(dgx - dfx, dgy - dfy, dgz - dfz, jj, stride, force);

  AT dhx, dhy, dhz;
  calc_component_force<AT, CT>(gbb, bx, by, bz, dhx, dhy, dhz);
  write_force<AT>(-dhx - dgx, -dhy - dgy, -dhz - dgz, kk, stride, force);
  write_force<AT>(dhx, dhy, dhz, ll, stride, force);

  // Store virial
  if (calc_virial) {
#ifdef USE_DP_SFORCE
    if (ish != 13) {
      atomicAdd(&virial->sforce_dp[ish][0], (double)(-gaa * ax));
      atomicAdd(&virial->sforce_dp[ish][1], (double)(-gaa * ay));
      atomicAdd(&virial->sforce_dp[ish][2], (double)(-gaa * az));
    }
    if (jsh != 13) {
      atomicAdd(&virial->sforce_dp[jsh][0],
                (double)(fga * ax - hgb * bx + gaa * ax));
      atomicAdd(&virial->sforce_dp[jsh][1],
                (double)(fga * ay - hgb * by + gaa * ay));
      atomicAdd(&virial->sforce_dp[jsh][2],
                (double)(fga * az - hgb * bz + gaa * az));
    }
    if (lsh != 13) {
      atomicAdd(&virial->sforce_dp[lsh][0], (double)(gbb * bx));
      atomicAdd(&virial->sforce_dp[lsh][1], (double)(gbb * by));
      atomicAdd(&virial->sforce_dp[lsh][2], (double)(gbb * bz));
    }
#else
    dfx /= CONVERT_TO_VIR;
    dfy /= CONVERT_TO_VIR;
    dfz /= CONVERT_TO_VIR;
    dgx /= CONVERT_TO_VIR;
    dgy /= CONVERT_TO_VIR;
    dgz /= CONVERT_TO_VIR;
    if (ish != 13) {
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][0],
                llitoulli(dfx));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][1],
                llitoulli(dfy));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][2],
                llitoulli(dfz));
    }
    if (jsh != 13) {
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh][0],
                llitoulli(dgx - dfx));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh][1],
                llitoulli(dgy - dfy));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh][2],
                llitoulli(dgz - dfz));
    }
    if (lsh != 13) {
      dhx /= CONVERT_TO_VIR;
      dhy /= CONVERT_TO_VIR;
      dhz /= CONVERT_TO_VIR;
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh][0],
                llitoulli(dhx));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh][1],
                llitoulli(dhy));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh][2],
                llitoulli(dhz));
    }
#endif
  }
}

//
// dihecoef.x = cpd (integer)
// dihecoef.y = cpc
// dihecoef.z = cpsin
// dihecoef.w = cpcos
//
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_dihe_force_kernel(
    const int ndihelist, const dihelist_t *__restrict__ dihelist,
    const float4 *__restrict__ dihecoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_dihe,
    Virial_t *__restrict__ virial) {
  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy)
    epot = 0.0;

  while (pos < ndihelist) {
    calc_dihe_force_device<AT, CT, true, calc_energy, calc_virial>(
        pos, dihelist, dihecoef, xyzq, stride, boxx, boxy, boxz, (AT *)force,
        epot, virial);
    // printf("epot %f\n", epot);
    pos += blockDim.x * gridDim.x;
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_dihe);
  }
  // if (pos == 8064 - blockDim.x * gridDim.x) printf("pos: %d, epot: %.6f\n",
  // pos, epot); printf("pos: %d, epot: %.6f type : %d\n", pos - blockDim.x *
  // gridDim.x, epot, dihelist[pos - blockDim.x * gridDim.x].itype);
}

//
// dihecoef.x = cpd (integer)
// dihecoef.y = cpc
// dihecoef.z = cpsin
// dihecoef.w = cpcos
//
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_imdihe_force_kernel(
    const int nimdihelist, const dihelist_t *__restrict__ imdihelist,
    const float4 *__restrict__ imdihecoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_imdihe,
    Virial_t *__restrict__ virial) {
  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy)
    epot = 0.0;

  while (pos < nimdihelist) {
    calc_dihe_force_device<AT, CT, false, calc_energy, calc_virial>(
        pos, imdihelist, imdihecoef, xyzq, stride, boxx, boxy, boxz,
        (AT *)force, epot, virial);
    pos += blockDim.x * gridDim.x;
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_imdihe);
  }
}

/*
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_all_forces_kernel() {

  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x*blockDim.x;

  double epot;
  if (calc_energy) {
    epot = 0.0;
  }

  if (pos < d_setup.nbondlist) {
    calc_bond_force_device<AT, CT, calc_energy, calc_virial>
      (pos, d_setup.bondlist, d_setup.bondcoef, d_setup.xyzq,
       d_setup.stride, d_setup.boxx, d_setup.boxy, d_setup.boxz,
       (AT *)d_setup.force, epot);
  } else if (pos < d_setup.nureyblist + d_setup.nbondlist) {
    calc_bond_force_device<AT, CT, calc_energy, calc_virial>
      (pos - d_setup.nbondlist, d_setup.ureyblist, d_setup.ureybcoef,
d_setup.xyzq,
       d_setup.stride, d_setup.boxx, d_setup.boxy, d_setup.boxz,
       (AT *)d_setup.force, epot);
  } else if (pos < d_setup.nanglelist + d_setup.nureyblist + d_setup.nbondlist)
{
    calc_angle_force_device<AT, CT, calc_energy, calc_virial>
      (pos - d_setup.nureyblist - d_setup.nbondlist,
       d_setup.anglelist, d_setup.anglecoef, d_setup.xyzq, d_setup.stride,
       d_setup.boxx, d_setup.boxy, d_setup.boxz, (AT *)d_setup.force, epot);
  } else if (pos < d_setup.ndihelist + d_setup.nanglelist +
             d_setup.nureyblist + d_setup.nbondlist) {
    calc_dihe_force_device<AT, CT, true, calc_energy, calc_virial>
      (pos - d_setup.nanglelist - d_setup.nureyblist - d_setup.nbondlist,
       d_setup.dihelist, d_setup.dihecoef, d_setup.xyzq, d_setup.stride,
       d_setup.boxx, d_setup.boxy, d_setup.boxz, (AT *)d_setup.force, epot);
  } else if (pos < d_setup.nimdihelist + d_setup.ndihelist + d_setup.nanglelist
+
             d_setup.nureyblist + d_setup.nbondlist) {
    calc_dihe_force_device<AT, CT, false, calc_energy, calc_virial>
      (pos - d_setup.ndihelist - d_setup.nanglelist - d_setup.nureyblist -
d_setup.nbondlist,
       d_setup.imdihelist, d_setup.imdihecoef, d_setup.xyzq, d_setup.stride,
       d_setup.boxx, d_setup.boxy, d_setup.boxz, (AT *)d_setup.force, epot);
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, &d_energy_virial.energy_bond);
  }

}
*/

//
// Calculates all forces in a single kernel call
// NOTE: Energy calculation is disabled here because we cannot get individual
// energy terms out
//
template <typename AT, typename CT, bool calc_virial>
__global__ void calc_all_forces_kernel(
    const int nbondlist, const bondlist_t *__restrict__ bondlist,
    const float2 *__restrict__ bondcoef,

    const int nureyblist, const bondlist_t *__restrict__ ureyblist,
    const float2 *__restrict__ ureybcoef,

    const int nanglelist, const anglelist_t *__restrict__ anglelist,
    const float2 *__restrict__ anglecoef,

    const int ndihelist, const dihelist_t *__restrict__ dihelist,
    const float4 *__restrict__ dihecoef,

    const int nimdihelist, const dihelist_t *__restrict__ imdihelist,
    const float4 *__restrict__ imdihecoef,

    const float4 *__restrict__ xyzq, const int stride, const CT boxx,
    const CT boxy, const CT boxz, AT *__restrict__ force,
    Virial_t *__restrict__ virial) {
  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  // Dummy variable
  double epot;

  if (pos < nbondlist) {
    calc_bond_force_device<AT, CT, false, calc_virial>(
        pos, bondlist, bondcoef, xyzq, stride, boxx, boxy, boxz, force, epot,
        virial);
  } else if (pos < nureyblist + nbondlist) {
    calc_bond_force_device<AT, CT, false, calc_virial>(
        pos - nbondlist, ureyblist, ureybcoef, xyzq, stride, boxx, boxy, boxz,
        force, epot, virial);
  } else if (pos < nanglelist + nureyblist + nbondlist) {
    calc_angle_force_device<AT, CT, false, calc_virial>(
        pos - nureyblist - nbondlist, anglelist, anglecoef, xyzq, stride, boxx,
        boxy, boxz, force, epot, virial);
  } else if (pos < ndihelist + nanglelist + nureyblist + nbondlist) {
    calc_dihe_force_device<AT, CT, true, false, calc_virial>(
        pos - nanglelist - nureyblist - nbondlist, dihelist, dihecoef, xyzq,
        stride, boxx, boxy, boxz, force, epot, virial);
  } else if (pos <
             nimdihelist + ndihelist + nanglelist + nureyblist + nbondlist) {
    calc_dihe_force_device<AT, CT, false, false, calc_virial>(
        pos - ndihelist - nanglelist - nureyblist - nbondlist, imdihelist,
        imdihecoef, xyzq, stride, boxx, boxy, boxz, force, epot, virial);
  }
}

//---------------------------------------------------------------------------------------------------------
//
// Setups lists
//

__device__ void setup_bondlist_kernel(const int i,
                                      const int *__restrict__ bond_tbl,
                                      const bond_t *__restrict__ bond,
                                      bondlist_t *__restrict__ bondlist,
                                      const float4 *__restrict__ xyzq,
                                      const float3 half_box,
                                      const int *__restrict__ glo2loc_ind) {
  int j = bond_tbl[i];
  bond_t bondv = bond[j];
  bondlist_t bondlistv;
  bondlistv.i = glo2loc_ind[bondv.i];
  bondlistv.j = glo2loc_ind[bondv.j];
  bondlistv.itype = bondv.itype;
  float4 xyzq_i = xyzq[bondlistv.i];
  float4 xyzq_j = xyzq[bondlistv.j];
  bondlistv.ishift = calc_ishift(xyzq_i, xyzq_j, half_box);
  bondlist[i] = bondlistv;
}

__device__ void setup_anglelist_kernel(const int i,
                                       const int *__restrict__ angle_tbl,
                                       const angle_t *__restrict__ angle,
                                       anglelist_t *__restrict__ anglelist,
                                       const float4 *__restrict__ xyzq,
                                       const float3 half_box,
                                       const int *__restrict__ glo2loc_ind) {
  int j = angle_tbl[i];
  angle_t anglev = angle[j];
  anglelist_t anglelistv;
  anglelistv.i = glo2loc_ind[anglev.i];
  anglelistv.j = glo2loc_ind[anglev.j];
  anglelistv.k = glo2loc_ind[anglev.k];
  anglelistv.itype = anglev.itype;
  float4 xyzq_i = xyzq[anglelistv.i];
  float4 xyzq_j = xyzq[anglelistv.j];
  float4 xyzq_k = xyzq[anglelistv.k];
  anglelistv.ishift1 = calc_ishift(xyzq_i, xyzq_j, half_box);
  anglelistv.ishift2 = calc_ishift(xyzq_k, xyzq_j, half_box);
  anglelist[i] = anglelistv;
}

__device__ void setup_dihelist_kernel(const int i,
                                      const int *__restrict__ dihe_tbl,
                                      const dihe_t *__restrict__ dihe,
                                      dihelist_t *__restrict__ dihelist,
                                      const float4 *__restrict__ xyzq,
                                      const float3 half_box,
                                      const int *__restrict__ glo2loc_ind) {
  int j = dihe_tbl[i];
  dihe_t dihev = dihe[j];
  dihelist_t dihelistv;
  dihelistv.i = glo2loc_ind[dihev.i];
  dihelistv.j = glo2loc_ind[dihev.j];
  dihelistv.k = glo2loc_ind[dihev.k];
  dihelistv.l = glo2loc_ind[dihev.l];
  dihelistv.itype = dihev.itype;
  float4 xyzq_i = xyzq[dihelistv.i];
  float4 xyzq_j = xyzq[dihelistv.j];
  float4 xyzq_k = xyzq[dihelistv.k];
  float4 xyzq_l = xyzq[dihelistv.l];
  dihelistv.ishift1 = calc_ishift(xyzq_i, xyzq_k, half_box);
  dihelistv.ishift2 = calc_ishift(xyzq_j, xyzq_k, half_box);
  dihelistv.ishift3 = calc_ishift(xyzq_l, xyzq_k, half_box);
  dihelist[i] = dihelistv;
}

__device__ void setup_cmaplist_kernel(const int i,
                                      const int *__restrict__ cmap_tbl,
                                      const cmap_t *__restrict__ cmap,
                                      cmaplist_t *__restrict__ cmaplist,
                                      const float4 *__restrict__ xyzq,
                                      const float3 half_box,
                                      const int *__restrict__ glo2loc_ind) {
  int j = cmap_tbl[i];
  cmap_t cmapv = cmap[j];
  cmaplist_t cmaplistv;
  cmaplistv.i1 = glo2loc_ind[cmapv.i1];
  cmaplistv.j1 = glo2loc_ind[cmapv.j1];
  cmaplistv.k1 = glo2loc_ind[cmapv.k1];
  cmaplistv.l1 = glo2loc_ind[cmapv.l1];
  cmaplistv.i2 = glo2loc_ind[cmapv.i2];
  cmaplistv.j2 = glo2loc_ind[cmapv.j2];
  cmaplistv.k2 = glo2loc_ind[cmapv.k2];
  cmaplistv.l2 = glo2loc_ind[cmapv.l2];
  cmaplistv.itype = cmapv.itype;
  float4 xyzq_i1 = xyzq[cmaplistv.i1];
  float4 xyzq_j1 = xyzq[cmaplistv.j1];
  float4 xyzq_k1 = xyzq[cmaplistv.k1];
  float4 xyzq_l1 = xyzq[cmaplistv.l1];
  /* not used in this function
  float4 xyzq_i2 = xyzq[cmaplistv.i2];
  float4 xyzq_j2 = xyzq[cmaplistv.j2];
  float4 xyzq_k2 = xyzq[cmaplistv.k2];
  float4 xyzq_l2 = xyzq[cmaplistv.l2]; */
  cmaplistv.ishift1 = calc_ishift(xyzq_i1, xyzq_k1, half_box);
  cmaplistv.ishift2 = calc_ishift(xyzq_j1, xyzq_k1, half_box);
  cmaplistv.ishift3 = calc_ishift(xyzq_l1, xyzq_k1, half_box);
  cmaplist[i] = cmaplistv;
}

__global__ void setup_list_kernel(
    const int nbond_tbl, const int *__restrict__ bond_tbl,
    const bond_t *__restrict__ bond, bondlist_t *__restrict__ bondlist,
    const int nureyb_tbl, const int *__restrict__ ureyb_tbl,
    const bond_t *__restrict__ ureyb, bondlist_t *__restrict__ ureyblist,
    const int nangle_tbl, const int *__restrict__ angle_tbl,
    const angle_t *__restrict__ angle, anglelist_t *__restrict__ anglelist,
    const int ndihe_tbl, const int *__restrict__ dihe_tbl,
    const dihe_t *__restrict__ dihe, dihelist_t *__restrict__ dihelist,
    const int nimdihe_tbl, const int *__restrict__ imdihe_tbl,
    const dihe_t *__restrict__ imdihe, dihelist_t *__restrict__ imdihelist,
    const int ncmap_tbl, const int *__restrict__ cmap_tbl,
    const cmap_t *__restrict__ cmap, cmaplist_t *__restrict__ cmaplist,
    const float4 *__restrict__ xyzq, const float3 half_box,
    const int *__restrict__ glo2loc_ind) {
  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < nbond_tbl) {
    setup_bondlist_kernel(pos, bond_tbl, bond, bondlist, xyzq, half_box,
                          glo2loc_ind);
  } else if (pos < nbond_tbl + nureyb_tbl) {
    setup_bondlist_kernel(pos - nbond_tbl, ureyb_tbl, ureyb, ureyblist, xyzq,
                          half_box, glo2loc_ind);
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl) {
    setup_anglelist_kernel(pos - nbond_tbl - nureyb_tbl, angle_tbl, angle,
                           anglelist, xyzq, half_box, glo2loc_ind);
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl) {
    setup_dihelist_kernel(pos - nbond_tbl - nureyb_tbl - nangle_tbl, dihe_tbl,
                          dihe, dihelist, xyzq, half_box, glo2loc_ind);
  } else if (pos <
             nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl) {
    setup_dihelist_kernel(pos - nbond_tbl - nureyb_tbl - nangle_tbl - ndihe_tbl,
                          imdihe_tbl, imdihe, imdihelist, xyzq, half_box,
                          glo2loc_ind);
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl +
                       nimdihe_tbl + ncmap_tbl) {
    setup_cmaplist_kernel(
        pos - nbond_tbl - nureyb_tbl - nangle_tbl - ndihe_tbl - nimdihe_tbl,
        cmap_tbl, cmap, cmaplist, xyzq, half_box, glo2loc_ind);
  }
}

//-----------------------------------------------------------------------------------------------------------

// #############################################################################################

//
// Dummy constructor
//
/*template <typename AT, typename CT>
CudaBondedForce<AT, CT>::CudaBondedForce(){
}*/

//
// Class creator
//
template <typename AT, typename CT>
CudaBondedForce<AT, CT>::CudaBondedForce(
    CudaEnergyVirial &energyVirial, const char *nameBond, const char *nameUreyb,
    const char *nameAngle, const char *nameDihe, const char *nameImdihe,
    const char *nameCmap)
    : energyVirial(energyVirial) {
  // Insert energy terms
  if (nameBond != NULL) {
    energyVirial.insert(nameBond);
    strBond = nameBond;
  }

  if (nameUreyb != NULL) {
    energyVirial.insert(nameUreyb);
    strUreyb = nameUreyb;
  }

  if (nameAngle != NULL) {
    energyVirial.insert(nameAngle);
    strAngle = nameAngle;
  }

  if (nameDihe != NULL) {
    energyVirial.insert(nameDihe);
    strDihe = nameDihe;
  }

  if (nameImdihe != NULL) {
    energyVirial.insert(nameImdihe);
    strImdihe = nameImdihe;
  }

  if (nameCmap != NULL) {
    energyVirial.insert(nameCmap);
    strCmap = nameCmap;
  }

  nbondlist = 0;
  nbondcoef = 0;
  bondlist_len = 0;
  bondlist = NULL;
  bondcoef_len = 0;
  bondcoef = NULL;

  nureyblist = 0;
  nureybcoef = 0;
  ureyblist_len = 0;
  ureyblist = NULL;
  ureybcoef_len = 0;
  ureybcoef = NULL;

  nanglelist = 0;
  nanglecoef = 0;
  anglelist_len = 0;
  anglelist = NULL;
  anglecoef_len = 0;
  anglecoef = NULL;

  ndihelist = 0;
  ndihecoef = 0;
  dihelist_len = 0;
  dihelist = NULL;
  dihecoef_len = 0;
  dihecoef = NULL;

  nimdihelist = 0;
  nimdihecoef = 0;
  imdihelist_len = 0;
  imdihelist = NULL;
  imdihecoef_len = 0;
  imdihecoef = NULL;

  ncmaplist = 0;
  ncmapcoef = 0;
  cmaplist_len = 0;
  cmaplist = NULL;
  cmapcoef_len = 0;
  cmapcoef = NULL;

  //  allocate_host<BondedEnergyVirial_t>(&h_energy_virial, 1);
}

// Move Constructor
template <typename AT, typename CT>
CudaBondedForce<AT, CT>::CudaBondedForce(CudaBondedForce &&other)
    : energyVirial(other.energyVirial), strBond(other.strBond),
      strUreyb(other.strUreyb), strAngle(other.strAngle),
      strDihe(other.strDihe), strImdihe(other.strImdihe),
      strCmap(other.strCmap) {

  nbondlist = other.nbondlist;
  nbondcoef = other.nbondcoef;
  bondlist_len = other.bondlist_len;
  bondlist = other.bondlist;
  bondcoef_len = other.bondcoef_len;
  bondcoef = other.bondcoef;

  other.nbondlist = 0;
  other.nbondcoef = 0;
  other.bondlist_len = 0;
  other.bondlist = NULL;
  other.bondcoef_len = 0;
  other.bondcoef = NULL;

  nureyblist = other.nureyblist;
  nureybcoef = other.nureybcoef;
  ureyblist_len = other.ureyblist_len;
  ureyblist = other.ureyblist;
  ureybcoef_len = other.ureybcoef_len;
  ureybcoef = other.ureybcoef;

  other.nureyblist = 0;
  other.nureybcoef = 0;
  other.ureyblist_len = 0;
  other.ureyblist = NULL;
  other.ureybcoef_len = 0;
  other.ureybcoef = NULL;

  nanglelist = other.nanglelist;
  nanglecoef = other.nanglecoef;
  anglelist_len = other.anglelist_len;
  anglelist = other.anglelist;
  anglecoef_len = other.anglecoef_len;
  anglecoef = other.anglecoef;

  other.nanglelist = 0;
  other.nanglecoef = 0;
  other.anglelist_len = 0;
  other.anglelist = NULL;
  other.anglecoef_len = 0;
  other.anglecoef = NULL;

  ndihelist = other.ndihelist;
  ndihecoef = other.ndihecoef;
  dihelist_len = other.dihelist_len;
  dihelist = other.dihelist;
  dihecoef_len = other.dihecoef_len;
  dihecoef = other.dihecoef;

  other.ndihelist = 0;
  other.ndihecoef = 0;
  other.dihelist_len = 0;
  other.dihelist = NULL;
  other.dihecoef_len = 0;
  other.dihecoef = NULL;

  nimdihelist = other.nimdihelist;
  nimdihecoef = other.nimdihecoef;
  imdihelist_len = other.imdihelist_len;
  imdihelist = other.imdihelist;
  imdihecoef_len = other.imdihecoef_len;
  imdihecoef = other.imdihecoef;

  other.nimdihelist = 0;
  other.nimdihecoef = 0;
  other.imdihelist_len = 0;
  other.imdihelist = NULL;
  other.imdihecoef_len = 0;
  other.imdihecoef = NULL;

  ncmaplist = other.ncmaplist;
  ncmapcoef = other.ncmapcoef;
  cmaplist_len = other.cmaplist_len;
  cmaplist = other.cmaplist;
  cmapcoef_len = other.cmapcoef_len;
  cmapcoef = other.cmapcoef;

  other.ncmaplist = 0;
  other.ncmapcoef = 0;
  other.cmaplist_len = 0;
  other.cmaplist = NULL;
  other.cmapcoef_len = 0;
  other.cmapcoef = NULL;

  forceVal = other.forceVal;
  bondedStream = other.bondedStream;
}

//
// Class destructor
//
template <typename AT, typename CT>
CudaBondedForce<AT, CT>::~CudaBondedForce() {
  if (bondlist != NULL)
    deallocate<bondlist_t>(&bondlist);
  if (bondcoef != NULL)
    deallocate<float2>(&bondcoef);

  if (ureyblist != NULL)
    deallocate<bondlist_t>(&ureyblist);
  if (ureybcoef != NULL)
    deallocate<float2>(&ureybcoef);

  if (anglelist != NULL)
    deallocate<anglelist_t>(&anglelist);
  if (anglecoef != NULL)
    deallocate<float2>(&anglecoef);

  if (dihelist != NULL)
    deallocate<dihelist_t>(&dihelist);
  if (dihecoef != NULL)
    deallocate<float4>(&dihecoef);

  if (imdihelist != NULL)
    deallocate<dihelist_t>(&imdihelist);
  if (imdihecoef != NULL)
    deallocate<float4>(&imdihecoef);

  if (cmaplist != NULL)
    deallocate<cmaplist_t>(&cmaplist);
  if (cmapcoef != NULL)
    deallocate<float2>(&cmapcoef);

  //  deallocate_host<BondedEnergyVirial_t>(&h_energy_virial);
}

//
// Setup the coefficients using vectors
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setup_coef(
    const std::vector<int> &size, const std::vector<std::vector<float>> &val) {
  assert(size.size() == 6);
  assert(val.size() == std::accumulate(size.begin(), size.end(), 0));

  nbondcoef = size[0];
  nureybcoef = size[1];
  nanglecoef = size[2];
  ndihecoef = size[3];
  nimdihecoef = size[4];
  ncmapcoef = size[5];
  size_t pos = 0;
  if (nbondcoef > 0) {
    float2 *h_bondcoef;
    h_bondcoef = (float2 *)malloc(nbondcoef * sizeof(float2));
    for (int i = 0; i < nbondcoef; ++i) {
      float2 elem;
      elem.x = val[i][0];
      elem.y = val[i][1];
      h_bondcoef[i] = elem;
    }
    reallocate<float2>(&bondcoef, &bondcoef_len, nbondcoef, 1.2f);
    copy_HtoD<float2>(h_bondcoef, bondcoef, nbondcoef);
    pos += nbondcoef;
    free(h_bondcoef);
  }

  if (nureybcoef > 0) {
    float2 *h_ureybcoef;
    h_ureybcoef = new float2[nureybcoef];
    for (int i = 0; i < nureybcoef; ++i) {
      float2 elem;
      elem.x = val[pos + i][0];
      elem.y = val[pos + i][1];
      h_ureybcoef[i] = elem;
    }
    reallocate<float2>(&ureybcoef, &ureybcoef_len, nureybcoef, 1.2f);
    copy_HtoD<float2>(h_ureybcoef, ureybcoef, nureybcoef);
    pos += nureybcoef;
    delete h_ureybcoef;
  }

  if (nanglecoef > 0) {
    float2 *h_anglecoef = new float2[nanglecoef];
    for (int i = 0; i < nanglecoef; ++i) {
      float2 elem;
      elem.x = val[pos + i][0];
      elem.y = val[pos + i][1];
      h_anglecoef[i] = elem;
    }
    reallocate<float2>(&anglecoef, &anglecoef_len, nanglecoef, 1.2f);
    copy_HtoD<float2>(h_anglecoef, anglecoef, nanglecoef);
    pos += nanglecoef;
    delete h_anglecoef;
  }

  if (ndihecoef > 0) {
    float4 *h_dihecoef = new float4[ndihecoef];
    for (int i = 0; i < ndihecoef; ++i) {
      float4 elem;
      elem.x = val[pos + i][0];
      elem.y = val[pos + i][1];
      elem.z = val[pos + i][2];
      elem.w = val[pos + i][3];
      h_dihecoef[i] = elem;
    }
    reallocate<float4>(&dihecoef, &dihecoef_len, ndihecoef, 1.2f);
    copy_HtoD<float4>(h_dihecoef, dihecoef, ndihecoef);
    pos += ndihecoef;
    delete h_dihecoef;
  }

  if (nimdihecoef > 0) {
    float4 *h_imdihecoef = new float4[nimdihecoef];
    for (int i = 0; i < nimdihecoef; ++i) {
      float4 elem;
      elem.x = val[pos + i][0];
      elem.y = val[pos + i][1];
      elem.z = val[pos + i][2];
      elem.w = val[pos + i][3];
      h_imdihecoef[i] = elem;
    }
    reallocate<float4>(&imdihecoef, &imdihecoef_len, nimdihecoef, 1.2f);
    copy_HtoD<float4>(h_imdihecoef, imdihecoef, nimdihecoef);
    pos += nimdihecoef;
    delete h_imdihecoef;
  }

  if (ncmapcoef > 0) {
    float2 *h_cmapcoef = new float2[ncmapcoef];
    for (int i = 0; i < ncmapcoef; ++i) {
      float2 elem;
      elem.x = val[pos + i][0];
      elem.y = val[pos + i][1];
      h_cmapcoef[i] = elem;
    }
    reallocate<float2>(&cmapcoef, &cmapcoef_len, ncmapcoef, 1.2f);
    copy_HtoD<float2>(h_cmapcoef, cmapcoef, ncmapcoef);
    pos += ncmapcoef;
    delete h_cmapcoef;
  }
  assert(pos == val.size());
}

//
// Setup coefficients (copies them from CPU to GPU)
// NOTE: This only has to be once in the beginning of the simulation
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setup_coef(
    const int nbondcoef, const float2 *h_bondcoef, const int nureybcoef,
    const float2 *h_ureybcoef, const int nanglecoef, const float2 *h_anglecoef,
    const int ndihecoef, const float4 *h_dihecoef, const int nimdihecoef,
    const float4 *h_imdihecoef, const int ncmapcoef, const float2 *h_cmapcoef) {
  assert((nureybcoef == 0) || (nureybcoef > 0 && nureybcoef == nanglecoef));

  this->nbondcoef = nbondcoef;
  if (nbondcoef > 0) {
    reallocate<float2>(&bondcoef, &bondcoef_len, nbondcoef, 1.2f);
    copy_HtoD<float2>(h_bondcoef, bondcoef, nbondcoef);
  }

  this->nureybcoef = nureybcoef;
  if (nureybcoef > 0) {
    reallocate<float2>(&ureybcoef, &ureybcoef_len, nureybcoef, 1.2f);
    copy_HtoD<float2>(h_ureybcoef, ureybcoef, nureybcoef);
  }

  this->nanglecoef = nanglecoef;
  if (nanglecoef > 0) {
    reallocate<float2>(&anglecoef, &anglecoef_len, nanglecoef, 1.2f);
    copy_HtoD<float2>(h_anglecoef, anglecoef, nanglecoef);
  }

  this->ndihecoef = ndihecoef;
  if (ndihecoef > 0) {
    reallocate<float4>(&dihecoef, &dihecoef_len, ndihecoef, 1.2f);
    copy_HtoD<float4>(h_dihecoef, dihecoef, ndihecoef);
  }

  this->nimdihecoef = nimdihecoef;
  if (nimdihecoef > 0) {
    reallocate<float4>(&imdihecoef, &imdihecoef_len, nimdihecoef, 1.2f);
    copy_HtoD<float4>(h_imdihecoef, imdihecoef, nimdihecoef);
  }

  this->ncmapcoef = ncmapcoef;
  if (ncmapcoef > 0) {
    reallocate<float2>(&cmapcoef, &cmapcoef_len, ncmapcoef, 1.2f);
    copy_HtoD<float2>(h_cmapcoef, cmapcoef, ncmapcoef);
  }
}

//
// Setup bondlists from vectors
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setup_list(
    const std::vector<int> &size, const std::vector<std::vector<int>> &val,
    cudaStream_t stream) {
  nbondlist = size[0];
  nureyblist = size[1];
  nanglelist = size[2];
  ndihelist = size[3];
  nimdihelist = size[4];
  ncmaplist = size[5];
  // assert((nureyblist == 0) || (nureyblist > 0 && nureyblist ==
  // nanglelist));
  size_t pos = 0;
  if (nbondlist > 0) {
    bondlist_t *h_bondlist = new bondlist_t[nbondlist];
    for (int i = 0; i < nbondlist; ++i) {
      h_bondlist[i].i = val[i][0];
      h_bondlist[i].j = val[i][1];
      h_bondlist[i].itype = val[i][2];
      h_bondlist[i].ishift = val[i][3];
    }
    reallocate<bondlist_t>(&bondlist, &bondlist_len, nbondlist, 1.2f);
    copy_HtoD<bondlist_t>(h_bondlist, bondlist, nbondlist, stream);
    pos += nbondlist;
    delete h_bondlist;
  }

  if (nureyblist > 0) {
    bondlist_t *h_ureyblist = new bondlist_t[nureyblist];
    for (int i = 0; i < nureyblist; ++i) {
      h_ureyblist[i].i = val[pos + i][0];
      h_ureyblist[i].j = val[pos + i][1];
      h_ureyblist[i].itype = val[pos + i][2];
      h_ureyblist[i].ishift = val[pos + i][3];
    }
    reallocate<bondlist_t>(&ureyblist, &ureyblist_len, nureyblist, 1.2f);
    copy_HtoD<bondlist_t>(h_ureyblist, ureyblist, nureyblist, stream);
    pos += nureyblist;
    delete h_ureyblist;
  }

  if (nanglelist > 0) {
    anglelist_t *h_anglelist = new anglelist_t[nanglelist];
    for (int i = 0; i < nanglelist; ++i) {
      h_anglelist[i].i = val[pos + i][0];
      h_anglelist[i].j = val[pos + i][1];
      h_anglelist[i].k = val[pos + i][2];
      h_anglelist[i].itype = val[pos + i][3];
      h_anglelist[i].ishift1 = val[pos + i][4];
      h_anglelist[i].ishift2 = val[pos + i][5];
    }
    reallocate<anglelist_t>(&anglelist, &anglelist_len, nanglelist, 1.2f);
    copy_HtoD<anglelist_t>(h_anglelist, anglelist, nanglelist, stream);
    pos += nanglelist;
    delete h_anglelist;
  }

  if (ndihelist > 0) {
    dihelist_t *h_dihelist = new dihelist_t[ndihelist];
    for (int i = 0; i < ndihelist; ++i) {
      h_dihelist[i].i = val[pos + i][0];
      h_dihelist[i].j = val[pos + i][1];
      h_dihelist[i].k = val[pos + i][2];
      h_dihelist[i].l = val[pos + i][3];
      h_dihelist[i].itype = val[pos + i][4];
      h_dihelist[i].ishift1 = val[pos + i][5];
      h_dihelist[i].ishift2 = val[pos + i][6];
      h_dihelist[i].ishift3 = val[pos + i][7];
    }
    reallocate<dihelist_t>(&dihelist, &dihelist_len, ndihelist, 1.2f);
    copy_HtoD<dihelist_t>(h_dihelist, dihelist, ndihelist, stream);
    pos += ndihelist;
    delete h_dihelist;
  }

  if (nimdihelist > 0) {
    dihelist_t *h_imdihelist = new dihelist_t[nimdihelist];
    for (int i = 0; i < nimdihelist; ++i) {
      h_imdihelist[i].i = val[pos + i][0];
      h_imdihelist[i].j = val[pos + i][1];
      h_imdihelist[i].k = val[pos + i][2];
      h_imdihelist[i].l = val[pos + i][3];
      h_imdihelist[i].itype = val[pos + i][4];
      h_imdihelist[i].ishift1 = val[pos + i][5];
      h_imdihelist[i].ishift2 = val[pos + i][6];
      h_imdihelist[i].ishift3 = val[pos + i][7];
    }
    reallocate<dihelist_t>(&imdihelist, &imdihelist_len, nimdihelist, 1.2f);
    copy_HtoD<dihelist_t>(h_imdihelist, imdihelist, nimdihelist, stream);
    pos += nimdihelist;
    delete h_imdihelist;
  }

  if (ncmaplist > 0) {
    cmaplist_t *h_cmaplist = new cmaplist_t[ncmaplist];
    for (int i = 0; i < ncmaplist; ++i) {
      h_cmaplist[i].i1 = val[pos + i][0];
      h_cmaplist[i].j1 = val[pos + i][1];
      h_cmaplist[i].k1 = val[pos + i][2];
      h_cmaplist[i].l1 = val[pos + i][3];
      h_cmaplist[i].i2 = val[pos + i][4];
      h_cmaplist[i].j2 = val[pos + i][5];
      h_cmaplist[i].k2 = val[pos + i][6];
      h_cmaplist[i].l2 = val[pos + i][7];
      h_cmaplist[i].itype = val[pos + i][8];
      h_cmaplist[i].ishift1 = val[pos + i][9];
      h_cmaplist[i].ishift2 = val[pos + i][10];
      h_cmaplist[i].ishift3 = val[pos + i][11];
    }
    reallocate<cmaplist_t>(&cmaplist, &cmaplist_len, ncmaplist, 1.2f);
    copy_HtoD<cmaplist_t>(h_cmaplist, cmaplist, ncmaplist, stream);
    pos += ncmaplist;
    delete h_cmaplist;
  }
}

//
// Setup bondlists (copies them from CPU to GPU)
// NOTE: This has to be done after neighborlist update
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setup_list(
    const int nbondlist, const bondlist_t *h_bondlist, const int nureyblist,
    const bondlist_t *h_ureyblist, const int nanglelist,
    const anglelist_t *h_anglelist, const int ndihelist,
    const dihelist_t *h_dihelist, const int nimdihelist,
    const dihelist_t *h_imdihelist, const int ncmaplist,
    const cmaplist_t *h_cmaplist, cudaStream_t stream) {
  assert((nureyblist == 0) || (nureyblist > 0 && nureyblist == nanglelist));

  this->nbondlist = nbondlist;
  if (nbondlist > 0) {
    reallocate<bondlist_t>(&bondlist, &bondlist_len, nbondlist, 1.2f);
    copy_HtoD<bondlist_t>(h_bondlist, bondlist, nbondlist, stream);
  }

  this->nureyblist = nureyblist;
  if (nureyblist > 0) {
    reallocate<bondlist_t>(&ureyblist, &ureyblist_len, nureyblist, 1.2f);
    copy_HtoD<bondlist_t>(h_ureyblist, ureyblist, nureyblist, stream);
  }

  this->nanglelist = nanglelist;
  if (nanglelist > 0) {
    reallocate<anglelist_t>(&anglelist, &anglelist_len, nanglelist, 1.2f);
    copy_HtoD<anglelist_t>(h_anglelist, anglelist, nanglelist, stream);
  }

  this->ndihelist = ndihelist;
  if (ndihelist > 0) {
    reallocate<dihelist_t>(&dihelist, &dihelist_len, ndihelist, 1.2f);
    copy_HtoD<dihelist_t>(h_dihelist, dihelist, ndihelist, stream);
  }

  this->nimdihelist = nimdihelist;
  if (nimdihelist > 0) {
    reallocate<dihelist_t>(&imdihelist, &imdihelist_len, nimdihelist, 1.2f);
    copy_HtoD<dihelist_t>(h_imdihelist, imdihelist, nimdihelist, stream);
  }

  this->ncmaplist = ncmaplist;
  if (ncmaplist > 0) {
    reallocate<cmaplist_t>(&cmaplist, &cmaplist_len, ncmaplist, 1.2f);
    copy_HtoD<cmaplist_t>(h_cmaplist, cmaplist, ncmaplist, stream);
  }
}

//
// Setup lists from device memory using global bond data:
// bond[]                  = global bond data
// bond_tbl[0:nbond_tbl-1] = the index of bond in bond[]
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setup_list(
    const float4 *xyzq, const CT boxx, const CT boxy, const CT boxz,
    const int *glo2loc_ind, const int nbond_tbl, const int *bond_tbl,
    const bond_t *bond, const int nureyb_tbl, const int *ureyb_tbl,
    const bond_t *ureyb, const int nangle_tbl, const int *angle_tbl,
    const angle_t *angle, const int ndihe_tbl, const int *dihe_tbl,
    const dihe_t *dihe, const int nimdihe_tbl, const int *imdihe_tbl,
    const dihe_t *imdihe, const int ncmap_tbl, const int *cmap_tbl,
    const cmap_t *cmap, cudaStream_t stream) {
  this->nbondlist = nbond_tbl;
  if (nbondlist > 0)
    reallocate<bondlist_t>(&bondlist, &bondlist_len, nbondlist, 1.2f);

  this->nureyblist = nureyb_tbl;
  if (nureyblist > 0)
    reallocate<bondlist_t>(&ureyblist, &ureyblist_len, nureyblist, 1.2f);

  this->nanglelist = nangle_tbl;
  if (nanglelist > 0)
    reallocate<anglelist_t>(&anglelist, &anglelist_len, nanglelist, 1.2f);

  this->ndihelist = ndihe_tbl;
  if (ndihelist > 0)
    reallocate<dihelist_t>(&dihelist, &dihelist_len, ndihelist, 1.2f);

  this->nimdihelist = nimdihe_tbl;
  if (nimdihelist > 0)
    reallocate<dihelist_t>(&imdihelist, &imdihelist_len, nimdihelist, 1.2f);

  this->ncmaplist = ncmap_tbl;
  if (ncmaplist > 0)
    reallocate<cmaplist_t>(&cmaplist, &cmaplist_len, ncmaplist, 1.2f);

  float3 half_box;
  half_box.x = boxx * 0.5f;
  half_box.y = boxy * 0.5f;
  half_box.z = boxz * 0.5f;

  int nthread = 512;
  int nblock = (nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl +
                ncmap_tbl - 1) /
                   nthread +
               1;
  setup_list_kernel<<<nblock, nthread, 0, stream>>>(
      nbond_tbl, bond_tbl, bond, bondlist, nureyb_tbl, ureyb_tbl, ureyb,
      ureyblist, nangle_tbl, angle_tbl, angle, anglelist, ndihe_tbl, dihe_tbl,
      dihe, dihelist, nimdihe_tbl, imdihe_tbl, imdihe, imdihelist, ncmap_tbl,
      cmap_tbl, cmap, cmaplist, xyzq, half_box, glo2loc_ind);
  cudaCheck(cudaGetLastError());
}

//
// Print info
//
template <typename AT, typename CT> void CudaBondedForce<AT, CT>::print() {
  int maxnum = nbondlist;
  maxnum = std::max(maxnum, nureyblist);
  maxnum = std::max(maxnum, nanglelist);
  maxnum = std::max(maxnum, ndihelist);
  maxnum = std::max(maxnum, nimdihelist);
  maxnum = std::max(maxnum, ncmaplist);
  maxnum = std::max(maxnum, nbondcoef);
  maxnum = std::max(maxnum, nureybcoef);
  maxnum = std::max(maxnum, nanglecoef);
  maxnum = std::max(maxnum, ndihecoef);
  maxnum = std::max(maxnum, nimdihecoef);
  maxnum = std::max(maxnum, ncmapcoef);
  int maxw = (int)log10(maxnum + 1) + 2;
  std::cout << "BOND:   " << std::setw(maxw) << nbondlist << " "
            << std::setw(maxw) << nbondcoef << std::endl;
  std::cout << "UREYB:  " << std::setw(maxw) << nureyblist << " "
            << std::setw(maxw) << nureybcoef << std::endl;
  std::cout << "ANGLE:  " << std::setw(maxw) << nanglelist << " "
            << std::setw(maxw) << nanglecoef << std::endl;
  std::cout << "DIHE:   " << std::setw(maxw) << ndihelist << " "
            << std::setw(maxw) << ndihecoef << std::endl;
  std::cout << "IMDIHE: " << std::setw(maxw) << nimdihelist << " "
            << std::setw(maxw) << nimdihecoef << std::endl;
}

template <typename AT, typename CT> void CudaBondedForce<AT, CT>::clear(void) {
  energyVirial.clear(*bondedStream);
  forceVal->clear(*bondedStream);
  return;
}

//
// Calculates forces
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::calc_force(
    const float4 *xyzq, const CT boxx, const CT boxy, const CT boxz,
    const bool calc_energy, const bool calc_virial, const int stride, AT *force,
    const bool calc_bond, const bool calc_ureyb, const bool calc_angle,
    const bool calc_dihe, const bool calc_imdihe, const bool calc_cmap,
    cudaStream_t stream) {
  if (ncmaplist > 0) {
    // std::cerr << "CudaBondedForce<AT, CT>::calc_force, cmap not implemented"
    //           << std::endl;
    throw std::invalid_argument(
        "CudaBondedForce<AT, CT>::calc_force, cmap not implemented\n");
    exit(1);
  }
  if (calc_energy) {
    if (calc_bond) {
      int nthread = 512;
      int nblock = (nbondlist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);
      if (calc_virial) {
        calc_bond_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                nbondlist, bondlist, bondcoef, xyzq, stride, boxx, boxy, boxz,
                force, energyVirial.getEnergyPointer(strBond),
                energyVirial.getVirialPointer());
      } else {
        calc_bond_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                nbondlist, bondlist, bondcoef, xyzq, stride, boxx, boxy, boxz,
                force, energyVirial.getEnergyPointer(strBond), NULL);
      }
      cudaCheck(cudaGetLastError());
    }

    if (calc_ureyb) {
      int nthread = 512;
      int nblock = (nureyblist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);
      if (calc_virial) {
        calc_ureyb_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                nureyblist, ureyblist, ureybcoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strUreyb),
                energyVirial.getVirialPointer());
      } else {
        calc_ureyb_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                nureyblist, ureyblist, ureybcoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strUreyb), NULL);
      }
      cudaCheck(cudaGetLastError());
    }
    if (calc_angle) {
      // int nthread = 512;
      int nthread = 128;
      int nblock = (nanglelist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);
      if (calc_virial) {
        calc_angle_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                nanglelist, anglelist, anglecoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strAngle),
                energyVirial.getVirialPointer());
      } else {
        calc_angle_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                nanglelist, anglelist, anglecoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strAngle), NULL);
      }
      cudaCheck(cudaGetLastError());
    }

    if (calc_dihe) {
      // int nthread = 512;
      int nthread = 128;
      int nblock = (ndihelist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);
      if (calc_virial) {
        calc_dihe_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                ndihelist, dihelist, dihecoef, xyzq, stride, boxx, boxy, boxz,
                force, energyVirial.getEnergyPointer(strDihe),
                energyVirial.getVirialPointer());
      } else {
        calc_dihe_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                ndihelist, dihelist, dihecoef, xyzq, stride, boxx, boxy, boxz,
                force, energyVirial.getEnergyPointer(strDihe), NULL);
      }
      cudaCheck(cudaGetLastError());
    }

    if (calc_imdihe) {
      // int nthread = 512;
      int nthread = 128;
      int nblock = (nimdihelist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);
      if (calc_virial) {
        calc_imdihe_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                nimdihelist, imdihelist, imdihecoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strImdihe),
                energyVirial.getVirialPointer());
      } else {
        calc_imdihe_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                nimdihelist, imdihelist, imdihecoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strImdihe), NULL);
      }
      cudaCheck(cudaGetLastError());
    }
  } else {
    int nbondlist_loc = (calc_bond) ? nbondlist : 0;
    int nureyblist_loc = (calc_ureyb) ? nureyblist : 0;
    int nanglelist_loc = (calc_angle) ? nanglelist : 0;
    int ndihelist_loc = (calc_dihe) ? ndihelist : 0;
    int nimdihelist_loc = (calc_imdihe) ? nimdihelist : 0;

    // int nthread = 512;
    int nthread = 128;
    int nblock = (nbondlist_loc + nureyblist_loc + nanglelist_loc +
                  ndihelist_loc + nimdihelist_loc - 1) /
                     nthread +
                 1;
    int shmem_size = 0;

    if (calc_virial) {
      calc_all_forces_kernel<AT, CT, true>
          <<<nblock, nthread, shmem_size, stream>>>(
              nbondlist_loc, bondlist, bondcoef, nureyblist_loc, ureyblist,
              ureybcoef, nanglelist_loc, anglelist, anglecoef, ndihelist_loc,
              dihelist, dihecoef, nimdihelist_loc, imdihelist, imdihecoef, xyzq,
              stride, boxx, boxy, boxz, force, energyVirial.getVirialPointer());
    } else {
      calc_all_forces_kernel<AT, CT, false>
          <<<nblock, nthread, shmem_size, stream>>>(
              nbondlist_loc, bondlist, bondcoef, nureyblist_loc, ureyblist,
              ureybcoef, nanglelist_loc, anglelist, anglecoef, ndihelist_loc,
              dihelist, dihecoef, nimdihelist_loc, imdihelist, imdihecoef, xyzq,
              stride, boxx, boxy, boxz, force, NULL);
    }
    cudaCheck(cudaGetLastError());
  }
}

template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setForce(
    std::shared_ptr<Force<long long int>> &forceValIn) {
  forceVal = forceValIn;
}

template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setStream(
    std::shared_ptr<cudaStream_t> streamIn) {
  bondedStream = streamIn;
}

template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::calc_force(const float4 *xyzq, bool calcEnergy,
                                         bool calcVirial) {

  calc_force(xyzq, boxDimensions[0], boxDimensions[1], boxDimensions[2],
             calcEnergy, calcVirial, forceVal->stride(), forceVal->xyz(), true,
             true, true, true, true, false, *bondedStream);

  /*if (calcVirial) {
    energyVirial.calcVirial(10000, xyzq, boxDimensions[0], boxDimensions[1],
                            boxDimensions[2], forceVal->stride(),
                            forceVal->xyz(), *bondedStream);
  }
  */
}

template <typename AT, typename CT>
std::shared_ptr<Force<long long int>> CudaBondedForce<AT, CT>::getForce(void) {
  return forceVal;
}

//
// Explicit instances of CudaBondedForce
//
template class CudaBondedForce<long long int, float>;
template class CudaBondedForce<long long int, double>;
#endif // NOCUDAC
