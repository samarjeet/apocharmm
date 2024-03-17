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
#include "Bonded_struct.h"
#include "CudaBlock.h"
#include "CudaDirectForceTypes.h"
#include "CudaEnergyVirial.h"
#include "CudaNeighborListBuild.h"
#include "CudaP21NeighborListBuild.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cuda.h>

// #define USE_TEXTURES true
#define USE_TEXTURES false
// #undef USE_TEXTURE_OBJECTS

// Settings for direct computation in device memory
__constant__ DirectSettings_t d_setup;

// Energy and virial in device memory
// static __device__ DirectEnergyVirial_t d_energy_virial;

#ifndef USE_TEXTURE_OBJECTS
// VdW parameter texture reference
texture<float2, 1, cudaReadModeElementType> vdwparam_texref;
bool vdwparam_texref_bound = false;
texture<float2, 1, cudaReadModeElementType> vdwparam14_texref;
bool vdwparam14_texref_bound = false;
texture<float, 1, cudaReadModeElementType> blockParamTexRef;
bool blockParamTexRefBound = false;
texture<float2, 1, cudaReadModeElementType> *get_vdwparam_texref() {
  return &vdwparam_texref;
}
texture<float2, 1, cudaReadModeElementType> *get_vdwparam14_texref() {
  return &vdwparam14_texref;
}
texture<float, 1, cudaReadModeElementType> *getBlockParamTexRef() {
  return &blockParamTexRef;
}
bool get_vdwparam_texref_bound() { return vdwparam_texref_bound; }
bool get_vdwparam14_texref_bound() { return vdwparam14_texref_bound; }
bool getBlockParamTexRefBound() { return blockParamTexRefBound; }
void set_vdwparam_texref_bound(const bool val) { vdwparam_texref_bound = val; }
void set_vdwparam14_texref_bound(const bool val) {
  vdwparam14_texref_bound = val;
}
void setBlockParamTexRefBound(const bool val) { blockParamTexRefBound = val; }
#endif

static __constant__ const float ccelec = 332.0716f;
const int tilesize = 32;

/*
//
// Nonbonded virial
//
__global__ void calc_virial_kernel(const int ncoord, const float4* __restrict__
xyzq,
                                   const int stride, DirectEnergyVirial_t*
__restrict__ energy_virial,
                                   const double* __restrict__ force) {
  // Shared memory:
  // Required memory
  // blockDim.x*9*sizeof(double) for __CUDA_ARCH__ < 300
  // blockDim.x*9*sizeof(double)/warpsize for __CUDA_ARCH__ >= 300
  extern __shared__ volatile double sh_vir[];

  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  int ish = (i - ncoord)*3 + 1;

  double vir[9];
  if (i < ncoord) {
    float4 xyzqi = xyzq[i];
    double x = (double)xyzqi.x;
    double y = (double)xyzqi.y;
    double z = (double)xyzqi.z;
    double fx = (double)force[i];
    double fy = (double)force[i+stride];
    double fz = (double)force[i+stride*2];
    vir[0] = x*fx;
    vir[1] = x*fy;
    vir[2] = x*fz;
    vir[3] = y*fx;
    vir[4] = y*fy;
    vir[5] = y*fz;
    vir[6] = z*fx;
    vir[7] = z*fy;
    vir[8] = z*fz;
  } else if (ish >= 1 && ish <= 26*3+1) {
    double sforcex = energy_virial->sforce[ish-1] +
((double)energy_virial->sforce_fp[ish-1])*INV_FORCE_SCALE_VIR_CPU;
    double sforcey = energy_virial->sforce[ish]   +
((double)energy_virial->sforce_fp[ish])*INV_FORCE_SCALE_VIR_CPU;
    double sforcez = energy_virial->sforce[ish+1] +
((double)energy_virial->sforce_fp[ish+1])*INV_FORCE_SCALE_VIR_CPU;
    double shx, shy, shz;
    calc_box_shift<double>(ish, (double)d_setup.boxx, (double)d_setup.boxy,
(double)d_setup.boxz, shx, shy, shz);
    vir[0] = shx*sforcex;
    vir[1] = shx*sforcey;
    vir[2] = shx*sforcez;
    vir[3] = shy*sforcex;
    vir[4] = shy*sforcey;
    vir[5] = shy*sforcez;
    vir[6] = shz*sforcex;
    vir[7] = shz*sforcey;
    vir[8] = shz*sforcez;
  } else {
#pragma unroll
    for (int k=0;k < 9;k++)
      vir[k] = 0.0;
  }

  // Reduce
  //#if __CUDA_ARCH__ < 300
  // 0-2
#pragma unroll
  for (int k=0;k < 3;k++)
    sh_vir[threadIdx.x + k*blockDim.x] = vir[k];
  __syncthreads();
  for (int i=1;i < blockDim.x;i *= 2) {
    int pos = threadIdx.x + i;
    double vir_val[3];
#pragma unroll
    for (int k=0;k < 3;k++)
      vir_val[k] = (pos < blockDim.x) ? sh_vir[pos + k*blockDim.x] : 0.0;
    __syncthreads();
#pragma unroll
    for (int k=0;k < 3;k++)
      sh_vir[threadIdx.x + k*blockDim.x] += vir_val[k];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k=0;k < 3;k++)
      atomicAdd(&energy_virial->vir[k], -sh_vir[k*blockDim.x]);
  }

  // 3-5
#pragma unroll
  for (int k=0;k < 3;k++)
    sh_vir[threadIdx.x + k*blockDim.x] = vir[k+3];
  __syncthreads();
  for (int i=1;i < blockDim.x;i *= 2) {
    int pos = threadIdx.x + i;
    double vir_val[3];
#pragma unroll
    for (int k=0;k < 3;k++)
      vir_val[k] = (pos < blockDim.x) ? sh_vir[pos + k*blockDim.x] : 0.0;
    __syncthreads();
#pragma unroll
    for (int k=0;k < 3;k++)
      sh_vir[threadIdx.x + k*blockDim.x] += vir_val[k];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k=0;k < 3;k++)
      atomicAdd(&energy_virial->vir[k+3], -sh_vir[k*blockDim.x]);
  }

  // 6-8
#pragma unroll
  for (int k=0;k < 3;k++)
    sh_vir[threadIdx.x + k*blockDim.x] = vir[k+6];
  __syncthreads();
  for (int i=1;i < blockDim.x;i *= 2) {
    int pos = threadIdx.x + i;
    double vir_val[3];
#pragma unroll
    for (int k=0;k < 3;k++)
      vir_val[k] = (pos < blockDim.x) ? sh_vir[pos + k*blockDim.x] : 0.0;
    __syncthreads();
#pragma unroll
    for (int k=0;k < 3;k++)
      sh_vir[threadIdx.x + k*blockDim.x] += vir_val[k];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k=0;k < 3;k++)
      atomicAdd(&energy_virial->vir[k+6], -sh_vir[k*blockDim.x]);
  }

}
*/

//
// Calculates VdW pair force & energy
// NOTE: force (fij_vdw) is r*dU/dr
//

/**
 * @brief Calculates VdW pair force & energy
 *
 * NOTE: force (fij_vdw) is r*dU/dr
 */
template <int vdw_model, bool calc_energy>
__forceinline__ __device__ float
pair_vdw_force(const float r2, const float r, const float rinv, const float c6,
               const float c12, float &pot_vdw) {
  float rinv2 = rinv * rinv;
  float fij_vdw;

  if (vdw_model == VDW_VSH) {
    float r6 = r2 * r2 * r2;
    float rinv6 = rinv2 * rinv2 * rinv2;
    float rinv12 = rinv6 * rinv6;
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      pot_vdw = c12 * one_twelve *
                    (rinv12 + 2.0f * r6 * d_setup.roffinv18 -
                     3.0f * d_setup.roffinv12) -
                c6 * one_six *
                    (rinv6 + r6 * d_setup.roffinv12 - 2.0f * d_setup.roffinv6);
    }

    fij_vdw = c6 * (rinv6 - r6 * d_setup.roffinv12) -
              c12 * (rinv12 + r6 * d_setup.roffinv18);
  } else if (vdw_model == VDW_VSW) {
    float roff2_r2_sq = d_setup.roff2 - r2;
    roff2_r2_sq *= roff2_r2_sq;
    float sw = (r2 <= d_setup.ron2)
                   ? 1.0f
                   : roff2_r2_sq *
                         (d_setup.roff2 + 2.0f * r2 - 3.0f * d_setup.ron2) *
                         d_setup.inv_roff2_ron2_3;
    // dsw_6 = dsw/6.0
    float dsw_6 = (r2 <= d_setup.ron2)
                      ? 0.0f
                      : (d_setup.roff2 - r2) * (d_setup.ron2 - r2) *
                            d_setup.inv_roff2_ron2_3;
    float rinv4 = rinv2 * rinv2;
    float rinv6 = rinv4 * rinv2;
    fij_vdw = rinv4 * (c12 * rinv6 * (dsw_6 - sw * rinv2) -
                       c6 * (2.0f * dsw_6 - sw * rinv2));
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      pot_vdw = sw * rinv6 * (one_twelve * c12 * rinv6 - one_six * c6);
    }
  } else if (vdw_model == VDW_CUT) {
    float rinv6 = rinv2 * rinv2 * rinv2;
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      float rinv12 = rinv6 * rinv6;
      pot_vdw = c12 * one_twelve * rinv12 - c6 * one_six * rinv6;
      fij_vdw = c6 * rinv6 - c12 * rinv12;
    } else {
      fij_vdw = c6 * rinv6 - c12 * rinv6 * rinv6;
    }
  } else if (vdw_model == VDW_VFSW) {
    float rinv3 = rinv * rinv2;
    float rinv6 = rinv3 * rinv3;
    float A6 = (r2 > d_setup.ron2) ? d_setup.k6 : 1.0f;
    float B6 = (r2 > d_setup.ron2) ? d_setup.roffinv3 : 0.0f;
    float A12 = (r2 > d_setup.ron2) ? d_setup.k12 : 1.0f;
    float B12 = (r2 > d_setup.ron2) ? d_setup.roffinv6 : 0.0f;
    fij_vdw =
        c6 * A6 * (rinv3 - B6) * rinv3 - c12 * A12 * (rinv6 - B12) * rinv6;
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      float C6 = (r2 > d_setup.ron2) ? 0.0f : d_setup.dv6;
      float C12 = (r2 > d_setup.ron2) ? 0.0f : d_setup.dv12;

      float rinv3_B6_sq = rinv3 - B6;
      rinv3_B6_sq *= rinv3_B6_sq;

      float rinv6_B12_sq = rinv6 - B12;
      rinv6_B12_sq *= rinv6_B12_sq;

      pot_vdw = one_twelve * c12 * (A12 * rinv6_B12_sq + C12) -
                one_six * c6 * (A6 * rinv3_B6_sq + C6);
    }
  } else if (vdw_model == VDW_VGSH) {
    float rinv3 = rinv * rinv2;
    float rinv6 = rinv3 * rinv3;
    float rinv12 = rinv6 * rinv6;
    float r_ron = (r2 > d_setup.ron2) ? (r - d_setup.ron) : 0.0f;
    float r_ron2_r = r_ron * r_ron * r;

    fij_vdw = c6 * (rinv6 + (d_setup.ga6 + d_setup.gb6 * r_ron) * r_ron2_r) -
              c12 * (rinv12 + (d_setup.ga12 + d_setup.gb12 * r_ron) * r_ron2_r);

    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      const float one_third = (float)(1.0 / 3.0);
      float r_ron3 = r_ron * r_ron * r_ron;
      pot_vdw =
          c6 * (-one_six * rinv6 +
                (one_third * d_setup.ga6 + 0.25f * d_setup.gb6 * r_ron) *
                    r_ron3 +
                d_setup.gc6) +
          c12 * (one_twelve * rinv12 -
                 (one_third * d_setup.ga12 + 0.25f * d_setup.gb12 * r_ron) *
                     r_ron3 -
                 d_setup.gc12);
    }
    /*
    if (r > ctonnb) then
             d = 6.0f/r**7 + GA6*(r-ctonnb)**2 + GB6*(r-ctonnb)**3
             d = -(12.0f/r**13 + GA12*(r-ctonnb)**2 + GB12*(r-ctonnb)**3)

             e = -r**(-6) + (GA6*(r-ctonnb)**3)/3.0 +
    (GB6*(r-ctonnb)**4)/4.0 + GC6 e = r**(-12) - (GA12*(r-ctonnb)**3)/3.0 -
    (GB12*(r-ctonnb)**4)/4.0
    - GC12

          else
             d = 6.0f/r**7
             d = -12.0f/r**13

             e = - r**(-6) + GC6
             e = r**(-12) - GC12
          endif
    */
  } else if (vdw_model == VDW_DBEXP) {

    const float alpha = 17.470f;
    const float beta = 4.099f;
    const float k1 = beta * expf(alpha) / (alpha - beta);
    const float k2 = alpha * expf(beta) / (alpha - beta);
    const float fk1 = -k1 * alpha;
    const float fk2 = -k2 * beta;

    // TODO : this is a bad way of doing this
    // Just for test for now
    if (fabsf(c12) < 1e-7) {
      fij_vdw = 0.0f;
      if (calc_energy)
        pot_vdw = 0.0f;
    } else {
      const float sig_inv_6 = c6 / c12;
      const float sig_inv = __powf(sig_inv_6, 1.0f / 6.0f);
      const float eps = c6 * sig_inv_6 / 12.0f;
      fij_vdw = eps * (fk1 * expf(-alpha * r * sig_inv) * sig_inv -
                       fk2 * expf(-beta * r * sig_inv) * sig_inv);
      // fij_vdw = static_cast<float>(fij_vdw_d);
      if (calc_energy)
        pot_vdw = eps * (k1 * expf(-alpha * r * sig_inv) -
                         k2 * expf(-beta * r * sig_inv));
    }

  } else if (vdw_model == VDW_SC) {
    float rinv3 = rinv * rinv2;
    float rinv6 = rinv3 * rinv3;
    float A6 = (r2 > d_setup.ron2) ? d_setup.k6 : 1.0f;
    float B6 = (r2 > d_setup.ron2) ? d_setup.roffinv3 : 0.0f;
    float A12 = (r2 > d_setup.ron2) ? d_setup.k12 : 1.0f;
    float B12 = (r2 > d_setup.ron2) ? d_setup.roffinv6 : 0.0f;
    fij_vdw =
        c6 * A6 * (rinv3 - B6) * rinv3 - c12 * A12 * (rinv6 - B12) * rinv6;
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      float C6 = (r2 > d_setup.ron2) ? 0.0f : d_setup.dv6;
      float C12 = (r2 > d_setup.ron2) ? 0.0f : d_setup.dv12;

      float rinv3_B6_sq = rinv3 - B6;
      rinv3_B6_sq *= rinv3_B6_sq;

      float rinv6_B12_sq = rinv6 - B12;
      rinv6_B12_sq *= rinv6_B12_sq;

      pot_vdw = one_twelve * c12 * (A12 * rinv6_B12_sq + C12) -
                one_six * c6 * (A6 * rinv3_B6_sq + C6);
    }
  } else if (vdw_model == NONE) {
    fij_vdw = 0.0f;
    if (calc_energy) {
      pot_vdw = 0.0f;
    }
  }

  return fij_vdw;
}

// static texture<float, 1, cudaReadModeElementType> ewald_force_texref;

//
// Returns simple linear interpolation
// NOTE: Could the interpolation be done implicitly using the texture unit?
//
__forceinline__ __device__ float lookup_force(const float r, const float hinv) {
  float r_hinv = r * hinv;
  int ind = (int)r_hinv;
  float f1 = r_hinv - (float)ind;
  float f2 = 1.0f - f1;
#if __CUDA_ARCH__ < 350
  return f1 * d_setup.ewald_force[ind] + f2 * d_setup.ewald_force[ind + 1];
#else
  return f1 * __ldg(&d_setup.ewald_force[ind]) +
         f2 * __ldg(&d_setup.ewald_force[ind + 1]);
#endif
  // return f1*tex1Dfetch(ewald_force_texref, ind) +
  // f2*tex1Dfetch(ewald_force_texref, ind+1);
}

//
// Switching function from CHARMM J. Comp. Chem. 1983 paper
// -------------------------------------------------------------
// if (x <= xon) then
//   sw = one
// elseif (x > xoff) then
//   sw = zero
// else
//   sw = (xoff-x)**2*(xoff + 2.0f*x - 3.0f*xon)/(xoff-xon)**3
// endif
// -------------------------------------------------------------
//
__forceinline__ __device__ float sw(const float x, const float xon,
                                    const float xoff,
                                    const float inv_xoff_xon3) {
  float res = 0.0f;
  if (x <= xoff) {
    res = (x <= xon) ? 1.0f
                     : (xoff - x) * (xoff - x) *
                           (xoff + 2.0f * x - 3.0f * xon) * inv_xoff_xon3;
  }
  return res;
}

//
// Derivative of switching function from CHARMM J. Comp. Chem. 1983 paper
// -------------------------------------------------------------
//    if (x <= xon) then
//       dsw = zero
//    elseif (x > xoff) then
//       dsw = zero
//    else
//       dsw = six*(xoff-x)*(xon-x)/(xoff-xon)**3
//    endif
// -------------------------------------------------------------
//
__forceinline__ __device__ float dsw(const float x, const float xon,
                                     const float xoff,
                                     const float inv_xoff_xon3) {
  float res = 0.0f;
  if (x > xon && x <= xoff) {
    res = 6.0f * (xoff - x) * (xon - x) * inv_xoff_xon3;
  }
  return res;
}

/**
 *
 * @brief Calculates electrostatic force & energy
 *
 */
template <int elec_model, bool calc_energy, bool use_e14fac>
__forceinline__ __device__ float
pair_elec_force(const float r2, const float r, const float rinv, float qq,
                const float e14fac, float &pot_elec) {
  float fij_elec;

  if (use_e14fac && elec_model != EWALD_LOOKUP && elec_model != EWALD) {
    // If we're using non-Ewald method, e14fac scales the charges
    qq *= e14fac;
  }

  if (elec_model == EWALD_LOOKUP) {
    fij_elec = qq * lookup_force(r, d_setup.hinv);
  } else if (elec_model == EWALD) {
    float erfc_val = fasterfc(d_setup.kappa * r);
    float exp_val = expf(-d_setup.kappa2 * r2);
    float qq_efac_rinv;
    if (use_e14fac) {
      qq_efac_rinv = qq * (erfc_val + e14fac - 1.0f) * rinv;
    } else {
      qq_efac_rinv = qq * erfc_val * rinv;
    }
    if (calc_energy) {
      pot_elec = qq_efac_rinv;
    }
    const float two_sqrtpi = 1.12837916709551f; // 2/sqrt(pi)
    fij_elec = -qq * two_sqrtpi * d_setup.kappa * exp_val - qq_efac_rinv;

    /*
    float erfc_val = fasterfc(d_setup.kappa*r);
    float exp_val = expf(-d_setup.kappa2*r2);
    if (calc_energy) {
      pot_elec = qq*erfc_val*rinv;
    }
    const float two_sqrtpi = 1.12837916709551f;    // 2/sqrt(pi)
    fij_elec = qq*(two_sqrtpi*d_setup.kappa*exp_val + erfc_val*rinv);
    */

  } else if (elec_model == CSHIFT) {
    fij_elec = -qq * (rinv - r * d_setup.roffinv2);
    if (calc_energy) {
      pot_elec = qq * rinv *
                 (1.0f - 2.0f * r * d_setup.roffinv + r2 * d_setup.roffinv2);
    }
  } else if (elec_model == CFSWIT) {
    float r3 = r2 * r;
    float r5 = r3 * r2;
    fij_elec =
        (r <= d_setup.ron)
            ? -qq * rinv
            : -qq * (d_setup.Aconst * rinv + d_setup.Bconst * r +
                     3.0f * d_setup.Cconst * r3 + 5.0f * d_setup.Dconst * r5);
    if (calc_energy) {
      pot_elec = (r <= d_setup.ron)
                     ? qq * (rinv + d_setup.dvc)
                     : qq * (d_setup.Aconst * (rinv - d_setup.roffinv) +
                             d_setup.Bconst * (d_setup.roff - r) +
                             d_setup.Cconst * (d_setup.roff3 - r3) +
                             d_setup.Dconst * (d_setup.roff5 - r5));
    }
  } else if (elec_model == CSHFT) {
    // Shift 1/r energy
    float tmp = (1.0f - r2 * d_setup.roffinv2);
    fij_elec = -qq * (rinv * tmp * tmp + 4.0f * r * d_setup.roffinv2 * tmp);
    if (calc_energy) {
      pot_elec = qq * rinv * tmp * tmp;
    }
  } else if (elec_model == CSWIT) {
    // Switch 1/r energy
    float tmp = sw(r2, d_setup.ron2, d_setup.roff2, d_setup.inv_roff2_ron2_3);
    fij_elec = -qq * (rinv * tmp - 2.0f * r *
                                       dsw(r2, d_setup.ron2, d_setup.roff2,
                                           d_setup.inv_roff2_ron2_3));
    if (calc_energy) {
      pot_elec = qq * rinv * tmp;
    }
  } else if (elec_model == RSWIT) {
    // Switch 1/r^2 energy
    float rinv2 = rinv * rinv;
    float tmp = sw(r2, d_setup.ron2, d_setup.roff2, d_setup.inv_roff2_ron2_3);
    fij_elec = -2.0f * qq *
               (rinv2 * tmp -
                dsw(r2, d_setup.ron2, d_setup.roff2, d_setup.inv_roff2_ron2_3));
    if (calc_energy) {
      pot_elec = qq * rinv2 * tmp;
    }
  } else if (elec_model == RSHFT) {
    // Shift 1/r^2 energy
    float rinv2 = rinv * rinv;
    float tmp = (1.0f - r2 * d_setup.roffinv2);
    fij_elec = -qq * (2.0f * rinv2 * tmp * tmp + 4.0f * d_setup.roffinv2 * tmp);
    if (calc_energy) {
      pot_elec = qq * rinv2 * tmp * tmp;
    }
  } else if (elec_model == RSHIFT) {
    // Shift 1/r^2 force with (r/rc -1)
    fij_elec = -qq * rinv * 2.0f * (rinv - d_setup.roffinv);
    if (calc_energy) {
      pot_elec = qq * rinv * rinv *
                 (1.0f - 2.0f * r * d_setup.roffinv + r2 * d_setup.roffinv2);
    }
  } else if (elec_model == RFSWIT) {
    // Switch 1/r^2 force
    float rinv2 = rinv * rinv;
    fij_elec = (r <= d_setup.ron)
                   ? -2.0f * qq * rinv2
                   : -2.0f * qq *
                         (d_setup.Acoef * rinv2 + d_setup.Bcoef +
                          d_setup.Ccoef * r2 + 2.0f * d_setup.Denom * r2 * r2);
    if (calc_energy) {
      pot_elec =
          (r <= d_setup.ron)
              ? qq * (rinv2 + d_setup.Eaddr)
              : qq * (d_setup.Acoef * rinv2 - 2.0f * d_setup.Bcoef * logf(r) -
                      r2 * (d_setup.Ccoef + r2 * d_setup.Denom) +
                      d_setup.Constr);
    }
  } else if (elec_model == GSHFT) {
    // GROMACS style shift 1/r^2 force
    // MGL special casing ctonnb=0 might speed this up
    // NOTE THAT THIS EXPLICITLY ASSUMES ctonnb = 0
    // ctofnb4 = ctofnb2*ctofnb2
    // ctofnb5 = ctofnb4*ctofnb
    fij_elec =
        -qq *
        (rinv -
         (5.0f * d_setup.roffinv4 * r - 4.0f * d_setup.roffinv5 * r2) * r2);
    if (calc_energy) {
      pot_elec = qq * (rinv - d_setup.GAconst +
                       (d_setup.GBcoef * r - d_setup.roffinv5 * r2) * r2);
    }
    // d = -qscale*(one/r2 - 5.0*r2/ctofnb4 +4*r2*r/ctofnb5)
    // e = qscale*(one/r - GAconst + r*r2*GBcoef - r2*r2/ctofnb5)
  } else if (elec_model == NONE) {
    fij_elec = 0.0f;
    if (calc_energy) {
      pot_elec = 0.0f;
    }
  }

  return fij_elec;
}

/*
//
// Calculates electrostatic force & energy for 1-4 interactions and exclusions
//
template <int elec_model, bool calc_energy>
__forceinline__ __device__
float pair_elec_force_14(const float r2, const float r, const float rinv,
                         const float qq, const float e14fac, float &pot_elec) {

  float fij_elec;

  if (elec_model == EWALD) {
    float erfc_val = fasterfc(d_setup.kappa*r);
    float exp_val = expf(-d_setup.kappa2*r2);
    float qq_efac_rinv = qq*(erfc_val + e14fac - 1.0f)*rinv;
    if (calc_energy) {
      pot_elec = qq_efac_rinv;
    }
    const float two_sqrtpi = 1.12837916709551f;    // 2/sqrt(pi)
    fij_elec = -qq*two_sqrtpi*d_setup.kappa*exp_val - qq_efac_rinv;
  } else if (elec_model == NONE) {
    fij_elec = 0.0f;
    if (calc_energy) {
      pot_elec = 0.0f;
    }
  }

  return fij_elec;
}
*/

//
// 1-4 exclusion force
//
template <typename AT, typename CT, int elec_model, bool calc_energy,
          bool calc_virial>
__device__ void
calc_ex14_force_device(const int pos, const xx14list_t *ex14list,
                       const float4 *xyzq, const float fscale, const int stride,
                       AT *force, double &elec_pot,
                       Virial_t *__restrict__ virial) {
  int i = ex14list[pos].i;
  int j = ex14list[pos].j;
  int ish = ex14list[pos].ishift;
  float shx, shy, shz;
  calc_box_shift<float>(ish, d_setup.boxx, d_setup.boxy, d_setup.boxz, shx, shy,
                        shz);
  // Load atom coordinates
  float4 xyzqi = xyzq[i];
  float4 xyzqj = xyzq[j];
  // Calculate distance
  CT dx = xyzqi.x - xyzqj.x + shx;
  CT dy = xyzqi.y - xyzqj.y + shy;
  CT dz = xyzqi.z - xyzqj.z + shz;
  CT r2 = dx * dx + dy * dy + dz * dz;
  CT qq = ccelec * xyzqi.w * xyzqj.w;
  // Calculate the interaction
  CT r = sqrtf(r2);
  CT rinv = ((CT)1) / r;
  CT rinv2 = rinv * rinv;
  float dpot_elec;

  CT fij_elec = pair_elec_force<elec_model, calc_energy, true>(r2, r, rinv, qq,
                                                               0.0f, dpot_elec);

  if (calc_energy)
    elec_pot += (double)dpot_elec;
  CT fij = fij_elec * rinv2 * fscale;
  // Calculate force components
  AT fxij, fyij, fzij;
  calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

  // Store forces
  write_force<AT>(fxij, fyij, fzij, i, stride, force);
  write_force<AT>(-fxij, -fyij, -fzij, j, stride, force);
  // Store shifted forces
  if (calc_virial) {
    if (ish != 13) {
      atomicAdd(&virial->sforce_dp[ish][0], (double)(fij * dx));
      atomicAdd(&virial->sforce_dp[ish][1], (double)(fij * dy));
      atomicAdd(&virial->sforce_dp[ish][2], (double)(fij * dz));
      fxij /= CONVERT_TO_VIR;
      fyij /= CONVERT_TO_VIR;
      fzij /= CONVERT_TO_VIR;
      // atomicAdd((unsigned long long int
      // *)&energy_virial->sforce_fp[ish-1], llitoulli(fxij));
      // atomicAdd((unsigned long long int
      // *)&energy_virial->sforce_fp[ish], llitoulli(fyij));
      // atomicAdd((unsigned long long int
      // *)&energy_virial->sforce_fp[ish+1], llitoulli(fzij));
    }
    // sforce(is)   = sforce(is)   + fijx
    // sforce(is+1) = sforce(is+1) + fijy
    // sforce(is+2) = sforce(is+2) + fijz
  }
}

//
// 1-4 interaction force
//
template <typename AT, typename CT, int vdw_model, int elec_model,
          bool calc_energy, bool calc_virial, bool calc_softcore,
          bool tex_vdwparam>
__device__ void calc_in14_force_device(
#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t vdwParam14TexObj,
#endif
    const int pos, const xx14list_t *in14list, const int *vdwtype,
    const float *vdwparam14, const float4 *xyzq, const float fscale,
    const int stride, AT *force, double &vdw_pot, double &elec_pot,
    double &soft_f, Virial_t *__restrict__ virial) {

  int i = in14list[pos].i;
  int j = in14list[pos].j;
  int ish = in14list[pos].ishift;
  float shx, shy, shz;
  calc_box_shift<float>(ish, d_setup.boxx, d_setup.boxy, d_setup.boxz, shx, shy,
                        shz);
  // Load atom coordinates
  float4 xyzqi = xyzq[i];
  float4 xyzqj = xyzq[j];
  // Calculate distance
  CT dx = xyzqi.x - xyzqj.x + shx;
  CT dy = xyzqi.y - xyzqj.y + shy;
  CT dz = xyzqi.z - xyzqj.z + shz;
  CT r2 = dx * dx + dy * dy + dz * dz;
  CT qq = ccelec * xyzqi.w * xyzqj.w;
  // Calculate the interaction
  CT r = sqrtf(r2);
  CT rinv = ((CT)1) / r;

  int ia = vdwtype[i];
  int ja = vdwtype[j];
  int aa = max(ja, ia);

  CT c6, c12;
  if (tex_vdwparam) {
    int ivdw = aa * (aa - 1) / 2 + (ja + ia);
// c6 = __ldg(&vdwparam14[ivdw]);
// c12 = __ldg(&vdwparam14[ivdw+1]);
#ifdef USE_TEXTURE_OBJECTS
    float2 c6c12 = tex1Dfetch<float2>(vdwParam14TexObj, ivdw);
#else
    float2 c6c12 = tex1Dfetch(vdwparam14_texref, ivdw);
#endif
    c6 = c6c12.x;
    c12 = c6c12.y;
  } else {
    int ivdw = (aa * (aa - 1) + 2 * (ja + ia));
    c6 = vdwparam14[ivdw];
    c12 = vdwparam14[ivdw + 1];
  }

  CT fij;

  if (calc_softcore) {
    // Set to default values
    CT rp = r;
    CT drpdr = 1.0;
    CT drpds = 0.0;
    CT rp2 = r2;
    CT rpinv = rinv;

    CT rcsoft = ((CT)4.0) * (((CT)1.0) - fscale); // 2.0*sqrt(4.0)=4
    if (r < rcsoft) {
      CT rdivrcs = r / rcsoft;
      rp = ((CT)1.0) - ((CT)0.5) * rdivrcs;
      rp = rp * rdivrcs * rdivrcs * rdivrcs + ((CT)0.5);
      drpdr = ((CT)3.0) - ((CT)2.0) * rdivrcs;
      drpdr = drpdr * rdivrcs * rdivrcs;
      drpds = rp - drpdr * rdivrcs;
      drpds = ((CT)(-4.0)) * drpds; // -4.0=-2.0*sqrt(4.0)
      rp = rp * rcsoft; // rp = rcsoft*(0.5+rdivrcs**3-0.5*rdivrcs**4)
      rp2 = rp * rp;
      rpinv = rsqrtf(rp2);
    } // else: Use default values above for rp, drpdr, drpds, rp2, rpinv

    float dpot_vdw;
    CT fij_vdw = pair_vdw_force<vdw_model, calc_energy>(rp2, rp, rpinv, c6, c12,
                                                        dpot_vdw);
    if (calc_energy)
      vdw_pot += (double)dpot_vdw;

    float dpot_elec;
    CT fij_elec = pair_elec_force<elec_model, calc_energy, true>(
        rp2, rp, rpinv, qq, d_setup.e14fac, dpot_elec);
    if (calc_energy)
      elec_pot += (double)dpot_elec;

    // fij = (fij_vdw + fij_elec)*rpinv*rinv*drpdr*fscale;
    // soft_f += (double)((fij_vdw + fij_elec)*rpinv*fscale*drpds);

    fij = (fij_vdw + fij_elec) * rpinv * fscale;
    soft_f += (double)(fij * drpds);
    fij *= rinv * drpdr;

  } else {
    float dpot_vdw;
    CT fij_vdw =
        pair_vdw_force<vdw_model, calc_energy>(r2, r, rinv, c6, c12, dpot_vdw);
    // if (std::abs(xyzqi.x +3.709) < 0.01) printf("%f\t%f\t%f\t%f\n", xyzqi.x,
    // xyzqj.x, c6, dpot_vdw); printf("%f\t%f\t%f\t%f\n", xyzqi.x, xyzqj.x, c6,
    // dpot_vdw);
    if (calc_energy)
      vdw_pot += (double)dpot_vdw;

    float dpot_elec;
    CT fij_elec = pair_elec_force<elec_model, calc_energy, true>(
        r2, r, rinv, qq, d_setup.e14fac, dpot_elec);
    if (calc_energy)
      elec_pot += (double)dpot_elec;

    fij = (fij_vdw + fij_elec) * rinv * rinv * fscale;
  }

  // Calculate force components
  AT fxij, fyij, fzij;
  calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

  // Store forces
  write_force<AT>(fxij, fyij, fzij, i, stride, force);
  write_force<AT>(-fxij, -fyij, -fzij, j, stride, force);

  // Store shifted forces
  if (calc_virial) {
    if (ish != 13) {
      atomicAdd(&virial->sforce_dp[ish][0], (double)(fij * dx));
      atomicAdd(&virial->sforce_dp[ish][1], (double)(fij * dy));
      atomicAdd(&virial->sforce_dp[ish][2], (double)(fij * dz));
      fxij /= CONVERT_TO_VIR;
      fyij /= CONVERT_TO_VIR;
      fzij /= CONVERT_TO_VIR;
      // atomicAdd((unsigned long long int
      // *)&energy_virial->sforce_fp[ish-1], llitoulli(fxij));
      // atomicAdd((unsigned long long int
      // *)&energy_virial->sforce_fp[ish], llitoulli(fyij));
      // atomicAdd((unsigned long long int
      // *)&energy_virial->sforce_fp[ish+1], llitoulli(fzij)); sforce(is)
      // = sforce(is)   + fijx sforce(is+1) = sforce(is+1) + fijy
      // sforce(is+2) = sforce(is+2) + fijz
    }
  }
}

//
// 1-4 exclusion and interaction calculation kernels
//

//--------------------------------------------------------------
//-------------------- Regular version -------------------------
//--------------------------------------------------------------

#define CUDA_14_KERNEL_NAME calc_14_force_kernel
#include "CudaDirectForce14_util.h"
#undef CUDA_14_KERNEL_NAME

//------------------------------------------------------------
//-------------------- Block version -------------------------
//------------------------------------------------------------

#undef NUMBLOCK_LARGE

#define USE_BLOCK
#define CUDA_14_KERNEL_NAME calc_14_force_block_kernel
#include "CudaDirectForce14_util.h"
#undef USE_BLOCK
#undef CUDA_14_KERNEL_NAME

//------------------------------------------------------------
//-------------------- Softcore Block version ----------------
//------------------------------------------------------------

#undef NUMBLOCK_LARGE

#define USE_BLOCK
#define USE_BLOCK_SOFTCORE
#define CUDA_14_KERNEL_NAME calc_14_force_block_sc_kernel
#include "CudaDirectForce14_util.h"
#undef USE_BLOCK
#undef USE_BLOCK_SOFTCORE
#undef CUDA_14_KERNEL_NAME

#define CREATE_KERNEL(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, CALC_ENERGY,         \
                      CALC_VIRIAL, TEX_VDWPARAM, ...)                          \
  {                                                                            \
    KERNEL_NAME<AT, CT, tilesize, VDW_MODEL, ELEC_MODEL, CALC_ENERGY,          \
                CALC_VIRIAL, TEX_VDWPARAM>                                     \
        <<<nblock, nthread, shmem_size, stream>>>(__VA_ARGS__);                \
  }

#define CREATE_KERNEL14(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, CALC_ENERGY,       \
                        CALC_VIRIAL, TEX_VDWPARAM, ...)                        \
  {                                                                            \
    KERNEL_NAME<AT, CT, VDW_MODEL, ELEC_MODEL, CALC_ENERGY, CALC_VIRIAL,       \
                TEX_VDWPARAM>                                                  \
        <<<nblock, nthread, shmem_size, stream>>>(__VA_ARGS__);                \
  }

#define EXPAND_ENERGY_VIRIAL(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,           \
                             ELEC_MODEL, ...)                                  \
  {                                                                            \
    if (calc_energy) {                                                         \
      if (calc_virial) {                                                       \
        KERNEL_CREATOR(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, true, true,         \
                       USE_TEXTURES, __VA_ARGS__);                             \
      } else {                                                                 \
        KERNEL_CREATOR(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, true, false,        \
                       USE_TEXTURES, __VA_ARGS__);                             \
      }                                                                        \
    } else {                                                                   \
      if (calc_virial) {                                                       \
        KERNEL_CREATOR(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, false, true,        \
                       USE_TEXTURES, __VA_ARGS__);                             \
      } else {                                                                 \
        KERNEL_CREATOR(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, false, false,       \
                       USE_TEXTURES, __VA_ARGS__);                             \
      }                                                                        \
    }                                                                          \
  }

#define EXPAND_ENERGY_VIRIAL_NONE(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  ELEC_MODEL, ...)                             \
  { KERNEL_CREATOR(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, __VA_ARGS__); }

#define EXPAND_ELEC(EXPAND_ENERGY_VIRIAL_NAME, KERNEL_CREATOR, KERNEL_NAME,    \
                    VDW_MODEL, ...)                                            \
  {                                                                            \
    if (elec_model == EWALD) {                                                 \
      EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL, EWALD, \
                                __VA_ARGS__);                                  \
    } /*else if (elec_model == EWALD_LOOKUP) {                                 \
        EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  EWALD_LOOKUP, __VA_ARGS__);                  \
    } else if (elec_model == CSHIFT) {                                         \
        EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  CSHIFT, __VA_ARGS__);                        \
    } */                                                                       \
    else if (elec_model == CFSWIT) {                                           \
      EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,        \
                                CFSWIT, __VA_ARGS__);                          \
    } /*else if (elec_model == CSHFT) {                                        \
        EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  CSHFT, __VA_ARGS__);                         \
    } else if (elec_model == CSWIT) {                                          \
        EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  CSWIT, __VA_ARGS__);                         \
    } else if (elec_model == RSWIT) {                                          \
        EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  RSWIT, __VA_ARGS__);                         \
    } else if (elec_model == RSHFT) {                                          \
        EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  RSHFT, __VA_ARGS__);                         \
    } else if (elec_model == RSHIFT) {                                         \
        EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  RSHIFT, __VA_ARGS__);                        \
    } else if (elec_model == RFSWIT) {                                         \
        EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  RFSWIT, __VA_ARGS__);                        \
    } else if (elec_model == GSHFT) {                                          \
        EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  GSHFT, __VA_ARGS__);                         \
    } else if (elec_model == NONE) {                                           \
        EXPAND_ENERGY_VIRIAL_NAME(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL,      \
                                  NONE, __VA_ARGS__);                          \
    } */                                                                       \
    else {                                                                     \
      std::cout << __func__ << " Invalid EWALD model " << elec_model           \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  }

/**
 * @brief CREATE_KERNELS generates kernel based on several parameters.
 *
 * @param[in] EXPAND_ENERGY_VIRIAL
 *
 * @param[in] KERNEL_CREATOR
 *
 * @param[in] KERNEL_NAME
 *
 */

#define CREATE_KERNELS(EXPAND_ENERGY_VIRIAL_NAME, KERNEL_CREATOR, KERNEL_NAME, \
                       ...)                                                    \
  {                                                                            \
    if (vdw_model == VDW_VSH) {                                                \
      EXPAND_ELEC(EXPAND_ENERGY_VIRIAL_NAME, KERNEL_CREATOR, KERNEL_NAME,      \
                  VDW_VSH, __VA_ARGS__);                                       \
    } else if (vdw_model == VDW_VSW) {                                         \
      EXPAND_ELEC(EXPAND_ENERGY_VIRIAL_NAME, KERNEL_CREATOR, KERNEL_NAME,      \
                  VDW_VSW, __VA_ARGS__);                                       \
    } else if (vdw_model == VDW_VFSW) {                                        \
      EXPAND_ELEC(EXPAND_ENERGY_VIRIAL_NAME, KERNEL_CREATOR, KERNEL_NAME,      \
                  VDW_VFSW, __VA_ARGS__);                                      \
    } else if (vdw_model == VDW_CUT) {                                         \
      EXPAND_ELEC(EXPAND_ENERGY_VIRIAL_NAME, KERNEL_CREATOR, KERNEL_NAME,      \
                  VDW_CUT, __VA_ARGS__);                                       \
    } else if (vdw_model == VDW_VGSH) {                                        \
      EXPAND_ELEC(EXPAND_ENERGY_VIRIAL_NAME, KERNEL_CREATOR, KERNEL_NAME,      \
                  VDW_VGSH, __VA_ARGS__);                                      \
    } else if (vdw_model == VDW_DBEXP) {                                       \
      EXPAND_ELEC(EXPAND_ENERGY_VIRIAL_NAME, KERNEL_CREATOR, KERNEL_NAME,      \
                  VDW_DBEXP, __VA_ARGS__);                                     \
    } else {                                                                   \
      std::cout << __func__ << " Invalid VDW model " << vdw_model              \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  }

//--------------------------------------------------------------
//-------------------- Regular version -------------------------
//--------------------------------------------------------------

#define CUDA_KERNEL_NAME calcForceKernel
#include "CudaDirectForce_util.h"
#undef CUDA_KERNEL_NAME

//------------------------------------------------------------
//-------------------- Block version -------------------------
//------------------------------------------------------------

#undef NUMBLOCK_LARGE

#define USE_BLOCK
#define CUDA_KERNEL_NAME calcForceBlockKernel
#include "CudaDirectForce_util.h"
#undef USE_BLOCK
#undef CUDA_KERNEL_NAME

//------------------------------------------------------------
//-------------------- Softcore Block version ----------------
//------------------------------------------------------------

#undef NUMBLOCK_LARGE

#define USE_BLOCK
#define USE_BLOCK_SOFTCORE
#define CUDA_KERNEL_NAME calcForceBlockSCKernel
#include "CudaDirectForce_util.h"
#undef USE_BLOCK
#undef USE_BLOCK_SOFTCORE
#undef CUDA_KERNEL_NAME

//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------

template <typename AT, typename CT>
/**
 * @brief Generates a kernel (?)
 *
 * Using CREATE_KERNELS, generates a Kernel.
 */
void calcForceKernelChoice(const int nblock_tot_in, const int nthread,
                           const int shmem_size, cudaStream_t stream,
                           const int vdw_model, const int elec_model,
                           const bool calc_energy, const bool calc_virial,
                           const CudaNeighborListBuild<32> &nlist,
                           // const CudaP21NeighborListBuild &nlist,
                           const float *vdwparam, const int nvdwparam,
                           const int *vdwtype,
#ifdef USE_TEXTURE_OBJECTS
                           cudaTextureObject_t &vdwParamTexObj,
#endif
                           const float4 *xyzq, const int stride, AT *force,
                           Virial_t *virial, double *energy_vdw,
                           double *energy_elec, CudaBlock *cudaBlock,
                           AT *biflam, AT *biflam2) {

  int nblock_tot = nblock_tot_in;
  int3 max_nblock3 = get_max_nblock();
  unsigned int max_nblock = max_nblock3.x;
  unsigned int base = 0;

  while (nblock_tot != 0) {
    int nblock = (nblock_tot > max_nblock) ? max_nblock : nblock_tot;
    nblock_tot -= nblock;

    if (cudaBlock == NULL) {

#ifdef USE_TEXTURE_OBJECTS
      CREATE_KERNELS(EXPAND_ENERGY_VIRIAL, CREATE_KERNEL, calcForceKernel,
                     vdwParamTexObj, base, nlist.get_n_ientry(),
                     nlist.get_ientry(), nlist.get_tile_indj(),
                     nlist.get_tile_excl(), stride, vdwparam, nvdwparam, xyzq,
                     vdwtype, force, virial, energy_vdw, energy_elec);
#else
      CREATE_KERNELS(EXPAND_ENERGY_VIRIAL, CREATE_KERNEL, calcForceKernel, base,
                     nlist.get_n_ientry(), nlist.get_ientry(),
                     nlist.get_tile_indj(), nlist.get_tile_excl(), stride,
                     vdwparam, nvdwparam, xyzq, vdwtype, force, virial,
                     energy_vdw, energy_elec);
#endif
    } /*else if (cudaBlock->getUseSoftcore() >= 1) {
#ifdef USE_TEXTURE_OBJECTS
      CREATE_KERNELS(
          EXPAND_ENERGY_VIRIAL, CREATE_KERNEL, calcForceBlockSCKernel,
          vdwParamTexObj, base, nlist.get_n_ientry(), nlist.get_ientry(),
          nlist.get_tile_indj(), nlist.get_tile_excl(), stride, vdwparam,
          nvdwparam, xyzq, vdwtype, cudaBlock->getNumBlock(),
          cudaBlock->getBixlam(), cudaBlock->getBlockType(), biflam, biflam2,
          *(cudaBlock->getBlockParamTexObj()), force, virial, energy_vdw,
          energy_elec);
#else
      CREATE_KERNELS(EXPAND_ENERGY_VIRIAL, CREATE_KERNEL,
                     calcForceBlockSCKernel, base, nlist.get_n_ientry(),
                     nlist.get_ientry(), nlist.get_tile_indj(),
                     nlist.get_tile_excl(), stride, vdwparam, nvdwparam, xyzq,
                     vdwtype, cudaBlock->getNumBlock(), cudaBlock->getBixlam(),
                     cudaBlock->getBlockType(), biflam, biflam2, force, virial,
                     energy_vdw, energy_elec);
#endif
    } else {
#ifdef USE_TEXTURE_OBJECTS
      CREATE_KERNELS(EXPAND_ENERGY_VIRIAL, CREATE_KERNEL, calcForceBlockKernel,
                     vdwParamTexObj, base, nlist.get_n_ientry(),
                     nlist.get_ientry(), nlist.get_tile_indj(),
                     nlist.get_tile_excl(), stride, vdwparam, nvdwparam, xyzq,
                     vdwtype, cudaBlock->getNumBlock(), cudaBlock->getBixlam(),
                     cudaBlock->getBlockType(), biflam, biflam2,
                     *(cudaBlock->getBlockParamTexObj()), force, virial,
                     energy_vdw, energy_elec);
#else
      CREATE_KERNELS(EXPAND_ENERGY_VIRIAL, CREATE_KERNEL, calcForceBlockKernel,
                     base, nlist.get_n_ientry(), nlist.get_ientry(),
                     nlist.get_tile_indj(), nlist.get_tile_excl(), stride,
                     vdwparam, nvdwparam, xyzq, vdwtype,
                     cudaBlock->getNumBlock(), cudaBlock->getBixlam(),
                     cudaBlock->getBlockType(), biflam, biflam2, force, virial,
                     energy_vdw, energy_elec);
#endif
    }*/

    base += (nthread / warpsize) * nblock;

    cudaCheck(cudaGetLastError());
  }
}

template <typename AT, typename CT>
void calcForce14KernelChoice(
    const int nblock, const int nthread, const int shmem_size,
    cudaStream_t stream, const int vdw_model, const int elec_model,
    const bool calc_energy, const bool calc_virial, const int nin14list,
    const xx14list_t *in14list, const int nex14list, const xx14list_t *ex14list,
    const int nin14block, const int *in14TblBlockPos,
    const int *ex14TblBlockPos, const int *in14BlockToBlock,
    const int *ex14BlockToBlock, const int *in14BlockToTblPos,
    const int *ex14BlockToTblPos, const int *vdwtype, const float *vdwparam14,
#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t &vdwParam14TexObj,
#endif
    const float4 *xyzq, const float fscale, const int stride, AT *force,
    Virial_t *virial, double *energy_vdw, double *energy_elec,
    double *energy_excl, CudaBlock *cudaBlock) {

  if (nblock > 0) {
    if (cudaBlock == NULL) {
#ifdef USE_TEXTURE_OBJECTS
      CREATE_KERNELS(EXPAND_ENERGY_VIRIAL, CREATE_KERNEL14,
                     calc_14_force_kernel, vdwParam14TexObj, nin14list,
                     nex14list, nin14block, in14list, ex14list, vdwtype,
                     vdwparam14, xyzq, fscale, stride, force, virial,
                     energy_vdw, energy_elec, energy_excl);
#else
      CREATE_KERNELS(EXPAND_ENERGY_VIRIAL, CREATE_KERNEL14,
                     calc_14_force_kernel, nin14list, nex14list, nin14block,
                     in14list, ex14list, vdwtype, vdwparam14, xyzq, fscale,
                     stride, force, virial, energy_vdw, energy_elec,
                     energy_excl);
#endif
    } else if (cudaBlock->getUseSoftcore() == 2) {
#ifdef USE_TEXTURE_OBJECTS
      CREATE_KERNELS(EXPAND_ENERGY_VIRIAL, CREATE_KERNEL14,
                     calc_14_force_block_sc_kernel, vdwParam14TexObj, nin14list,
                     nex14list, nin14block, in14list, ex14list, in14TblBlockPos,
                     ex14TblBlockPos, in14BlockToBlock, ex14BlockToBlock,
                     in14BlockToTblPos, ex14BlockToTblPos, vdwtype, vdwparam14,
                     xyzq, *(cudaBlock->getBlockParamTexObj()),
                     cudaBlock->getUsePMEL(), cudaBlock->getBlockParamEx(),
                     cudaBlock->getDSoftDFscale(), stride, force, virial,
                     energy_vdw, energy_elec, energy_excl);
#else
      CREATE_KERNELS(
          EXPAND_ENERGY_VIRIAL, CREATE_KERNEL14, calc_14_force_block_sc_kernel,
          nin14list, nex14list, nin14block, in14list, ex14list, in14TblBlockPos,
          ex14TblBlockPos, in14BlockToBlock, ex14BlockToBlock,
          in14BlockToTblPos, ex14BlockToTblPos, vdwtype, vdwparam14, xyzq,
          // Use blockParamTexRef instead
          cudaBlock->getUsePMEL(), cudaBlock->getBlockParamEx(),
          cudaBlock->getDSoftDFscale(), stride, force, virial, energy_vdw,
          energy_elec, energy_excl);
#endif
    } else {
#ifdef USE_TEXTURE_OBJECTS
      CREATE_KERNELS(
          EXPAND_ENERGY_VIRIAL, CREATE_KERNEL14, calc_14_force_block_kernel,
          vdwParam14TexObj, nin14list, nex14list, nin14block, in14list,
          ex14list, in14TblBlockPos, ex14TblBlockPos, in14BlockToBlock,
          ex14BlockToBlock, in14BlockToTblPos, ex14BlockToTblPos, vdwtype,
          vdwparam14, xyzq, *(cudaBlock->getBlockParamTexObj()),
          cudaBlock->getUsePMEL(), cudaBlock->getBlockParamEx(), stride, force,
          virial, energy_vdw, energy_elec, energy_excl);
#else
      CREATE_KERNELS(
          EXPAND_ENERGY_VIRIAL, CREATE_KERNEL14, calc_14_force_block_kernel,
          nin14list, nex14list, nin14block, in14list, ex14list, in14TblBlockPos,
          ex14TblBlockPos, in14BlockToBlock, ex14BlockToBlock,
          in14BlockToTblPos, ex14BlockToTblPos, vdwtype, vdwparam14, xyzq,
          // Use blockParamTexRef instead
          cudaBlock->getUsePMEL(), cudaBlock->getBlockParamEx(), stride, force,
          virial, energy_vdw, energy_elec, energy_excl);
#endif
    }
  }

  cudaCheck(cudaGetLastError());
}

/*
void calcVirial(const int ncoord, const float4 *xyzq,
                DirectEnergyVirial_t* energy_virial,
                const int stride, double *force,
                cudaStream_t stream) {

  int nthread, nblock, shmem_size;
  nthread = 256;
  nblock = (ncoord+27-1)/nthread + 1;
  shmem_size = nthread*3*sizeof(double);

  calc_virial_kernel<<< nblock, nthread, shmem_size, stream>>>
    (ncoord, xyzq, stride, energy_virial, force);

  cudaCheck(cudaGetLastError());
}
*/

void updateDirectForceSetup(const DirectSettings_t *h_setup) {
  cudaCheck(cudaMemcpyToSymbol(d_setup, h_setup, sizeof(DirectSettings_t)));
}

// Explicit instances of templates:
template void calcForceKernelChoice<long long int, float>(
    const int nblock_tot_in, const int nthread, const int shmem_size,
    cudaStream_t stream, const int vdw_model, const int elec_model,
    const bool calc_energy, const bool calc_virial,
    const CudaNeighborListBuild<32> &nlist,
    // const CudaP21NeighborListBuild &nlist,

    const float *vdwparam, const int nvdwparam, const int *vdwtype,
#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t &vdwParamTexObj,
#endif
    const float4 *xyzq, const int stride, long long int *force,
    Virial_t *virial, double *energy_vdw, double *energy_elec,
    CudaBlock *cudaBlock, long long int *biflam, long long int *biflam2);

template void calcForce14KernelChoice<long long int, float>(
    const int nblock, const int nthread, const int shmem_size,
    cudaStream_t stream, const int vdw_model, const int elec_model,
    const bool calc_energy, const bool calc_virial, const int nin14list,
    const xx14list_t *in14list, const int nex14list, const xx14list_t *ex14list,
    const int nin14block, const int *in14TblBlockPos,
    const int *ex14TblBlockPos, const int *in14BlockToBlock,
    const int *ex14BlockToBlock, const int *in14BlockToTblPos,
    const int *ex14BlockToTblPos, const int *vdwtype, const float *vdwparam14,
#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t &vdwParam14TexObj,
#endif
    const float4 *xyzq, const float fscale, const int stride,
    long long int *force, Virial_t *virial, double *energy_vdw,
    double *energy_elec, double *energy_excl, CudaBlock *cudaBlock);

// void calcVirial(const int ncoord, const float4 *xyzq,
//		DirectEnergyVirial_t* energy_virial,
//		const int stride, double* force,
//		cudaStream_t stream);
#endif // NOCUDAC
