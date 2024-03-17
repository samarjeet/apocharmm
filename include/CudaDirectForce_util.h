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
#include "gpu_utils.h"
//
// Nonbonded force kernel
//
template <typename AT, typename CT, int tilesize, int vdw_model, int elec_model,
          bool calc_energy, bool calc_virial, bool tex_vdwparam>
__global__ void CUDA_KERNEL_NAME(
#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t vdwParamTexObj,
#endif
    const int base, const int n_ientry, const ientry_t *__restrict__ ientry,
    const int *__restrict__ tile_indj,
    const tile_excl_t<tilesize> *__restrict__ tile_excl, const int stride,
    const float *__restrict__ vdwparam, const int nvdwparam,
    const float4 *__restrict__ xyzq, const int *__restrict__ vdwtype,
    AT *__restrict__ force, Virial_t *__restrict__ virial,
    double *__restrict__ energy_vdw, double *__restrict__ energy_elec) {
  // Pre-computed constants
  const int num_excl = ((tilesize * tilesize - 1) / 32 + 1);
  const int num_thread_per_excl = (32 / num_excl);

  //
  // Shared data, common for the entire block
  //
  extern __shared__ char shmem[];

  // const unsigned int sh_start = tilesize*threadIdx.y;

  // Warp index (0...warpsize-1)
  const int wid = threadIdx.x % warpsize;

  // Load index (0...15 or 0...31)
  const int lid = (tilesize == 16) ? (wid % tilesize) : wid;

  int shmem_pos = 0;
  //
  // Shared memory requirements:
  // sh_xi, sh_yi, sh_zi, sh_qi: (blockDim.x/warpsize)*tilesize*sizeof(float)
  // sh_vdwtypei               : (blockDim.x/warpsize)*tilesize*sizeof(int)
  // sh_blocktypei               : (blockDim.x/warpsize)*tilesize*sizeof(int)
  // sh_fix, sh_fiy, sh_fiz    : (blockDim.x/warpsize)*warpsize*sizeof(AT)
  // sh_vdwparam               : nvdwparam*sizeof(float)
  //
  // ## For USE_BLOCK ##
  // sh_blocktypei             : (blockDim.x/warpsize)*tilesize*sizeof(int)
  // sh_bixlam                 : numBlock*sizeof(float)
  //
  // (x_i, y_i, z_i, q_i, vdwtype_i) are private to each warp
  // (fix, fiy, fiz) are private for each warp
  // vdwparam_sh is for the entire thread block

  volatile AT *sh_fix = (AT *)&shmem[shmem_pos + (threadIdx.x / warpsize) *
                                                     warpsize * sizeof(AT)];
  shmem_pos += (blockDim.x / warpsize) * warpsize * sizeof(AT);
  volatile AT *sh_fiy = (AT *)&shmem[shmem_pos + (threadIdx.x / warpsize) *
                                                     warpsize * sizeof(AT)];
  shmem_pos += (blockDim.x / warpsize) * warpsize * sizeof(AT);
  volatile AT *sh_fiz = (AT *)&shmem[shmem_pos + (threadIdx.x / warpsize) *
                                                     warpsize * sizeof(AT)];
  shmem_pos += (blockDim.x / warpsize) * warpsize * sizeof(AT);

  float *sh_vdwparam;
  if (!tex_vdwparam) {
    sh_vdwparam = (float *)&shmem[shmem_pos];
    shmem_pos += nvdwparam * sizeof(float);
  }

  // Load ientry. Single warp takes care of one ientry
  const int ientry_ind =
      (threadIdx.x + blockDim.x * blockIdx.x) / warpsize + base;

  int indi, ish, startj, endj;
  if (ientry_ind < n_ientry) {
    indi = ientry[ientry_ind].iatomStart;
    ish = ientry[ientry_ind].ish;
    startj = ientry[ientry_ind].tileStart;
    endj = ientry[ientry_ind].tileEnd;
  } else {
    indi = 0;
    ish = 1;
    startj = 1;
    endj = 0;
  }

  // Calculate shift for i-atom
  // ish = 1...26*3+1
  float shx, shy, shz;
  calc_box_shift<float>(ish, d_setup.boxx, d_setup.boxy, d_setup.boxz, shx, shy,
                        shz);

  // Load i-atom data to shared memory (and shift coordinates)
  float4 xyzq_tmp = xyzq[indi + lid];
  float xi = xyzq_tmp.x + shx;
  float yi, zi;
  if (shx != 0 && d_setup.q_p21) {
    yi = -xyzq_tmp.y + shy;
    zi = -xyzq_tmp.z + shz;
  } else {
    yi = xyzq_tmp.y + shy;
    zi = xyzq_tmp.z + shz;
  }
  float qi = xyzq_tmp.w * ccelec;
  int vdwtypei = vdwtype[indi + lid];

  sh_fix[wid] = (AT)0;
  sh_fiy[wid] = (AT)0;
  sh_fiz[wid] = (AT)0;

  if (!tex_vdwparam) {
    // Copy vdwparam to shared memory
    for (int i = threadIdx.x; i < nvdwparam; i += blockDim.x)
      sh_vdwparam[i] = vdwparam[i];
    __syncthreads();
  }

  double vdwpotl;
  double coulpotl;
  if (calc_energy) {
    vdwpotl = 0.0;
    coulpotl = 0.0;
  }

  for (int jtile = startj; jtile <= endj; jtile++) {
    // Load j-atom starting index and exclusion mask
    unsigned int excl;
    if (tilesize == 16) {
      // For 16x16 tile, the exclusion mask per is 8 bits per thread:
      // NUM_THREAD_PER_EXCL = 4
      excl = tile_excl[jtile].excl[wid / num_thread_per_excl] >>
             ((wid % num_thread_per_excl) * num_excl);
    } else {
      excl = tile_excl[jtile].excl[wid];
    }
    int indj = tile_indj[jtile];

    // Skip empty tile
    if (ALL(~excl == 0))
      continue;

    float4 xyzq_j = xyzq[indj + lid];
    int ja = vdwtype[indj + lid];

    // Clear j forces
    AT fjx = (AT)0;
    AT fjy = (AT)0;
    AT fjz = (AT)0;

    for (int t = 0; t < num_excl; t++) {
      int ii;
      if (tilesize == 16) {
        ii = (wid + t * 2 + (wid / tilesize) * (tilesize - 1)) % tilesize;
      } else {
        ii = ((wid + t) % tilesize);
      }

      float dx = SHFL(xi, ii) - xyzq_j.x;
      float dy = SHFL(yi, ii) - xyzq_j.y;
      float dz = SHFL(zi, ii) - xyzq_j.z;

      float r2 = dx * dx + dy * dy + dz * dz;

      float qq = SHFL(qi, ii) * xyzq_j.w;

      int ia = SHFL(vdwtypei, ii);

      if (!(excl & 1) && r2 < d_setup.roff2) {
        float rinv = rsqrtf(r2);
        float r = r2 * rinv;

        float dpot_elec;
        float fij_elec;
        fij_elec = pair_elec_force<elec_model, calc_energy, false>(
            r2, r, rinv, qq, 0.0f, dpot_elec);

        if (calc_energy)
          coulpotl += (double)dpot_elec;

        int aa = (ja > ia) ? ja : ia; // aa = max(ja,ia)
        float c6, c12;
        if (tex_vdwparam) {
          int ivdw = aa * (aa - 1) / 2 + (ja + ia);
// c6 = __ldg(&vdwparam[ivdw]);
// c12 = __ldg(&vdwparam[ivdw+1]);
#ifdef USE_TEXTURE_OBJECTS
          float2 c6c12 = tex1Dfetch<float2>(vdwParamTexObj, ivdw);
#else
          float2 c6c12 = tex1Dfetch(vdwparam_texref, ivdw);
#endif
          c6 = c6c12.x;
          c12 = c6c12.y;
        } else {
          int ivdw = (aa * (aa - 1) + 2 * (ja + ia));
          c6 = sh_vdwparam[ivdw];
          c12 = sh_vdwparam[ivdw + 1];
        }

        float dpot_vdw;
        float fij_vdw;

        fij_vdw = pair_vdw_force<vdw_model, calc_energy>(r2, r, rinv, c6, c12,
                                                         dpot_vdw);
        if (calc_energy)
          vdwpotl += (double)dpot_vdw;

        float fij = (fij_vdw + fij_elec) * rinv * rinv;

        AT fxij;
        AT fyij;
        AT fzij;
        calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

        fjx -= fxij;
        fjy -= fyij;
        fjz -= fzij;

        /*if (tilesize == 16) {
          // We need to re-calculate ii because ii must be warp sized in order
          // to
          // prevent race condition
          int tmp = (wid + t * 2) % 16 + (wid / 16) * 31;
          ii = tilesize * (threadIdx.x / warpsize) * 2 +
               (tmp + (tmp / 32) * 16) % 32;
        }
        */

        sh_fix[ii] += fxij;
        sh_fiy[ii] += fyij;
        sh_fiz[ii] += fzij;
      } // if (!(excl & 1) && r2 < d_setup.roff2)

      // Advance exclusion mask
      excl >>= 1;
    }

    // Dump register forces (fjx, fjy, fjz)
    write_force<AT>(fjx, fjy, fjz, indj + lid, stride, force);
  }

  if (shx != 0 && d_setup.q_p21) {
    sh_fiy[wid] = -sh_fiy[wid];
    sh_fiz[wid] = -sh_fiz[wid];
  }

  // Dump shared memory force (fi)
  // NOTE: no __syncthreads() required here because sh_fix is "volatile"
  write_force<AT>(sh_fix[wid], sh_fiy[wid], sh_fiz[wid], indi + lid, stride,
                  force);

  if (calc_virial) {
    // Virial is calculated from (sh_fix[], sh_fiy[], sh_fiz[])
    // Variable "ish" depends on warp => Reduce within warp
    // NOTE: we skip the center element because it doesn't contribute to the
    // virial
    if (ish != 13) {
      // Convert into double
      volatile double *sh_sfix = (double *)sh_fix;
      volatile double *sh_sfiy = (double *)sh_fiy;
      volatile double *sh_sfiz = (double *)sh_fiz;

      if (shx != 0 && d_setup.q_p21) {
        sh_fiy[wid] = -sh_fiy[wid];
        sh_fiz[wid] = -sh_fiz[wid];
      }

      sh_sfix[wid] = ((double)sh_fix[wid]) * INV_FORCE_SCALE;
      sh_sfiy[wid] = ((double)sh_fiy[wid]) * INV_FORCE_SCALE;
      sh_sfiz[wid] = ((double)sh_fiz[wid]) * INV_FORCE_SCALE;

      for (int d = 16; d >= 1; d /= 2) {
        if (wid < d) {
          sh_sfix[wid] += sh_sfix[wid + d];
          sh_sfiy[wid] += sh_sfiy[wid + d];
          sh_sfiz[wid] += sh_sfiz[wid + d];
        }
      }
      if (wid == 0) {
        atomicAdd(&virial->sforce_dp[ish][0], sh_sfix[0]);
        atomicAdd(&virial->sforce_dp[ish][1], sh_sfiy[0]);
        atomicAdd(&virial->sforce_dp[ish][2], sh_sfiz[0]);
      }
    }
  }

  if (calc_energy) {
    // Reduce energies across the entire thread block
    // Shared memory required:
    // blockDim.x*sizeof(double)*2
    __syncthreads();
    double2 *sh_pot = (double2 *)(shmem);
    sh_pot[threadIdx.x].x = vdwpotl;
    sh_pot[threadIdx.x].y = coulpotl;
    __syncthreads();
    for (int i = 1; i < blockDim.x; i *= 2) {
      int pos = threadIdx.x + i;
      double vdwpot_val = (pos < blockDim.x) ? sh_pot[pos].x : 0.0;
      double coulpot_val = (pos < blockDim.x) ? sh_pot[pos].y : 0.0;
      __syncthreads();
      sh_pot[threadIdx.x].x += vdwpot_val;
      sh_pot[threadIdx.x].y += coulpot_val;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicAdd(energy_vdw, sh_pot[0].x);
      atomicAdd(energy_elec, sh_pot[0].y);
    }
  }
}
#endif // NOCUDAC
