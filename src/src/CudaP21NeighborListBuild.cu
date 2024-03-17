// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaP21NeighborListBuild.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <fstream>
#include <iostream>

CudaP21NeighborListBuild::CudaP21NeighborListBuild() {

  // may be make _lo 0
  imx_lo = -1;
  imx_hi = 1;
  imy_lo = -1;
  imy_hi = 1;
  imz_lo = -1;
  imz_hi = 1;
  // std::cout << "[p21nlbuild] Constructor called" << std::endl;

  ientry = NULL;
  ientry_len = 0;
  ientry_raw = NULL;
  tile_indj = NULL;

  exclAtomHeap = NULL;
  tile_excl = NULL;
  //bucketPos = NULL;
  allocate<int>(&bucketPos, n_jlist_max + 1);
  bucketIndex = NULL;
}

//
// Calculates exclusive plus-scan across warp for binary (0 or 1) values
//
// wid = warp ID = threadIdx.x % warpsize
//
__forceinline__ __device__ int binary_excl_scan_p21(int val, int wid) {
  return __popc(BALLOT(val) & ((1 << wid) - 1));
  // return 1;
}

//
// Calculates reduction across warp for binary (0 or 1) values. Result is in all
// threads within the warp
//
__forceinline__ __device__ int binary_reduce_p21(int val) {
  return __popc(BALLOT(val));
  // return 1;
}

// One warp handles one cell
struct shVars_t {
  bb_t ibb;
};

__device__ void get_cell_bounds_z_p21(const int icell, const int ncell,
                                      const float minx, const float x0,
                                      const float x1,
                                      const float *__restrict__ bx,
                                      const float rcut, int &jcell0,
                                      int &jcell1) {
  int jcell_start_left, jcell_start_right;

  if (icell < 0) {
    jcell_start_left = -1;
    jcell_start_right = 0;
    jcell0 = 0;
    jcell1 = -1;
  } else if (icell >= ncell) {
    jcell_start_left = ncell - 1;
    jcell_start_right = ncell;
    jcell0 = ncell;
    jcell1 = ncell - 1;
  } else {
    jcell_start_left = icell - 1;
    jcell_start_right = icell + 1;
    jcell0 = icell;
    jcell1 = icell;
  }

  for (int j = jcell_start_left; j >= 0; j--) {
    float d = x0 - bx[j];
    if (d > rcut)
      break;
    jcell0 = j;
  }

  for (int j = jcell_start_right; j < ncell; j++) {
    float bx_j = (j > 0) ? bx[j - 1] : minx;
    float d = bx_j - x1;
    if (d > rcut)
      break;
    jcell1 = j;
  }
}

__device__ __forceinline__ int min_shfl_local(int val) {
  //
  for (int i = 16; i >= 1; i /= 2)
    val = min(val, SHFL_XOR(val, i));
  return val;
  //
}
__device__ __forceinline__ int max_shfl_local(int val) {
  //
  for (int i = 16; i >= 1; i /= 2)
    val = max(val, SHFL_XOR(val, i));
  return val;
  //
}
__global__ void p21buildKernel(
    const int4 *__restrict__ cell_xyz_zone, const int *__restrict__ col_ncellz,
    const int *__restrict__ col_cell, const float *__restrict__ cell_bz,
    const float boxx, const float boxy, const float boxz, const float rcut,
    const float rcut2, const bb_t *__restrict__ bb, int *__restrict__ tile_indj,
    ientry_t *__restrict__ ientry, const ZoneParam_t *__restrict__ ZoneParam,
    NlistParam_t *__restrict__ NlistParam) {
  //

  extern __shared__ char shbuf[];

  const int warpSize = 32;
  const int wid = threadIdx.x % warpSize;
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int icell = index / warpSize;

  if (icell >= NlistParam->ncell) {
    return;
  }

  int4 icell_xyz_zone = cell_xyz_zone[icell];
  int icellz = icell_xyz_zone.z;

  int shbuf_pos = 0;

  // Temporary j-cell list. Each warp has its own jlist
  volatile int *sh_jcell =
      (int *)&shbuf[shbuf_pos +
                    (threadIdx.x / warpSize) * n_jlist_max * sizeof(int)];
  shbuf_pos += (blockDim.x / warpSize) * n_jlist_max * sizeof(int);
  shVars_t *shVars = (shVars_t *)&shbuf[shbuf_pos + (threadIdx.x / warpSize) *
                                                        sizeof(shVars_t)];

  if (wid == 0)
    shVars->ibb = bb[icell];

  for (int imx = NlistParam->imx_lo; imx <= NlistParam->imx_hi; imx++) {
    float imbbx0 = shVars->ibb.x + imx * boxx;
    float x_min = ZoneParam[0].min_xyz.x;
    float inv_dx = ZoneParam[0].inv_celldx;
    int ncellx = ZoneParam[0].ncellx;

    // find the jcell0 and jcell1 that this (image) cell might interact with
    // along the x-direction
    float x0 = imbbx0 - shVars->ibb.wx;
    float x1 = imbbx0 + shVars->ibb.wx;

    int jcellx_min = max(0, (int)floorf((x0 - rcut - x_min) * inv_dx));
    int jcellx_max =
        min(ncellx - 1, (int)ceil((x1 + rcut - x_min) * inv_dx) - 1);

    int n_jcellx = max(0, jcellx_max - jcellx_min + 1);

    if (n_jcellx == 0)
      continue;

    for (int imy = NlistParam->imy_lo; imy <= NlistParam->imy_hi; imy++) {
      float imbby0;
      if (imx == 0)
        imbby0 = shVars->ibb.y + imy * boxy;
      else
        imbby0 = -(shVars->ibb.y + imy * boxy); // flip y-coordinate
      float y_min = ZoneParam[0].min_xyz.y;
      float inv_dy = ZoneParam[0].inv_celldy;
      int ncelly = ZoneParam[0].ncelly;

      // find the jcell0 and jcell1 that this (image) cell might interact with
      // along the y-direction
      float y0 = imbby0 - shVars->ibb.wy;
      float y1 = imbby0 + shVars->ibb.wy;

      int jcelly_min = max(0, (int)floorf((y0 - rcut - y_min) * inv_dy));
      int jcelly_max =
          min(ncelly - 1, (int)ceil((y1 + rcut - y_min) * inv_dy) - 1);

      int n_jcelly = max(0, jcelly_max - jcelly_min + 1);

      if (n_jcelly == 0)
        continue;

      for (int imz = NlistParam->imz_lo; imz <= NlistParam->imz_hi; imz++) {
        float imbbz0;
        if (imx == 0)
          imbbz0 = shVars->ibb.z + imz * boxz;
        else
          imbbz0 = -(shVars->ibb.z + imz * boxz); // flip z-coordinate
        int ish = imx + 1 + 3 * (imy + 1 + 3 * (imz + 1));

        do {
          int n_jlist = 0;
          int n_jcellx_t = n_jcellx;
          int n_jcelly_t = n_jcelly;

          int total_xy = n_jcellx_t * n_jcelly_t;
          if (total_xy > 0) {
            int jcellz_min = 1 << 30;
            int jcellz_max = 0;

            for (int ibase = 0; ibase < total_xy; ibase += warpSize) {
              int i = ibase + wid;
              int jcellz0_t = 1 << 30;
              int jcellz1_t = 0;
              if (i < total_xy) {
                int jcelly = i / n_jcellx_t;
                int jcellx = i - jcelly * n_jcellx_t;

                // TODO : remove the next 2 lines as
                // jcellx_min and jcelly_min are 0
                jcellx += jcellx_min;
                jcelly += jcelly_min;

                int jcol = jcellx + jcelly * ZoneParam[0].ncellx;
                int jcell0 = col_cell[jcol];

                get_cell_bounds_z_p21(icellz + imz * col_ncellz[jcol],
                                      col_ncellz[jcol], ZoneParam[0].min_xyz.z,
                                      imbbz0 - shVars->ibb.wz,
                                      imbbz0 + shVars->ibb.wz, &cell_bz[jcell0],
                                      rcut, jcellz0_t, jcellz1_t);
              }
              jcellz_min = min(jcellz_min, jcellz0_t);
              jcellz_max = max(jcellz_max, jcellz1_t);
            }

            jcellz_min = min_shfl_local(jcellz_min);
            jcellz_max = max_shfl_local(jcellz_max);

            int n_jcellz_max = jcellz_max - jcellz_min + 1;
            int total_xyz = total_xy * n_jcellz_max;

            if (total_xyz > 0) {
              // Cells are ordered in (y, x, z)

              for (int ibase = 0; ibase < total_xyz; ibase += warpSize) {
                int i = ibase + wid;
                int ok = 0;
                int jcell;
                if (i < total_xyz) {
                  // calculating (jcellx, jcelly, jcellz)
                  int it = i;
                  int jcelly = it / (n_jcellx_t * n_jcellz_max);
                  it -= jcelly * (n_jcellx_t * jcellz_max);

                  int jcellx = it / n_jcellz_max;
                  int jcellz = it - jcellx * n_jcellz_max;

                  jcellx += jcellx_min;
                  jcelly += jcelly_min;
                  jcellz += jcellz_min;

                  // j column index
                  int jcol = jcellx + jcelly * ZoneParam[0].ncellx;
                  jcell = col_cell[jcol] + jcellz;

                  if ((icell <= jcell) && jcellz >= 0 &&
                      jcellz < col_ncellz[jcol]) {
                    float dx = max(0.0f, fabsf(imbbx0 - bb[jcell].x) -
                                             shVars->ibb.wx - bb[jcell].wx);
                    float dy = max(0.0f, fabsf(imbby0 - bb[jcell].y) -
                                             shVars->ibb.wy - bb[jcell].wy);
                    float dz = max(0.0f, fabsf(imbbz0 - bb[jcell].z) -
                                             shVars->ibb.wz - bb[jcell].wz);
                    float r2 = dx * dx + dy * dy + dz * dz;
                    ok = (r2 < rcut2);
                  }
                }

                int pos = binary_excl_scan(ok, wid);
                int n_jlist_add = binary_reduce(ok);

                // Flush if the sh_jcell[] buffer would become
                // full
                if ((n_jlist + n_jlist_add) > n_jlist_max) {
                  // Write sh_jcell[] into global memory
                  int tileStart;
                  if (wid == 0)
                    tileStart = atomicAdd(&NlistParam->n_tile, n_jlist);
                  tileStart = SHFL(tileStart, 0);

                  // temporarily store jcell here
                  for (int jj = wid; jj < n_jlist; jj += warpSize) {
                    tile_indj[tileStart + jj] = sh_jcell[jj];
                  }
                  // Add to ientry list in global memory
                  if (wid == 0) {
                    int ientry_ind = atomicAdd(&NlistParam->n_ientry, 1);
                    int tileEnd = tileStart + n_jlist - 1;
                    ientry_t ientry_val;
                    ientry_val.iatomStart = icell; // temporarily store icell
                                                   // here
                    ientry_val.ish = ish;
                    ientry_val.tileStart = tileStart;
                    ientry_val.tileEnd = tileEnd;
                    ientry[ientry_ind] = ientry_val;
                  }

                  n_jlist = 0;
                }

                // Add to the list
                if (ok)
                  sh_jcell[n_jlist + pos] = jcell;
                n_jlist += n_jlist_add;
              }

              if (n_jlist > 0) {
                // Write sh_jcell[] into global memory
                int tileStart;
                if (wid == 0) {
                  tileStart = atomicAdd(&NlistParam->n_tile, n_jlist);
                }
                tileStart = SHFL(tileStart, 0);

                // temporarily store jcell here
                for (int jj = wid; jj < n_jlist; jj += warpSize) {
                  tile_indj[tileStart + jj] = sh_jcell[jj];
                }
                // Add to ientry list in global memory
                if (wid == 0) {
                  int ientry_ind = atomicAdd(&NlistParam->n_ientry, 1);
                  int tileEnd = tileStart + n_jlist - 1;
                  ientry_t ientry_val;
                  ientry_val.iatomStart = icell; // temporarily store icell here
                  ientry_val.ish = ish;
                  ientry_val.tileStart = tileStart;
                  ientry_val.tileEnd = tileEnd;
                  ientry[ientry_ind] = ientry_val;
                }
              }
            }
          }
        } while (false); // TODO : remove this do-while
      }
    }
  }
  //
}

__forceinline__ __device__ void
p21flushAtomjNew(const int wid, const int min_atomj, const int max_atomj,
              const int n_atomj, const int reg_atomj, const int minExclAtom,
              const int maxExclAtom, const int numExclAtom,
              const int *__restrict__ exclAtom, const int tileStart,
              tile_excl_t<32> *__restrict__ tile_excl,
              const int *__restrict__ tile_indj) {
  int tilesize = 32;
  if ((min_atomj <= maxExclAtom) && (max_atomj >= minExclAtom)) {
    int atomj = (wid < n_atomj) ? (reg_atomj >> n_jlist_max_shift) : -1;
    for (int ibase = 0; ibase < numExclAtom; ibase += warpsize) {
      int i = ibase + wid;
      // Load excluded atom from global memory and check if there are any
      // possible exclusions
      int exclAtomI = (i < numExclAtom) ? (exclAtom[i] >> 5) : -1;
      int has_excl =
          BALLOT((exclAtomI >= min_atomj) && (exclAtomI <= max_atomj));
      // Loop through possible exclusions
      while (has_excl) {
        // Get bit position for the exclusion
        int bitpos = __ffs(has_excl) - 1;
        i = ibase + bitpos;
        exclAtomI = exclAtom[i];
        // Check exclAtomI vs. reg_atomj[0...warpsize-1]
        if ((exclAtomI >> 5) == atomj) {
          // Thread wid found exclusion between atomj and (exclAtomI &
          // 31) NOTE: Only a single thread per warp enters here
          int itile = tileStart + (reg_atomj & n_jlist_max_mask);
          int jatomStart = tile_indj[itile];
          int excl_ind = atomj - jatomStart;
          int excl_shift = ((exclAtomI & 31) - excl_ind + tilesize) % tilesize;
          unsigned int excl_mask = 1 << excl_shift;
          tile_excl[itile].excl[excl_ind] |= excl_mask;
        }
        // Remove bit from has_excl
        has_excl ^= (1 << bitpos);
      }
    }
  }
}

// Each warp handles one ientry element i.e. one icell
__global__ void p21buildExclKernel(
    const int maxNumExcl, const int cellStart,
    const int *__restrict__ cell_patom, const int *__restrict__ loc2glo,
    const int *__restrict__ glo2loc, const int *__restrict__ atom_excl_pos,
    const int *__restrict__ atom_excl, const float4 *__restrict__ xyzq,
    const float boxx, const float boxy, const float boxz, const float rcut2,
    int *__restrict__ exclAtomHeap, int *__restrict__ tile_indj,
    tile_excl_t<32> *__restrict__ tile_excl, ientry_t *__restrict__ ientry,
    const NlistParam_t *__restrict__ NlistParam, int *__restrict__ bucketPos,
    int *__restrict__ bucketIndex) {

  // Warp index
  const int wid = threadIdx.x % warpsize;

  const int ientry_ind = (threadIdx.x + blockIdx.x * blockDim.x) / warpsize;

  if (ientry_ind >= NlistParam->n_ientry)
    return;

  const int icell = ientry[ientry_ind].iatomStart;
  const int iatomStart = cell_patom[icell];
  const int iatomEnd = cell_patom[icell + 1] - 1;
  // Replace icell with iatomStart
  ientry[ientry_ind].iatomStart = iatomStart;

  // Allocate space for exclusions in global memory
  // Each warp (icell) has tilesize*maxNumExcl amount of space
  int *__restrict__ exclAtom =
      &exclAtomHeap[(icell - cellStart) * 32 * maxNumExcl];

  //
  // Load exclusions for atoms in icell
  //
  int iatom = iatomStart + wid;
  int jstart_excl = 0;
  int jend_excl = -1;
  float4 xyzq_i;
  if (iatom <= iatomEnd) {
    int ig = loc2glo[iatom];
    jstart_excl = atom_excl_pos[ig];
    jend_excl = atom_excl_pos[ig + 1] - 1;
    xyzq_i = xyzq[iatomStart + wid];
  }
  float xi = xyzq_i.x;
  float yi = xyzq_i.y;
  float zi = xyzq_i.z;
  int tmp = ientry[ientry_ind].ish;
  int imz = tmp / 9;
  tmp -= imz * 9;
  int imy = tmp / 3;
  tmp -= imy * 3;
  int imx = tmp - 1;
  imy--;
  imz--;
  
  xi += imx * boxx;
  if (imx == 0) {
    yi += imy * boxy;
    zi += imz * boxz; 
  }
  else {
    yi = -yi + imy * boxy;
    zi = -zi + imz * boxz; 
  }

  int jlen_excl = jend_excl - jstart_excl + 1;
  int pos = incl_scan_shfl(jlen_excl, wid);
// Get the total number of excluded atoms by broadcasting the last value
// across all threads in the warp
  int numExclAtom = bcast_shfl(pos, warpsize - 1);
  // Get the exclusive sum position
  pos -= jlen_excl;
  // Loop through excluded atoms:
  // Find min and max indices
  // Store atom indices to exclAtom -buffer
  int minExclAtom = (1 << 30); // (= big number)
  int maxExclAtom = 0;
  int nexcl = 0;
  for (int jatom = jstart_excl; jatom <= jend_excl; jatom++) {
    int atom = glo2loc[atom_excl[jatom]];
    // Atoms that are not on this node are marked in glo2loc[] by value -1
    if (atom >= 0) {
      minExclAtom = min(minExclAtom, atom);
      maxExclAtom = max(maxExclAtom, atom);
    }
    // Store excluded atom index (atom) and atom i index
    exclAtom[pos + nexcl++] = (atom << 5) | wid;
  }
// Reduce minExclAtom and maxExclAtom across the warp
  minExclAtom = min_shfl(minExclAtom);
  maxExclAtom = max_shfl(maxExclAtom);

  int min_atomj = 1 << 30;
  int max_atomj = 0;
  int n_atomj = 0;
  int n_tile_new = 0;
  int tileStart = ientry[ientry_ind].tileStart;
  int reg_atomj;
  // Loop through tiles
  for (int tile = tileStart; tile <= ientry[ientry_ind].tileEnd; tile++) {
    int jcell = tile_indj[tile];
    // NOTE: We need jatomEnd here to compute the exclusion mask
    int jatomStart = cell_patom[jcell];
    int jatomEnd = cell_patom[jcell + 1] - 1;

    // Load j-atoms into registers (CUDA_ARCH >= 3.0) or shared memory
    // (CUDA_ARCH < 3.0)
    float4 xyzq_j;
    if (jatomStart + wid <= jatomEnd)
      xyzq_j = xyzq[jatomStart + wid];
    float xj = xyzq_j.x;
    float yj = xyzq_j.y;
    float zj = xyzq_j.z;

    bool first = true;
    for (int j = 0; j <= jatomEnd - jatomStart; j++) {
      float xt = SHFL(xj, j);
      float yt = SHFL(yj, j);
      float zt = SHFL(zj, j);

      float dx = xi - xt;
      float dy = yi - yt;
      float dz = zi - zt;

      float r2 = dx * dx + dy * dy + dz * dz;
      if (ANY((r2 < rcut2))) {
        if (first) {
          first = false;
          // ----------------------------
          // Set initial exclusion masks
          // ----------------------------
          // NOTE: In case i,j cells are less than tilesize atoms, add
          // exclusions
          int itile = tileStart + n_tile_new;
          int ni = (iatomEnd - iatomStart + 1);
          unsigned int mask =
              (jatomStart + wid <= jatomEnd) ? 0 : 0xffffffff; // j contribution
          int up = (ni >= wid) ? ni - wid : 32 + ni - wid;
          int dw = (wid >= ni) ? wid - ni : 32 + wid - ni;
          unsigned int imask = (1 << (32 - ni)) - 1;
          mask |= (imask << up) | (imask >> dw); // i contribution
          // Diagonal tile, exclude i >= j
          if (iatomStart == jatomStart) {
            mask |= (0xffffffff >> wid);
          }
          tile_excl[itile].excl[wid] = mask;
          // Re-write tile_indj with the atom index
          // NOTE: This re-write will not mess with the read above
          // because n_tile_new <= tile
          if (wid == 0) {
            tile_indj[itile] = jatomStart;
            // printf("Adding %d, ish %d, tile %d, j %d, r2 %.6f
            // %d\n", iatomStart,ientry[ientry_ind].ish,
            // tile_indj[tile],j,r2, n_tile_new + 1);
          }
          // Advance to the next tile
          n_tile_new++;
        }

        // This j-atom is within rcut of one of the i-atoms => add to
        // exclusion check list Add j-atom to the exclusion check list
        int atomj = jatomStart + j;
        min_atomj = min(min_atomj, atomj);
        max_atomj = max(max_atomj, atomj);
        if (wid == n_atomj)
          reg_atomj = (atomj << n_jlist_max_shift) | (n_tile_new - 1);
        n_atomj++;

        // Check reg_atomj[0...warpsize-1] for exclusions with any
        // of the i atoms in exclAtom[0...numExclAtom-1]
        if (n_atomj == warpsize) {
          // Check for topological exclusions
          p21flushAtomjNew(wid, min_atomj, max_atomj, n_atomj, reg_atomj,
                                  minExclAtom, maxExclAtom, numExclAtom,
                                  exclAtom, tileStart, tile_excl, tile_indj);
          min_atomj = 1 << 30;
          max_atomj = 0;
          n_atomj = 0;
        } // if (natomj == warpsize)

      } // if (__any((r2 < rcut2)))
    }   // for (int j=0;j <= jatomEnd-jatomStart;j++)

    //---------------------------------------------------------------------------------------

  } // for (int jcell=ientry[ientry_ind].tileStart;jcell <=
    // ientry[ientry_ind].tileEnd;jcell++)

  if (n_atomj > 0) {
    p21flushAtomjNew(wid, min_atomj, max_atomj, n_atomj, reg_atomj,
                            minExclAtom, maxExclAtom, numExclAtom, exclAtom,
                            tileStart, tile_excl, tile_indj);
  }

  // Re-write tileEnd with possibly reduced number of tiles
  ientry[ientry_ind].tileEnd = tileStart + n_tile_new - 1;

  // Add to bucket for sorting
  // ibucket = 0 ... n_jlist_max, is in inverse order since we want largest
  // first

  if (wid == 0) {
    int ibucket = n_jlist_max - n_tile_new;
    atomicAdd(&bucketPos[ibucket], 1);
    bucketIndex[ientry_ind] = ibucket;
  } 
}
//
// Sorts ientry with bucket sort using a single thread block
// NOTE: Works for any length array but will not be optimal for very long arrays
//
__global__ void p21bucketSortShortIentryKernel(const int numBucket, int *bucketPos,
                                            int *bucketIndex,
                                            const int n_ientry,
                                            const ientry_t *ientry_in,
                                            ientry_t *ientry_out) {
  // Shared memory
  // Requires: blockDim.x*sizeof(int)
  extern __shared__ int shBucketPos[];

  // Compute position of buckets
  int pos0 = 0;
  for (int base = 0; base < numBucket; base += blockDim.x) {
    // Load bucketPos into shared memory
    shBucketPos[threadIdx.x] =
        (base + threadIdx.x < numBucket) ? bucketPos[base + threadIdx.x] : 0;
    __syncthreads();
    // Perform inclusive cumsum in shared memory
    for (int d = 1; d < blockDim.x; d *= 2) {
      int val = (threadIdx.x >= d) ? shBucketPos[threadIdx.x - d] : 0;
      __syncthreads();
      shBucketPos[threadIdx.x] += val;
      __syncthreads();
    }
    // Store bucketPos back into global memory and switch to exclusive
    // cumsum
    if (base + threadIdx.x < numBucket)
      bucketPos[base + threadIdx.x] =
          pos0 + (threadIdx.x >= 1) ? shBucketPos[threadIdx.x - 1] : 0;
    // Get end position value for this block
    pos0 = shBucketPos[blockDim.x - 1];
  }
  __syncthreads();

  // Re-order according to buckets
  for (int i = threadIdx.x; i < n_ientry; i += blockDim.x) {
    int ibucket = bucketIndex[i];
    int pos = atomicAdd(&bucketPos[ibucket], 1);
    ientry_out[pos] = ientry_in[i];
  }
}


void CudaP21NeighborListBuild::estimateIentry(
    const ZoneParam_t *__restrict__ h_ZoneParam, float rcut) {

  n_tile_est = 0;
  int ncell_i =
      h_ZoneParam[0].ncellx * h_ZoneParam[0].ncelly * h_ZoneParam[0].ncellz_max;
  // Estimate the number of neighbors in each direction for the
  // positive direction and multiply by the number of cells
  int n_neigh_ij = ((int)ceilf(rcut / h_ZoneParam[0].celldx) + 1) *
                   ((int)ceilf(rcut / h_ZoneParam[0].celldy) + 1) *
                   ((int)ceilf(rcut / h_ZoneParam[0].celldz_min) + 1) * ncell_i;
  n_tile_est += 2 * n_neigh_ij;
  // Assume every i-j tile is in a separate ientry (worst case)
  n_ientry_est = n_tile_est;
}

void CudaP21NeighborListBuild::build(
    int ncell, const int maxNumExcl,
    const ZoneParam_t *__restrict__ h_zoneParam,
    const ZoneParam_t *__restrict__ d_zoneParam, const float boxx,
    const float boxy, const float boxz, const float rcut, const float4 *xyzq,
    const int *loc2glo, const int *glo2loc, const int *atomExclPos,
    const int *atomExcl, const int4 *cell_xyz_zone, const int *col_ncellz,
    const int *col_cell, const float *cell_bz, const int *cell_patom,
    const bb_t *bb, NlistParam_t *h_NlistParam, NlistParam_t *d_NlistParam,
    cudaStream_t stream) {
  std::cout << "[p21nlbuild] build() called" << std::endl;

  // estimate ientry
  estimateIentry(h_zoneParam, rcut);

  // TODO : this memory allocation is very high, lower it

  // allocate entry_raw, tile_excl, tile_indj, exclAtomheap, bucketIndex
  reallocate<ientry_t>(&ientry_raw, &ientry_raw_len, n_ientry_est, 1.4f);
  reallocate<int>(&tile_indj, &tile_indj_len, n_tile_est, 1.4f);
  reallocate<tile_excl_t<32>>(&tile_excl, &tile_excl_len, n_tile_est, 1.4f);
  reallocate<int>(&exclAtomHeap, &exclAtomHeapLen, ncell * 32 * maxNumExcl,
                  1.4f);
  reallocate<int>(&bucketIndex, &bucketIndexLen, n_ientry_est, 1.4f);

  // Clear bucketPos
  clear_gpu_array<int>(bucketPos, n_jlist_max + 1, stream);
  //
  std::cout << "n_ientry_est : " << n_ientry_est << "\n";
  std::cout << "ientry_raw_len : " << ientry_raw_len << "\n";

  std::cout << "n_tile_est : " << n_tile_est << "\n";
  std::cout << "tile_indj_len : " << tile_indj_len << "\n";
  //
  const int warpSize = 32;

  int numThreads = 128;
  int numBlocks = (ncell - 1) / (numThreads / warpSize) + 1;

  int shmemSize = (numThreads / warpSize) * n_jlist_max * sizeof(int);

  shmemSize += (numThreads / warpSize) * sizeof(shVars_t);

  p21buildKernel<<<numBlocks, numThreads, shmemSize, stream>>>(
      cell_xyz_zone, col_ncellz, col_cell, cell_bz, boxx, boxy, boxz, rcut,
      rcut * rcut, bb, tile_indj, ientry_raw, d_zoneParam, d_NlistParam);
  cudaCheck(cudaGetLastError());

  std::cout << "Neighborlist built\n";
  copy_DtoH<NlistParam_t>(d_NlistParam, h_NlistParam, 1, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  n_ientry = h_NlistParam->n_ientry;
  n_tile = h_NlistParam->n_tile;

  std::cout << "n_ientry : " << n_ientry << "\n";
  std::cout << "n_tile : " << n_tile << "\n";

  if (n_tile > n_tile_est) {
    std::cout
        << "CudaP21NeighborListBuild::build, Limit blown: n_tile > n_tile_est"
        << std::endl;
    exit(1);
  } else if (n_tile > n_tile_est / 2) {
    std::cout << "CudaNeighborListBuild::build, Limit should have blown: "
                 "n_tile > n_tile_est/2"
              << std::endl;
  }
  if (n_ientry > n_ientry_est) {
    std::cout << "CudaP21NeighborListBuild::build, Limit blown: n_ientry > "
                 "n_ientry_est"
              << std::endl;
    exit(1);
  }

  //
  std::ofstream ientry_file("ientry_raw.txt", std::ofstream::out);
  ientry_t *h_ientry = new ientry_t[n_ientry];
  copy_DtoH_sync<ientry_t>(ientry_raw, h_ientry, n_ientry);
  int *h_tile_indj = new int[tile_indj_len];
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, tile_indj_len);
  for (int i = 0; i < n_ientry; ++i) {
    auto element = h_ientry[i];
    for (int jtile = element.tileStart; jtile <= element.tileEnd; jtile++) {
      ientry_file << element.iatomStart << " "
                  << element.tileEnd - element.tileStart << " " << element.ish
                  << " " << h_tile_indj[jtile] << "\n";
    }
  }
  delete[] h_ientry;
  delete[] h_tile_indj;
  ientry_file.close();
  //
  // build exclusion kernel
  numBlocks = (n_ientry - 1) / (numThreads / warpSize) + 1;
  int shmem_size = 0;
  int cellStart = 0;
  p21buildExclKernel<<<numBlocks, numThreads, shmem_size, stream>>>(
      maxNumExcl, cellStart, cell_patom, loc2glo, glo2loc, atomExclPos,
      atomExcl, xyzq, boxx, boxy, boxz, rcut * rcut, exclAtomHeap, tile_indj,
      tile_excl, ientry_raw, d_NlistParam, bucketPos, bucketIndex);
  cudaCheck(cudaGetLastError());

  // bucket sort short ientry kernel

  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.4f);
  numThreads = get_max_nthread();
  numBlocks = 1;
  shmemSize = numThreads * sizeof(int);
  p21bucketSortShortIentryKernel<<<numBlocks, numThreads, shmemSize, stream>>>(
      n_jlist_max + 1, bucketPos, bucketIndex, n_ientry, ientry_raw, ientry);
  cudaCheck(cudaGetLastError());

  cudaDeviceSynchronize();

}
