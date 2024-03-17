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
#include "CudaNeighborListBuild.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>

// IF defined, uses strict (Factor = 1.0f) memory reallocation. Used for
// debuggin memory problems.
// #define STRICT_MEMORY_REALLOC

// static const int numNlistParam=2;
// static __device__ NeighborListParam_t d_NlistParam[numNlistParam];

// static __device__ ZoneParam_t d_ZoneParam[maxNumZone];

//
// Calculates overlap between volumes
//
double calc_volume_overlap(double Ax0, double Ay0, double Az0, double Ax1,
                           double Ay1, double Az1, double rcut, double Bx0,
                           double By0, double Bz0, double Bx1, double By1,
                           double Bz1, double &dx, double &dy, double &dz) {
  double x0 = Ax0 - rcut;
  double y0 = Ay0 - rcut;
  double z0 = Az0 - rcut;
  double x1 = Ax1 + rcut;
  double y1 = Ay1 + rcut;
  double z1 = Az1 + rcut;

  dx = min(x1, Bx1) - max(x0, Bx0);
  dy = min(y1, By1) - max(y0, By0);
  dz = min(z1, Bz1) - max(z0, Bz0);
  dx = (dx > 0.0) ? dx : 0.0;
  dy = (dy > 0.0) ? dy : 0.0;
  dz = (dz > 0.0) ? dz : 0.0;

  return dx * dy * dz;
}

static int BitCount(unsigned int u) {
  unsigned int uCount;

  uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
  return ((uCount + (uCount >> 3)) & 030707070707) % 63;
}

/*
static int BitCount_ref(unsigned int u) {
  unsigned int x = u;
  int res = 0;
  while (x != 0) {
    res += (x & 1);
    x >>= 1;
  }
  return res;
}
*/

//
// The entire warp enters here
// If IvsI = true, search within I zone
//
template <bool IvsI>
__device__ void get_cell_bounds_z(const int icell, const int ncell,
                                  const float minx, const float x0,
                                  const float x1, const float *__restrict__ bx,
                                  const float rcut, int &jcell0, int &jcell1) {
  int jcell_start_left, jcell_start_right;

  if (IvsI) {
    // Search within a single zone (I)
    if (icell < 0) {
      // This is one of the image cells on the left =>
      // set the left cell boundary (jcell0) to 1 and start looking for
      // the right boundary from 1
      jcell_start_left =
          -1; // with this value, we don't look for cells on the left
      jcell_start_right = 0; // start looking for cells at right from 0
      jcell0 = 0;            // left boundary set to minimum value
      jcell1 = -1;           // set to "no cells" value
    } else if (icell >= ncell) {
      // This is one of the image cells on the right =>
      // set the right cell boundary (icell1) to ncell and start looking
      // for the left boundary from ncell
      jcell_start_left =
          ncell - 1; // start looking for cells at left from ncell
      jcell_start_right =
          ncell;      // with this value, we don't look for cells on the right
      jcell0 = ncell; // set to "no cells" value
      jcell1 = ncell - 1; // right boundary set to maximum value
    } else {
      jcell_start_left = icell - 1;
      jcell_start_right = icell + 1;
      jcell0 = icell;
      jcell1 = icell;
    }
  } else {
    // Search between two different zones
    if (bx[0] >= x1 || (bx[0] < x1 && bx[0] > x0)) {
      // j-zone is to the right of i-zone
      // => no left search, start right search from 0
      jcell_start_left = -1;
      jcell_start_right = 0;
      jcell0 = 0;
      jcell1 = -1;
    } else if (bx[ncell] <= x0 || (bx[ncell] > x0 && bx[ncell] < x1)) {
      // j-zone is to the left of i-zone
      // => no right search, start left search from ncell
      jcell_start_left = ncell - 1;
      jcell_start_right = ncell;
      jcell0 = ncell;
      jcell1 = ncell - 1;
    } else {
      // i-zone is between j-zones
      // => safe choice is to search the entire range
      jcell_start_left = ncell - 1;
      jcell_start_right = 0;
      jcell0 = ncell - 1;
      jcell1 = 0;
    }
  }

  //
  // Check cells at left, stop once the distance to the cell right boundary
  // is greater than the cutoff.
  //
  // Cell right boundary is at bx[i]
  //
  for (int j = jcell_start_left; j >= 0; j--) {
    float d = x0 - bx[j];
    if (d > rcut)
      break;
    jcell0 = j;
  }

  //
  // Check cells at right, stop once the distance to the cell left boundary
  // is greater than the cutoff.
  //
  // Cell left boundary is at bx[i-1]
  //
  for (int j = jcell_start_right; j < ncell; j++) {
    float bx_j = (j > 0) ? bx[j - 1] : minx;
    float d = bx_j - x1;
    if (d > rcut)
      break;
    jcell1 = j;
  }

  // Cell bounds are jcell0:jcell1
}

//
// The entire warp enters here
// If IvsI = true, search within I zone
//
template <bool IvsI>
__device__ void get_cell_bounds_xy(const int ncell, const float minx,
                                   const float x0, const float x1,
                                   const float inv_dx, const float rcut,
                                   int &jcell0, int &jcell1) {
  if (IvsI) {
    // Search within a single zone (I)

    //
    // Check cells at left, stop once the distance to the cell right
    // boundary is greater than the cutoff.
    //
    // Cell right boundary is at bx
    // portion inside i-cell is (x0-bx)
    // => what is left of rcut on the left of i-cell is rcut-(x0-bx)
    //
    // float bx = minx + icell*dx;
    // jcell0 = max(0, icell - (int)ceilf((rcut - (x0 - bx))/dx));

    //
    // Check cells at right, stop once the distance to the cell left
    // boundary is greater than the cutoff.
    //
    // Cell left boundary is at bx
    // portion inside i-cell is (bx-x1)
    // => what is left of rcut on the right of i-cell is rcut-(bx-x1)
    //
    // bx = minx + (icell+1)*dx;
    // jcell1 = min(ncell-1, icell + (int)ceilf((rcut - (bx - x1))/dx));

    // Find first left boundary that is < x0-rcut
    jcell0 = max(0, (int)floorf((x0 - rcut - minx) * inv_dx));

    // Find first right boundary that is > x1+rcut
    jcell1 = min(ncell - 1, (int)ceilf((x1 + rcut - minx) * inv_dx) - 1);

    //
    // Take care of the boundaries:
    //
    // if (icell < 0) jcell0 = 0;
    // if (icell >= ncell) jcell1 = ncell - 1;

  } else {
    //
    // Search between zones izone and jzone
    // (x0, x1) are for izone
    // (dx, minx, ncell) are for jzone
    //

    //
    // jzone left boundaries are given by: minx + jcell*dx
    // jzone right boundaries are given by: minx + (jcell+1)*dx
    //
    // izone overlap region is: x0-rcut ... x1+rcut
    //

    // Find first left boundary that is < x0-rcut
    jcell0 = max(0, (int)floorf((x0 - rcut - minx) * inv_dx));

    // Find first right boundary that is > x1+rcut
    jcell1 = min(ncell - 1, (int)ceilf((x1 + rcut - minx) * inv_dx) - 1);
  }

  // Cell bounds are jcell0:jcell1
}

//
// Finds minimum of z0 and maximum of z1 across warp using __shfl -command
//
__forceinline__ __device__ void minmax_shfl(int z0, int z1, int &z0_min,
                                            int &z1_max) {
#if __CUDA_ARCH__ >= 300
  z0_min = z0;
  z1_max = z1;
  for (int i = 16; i >= 1; i /= 2) {
    z0_min = min(z0_min, SHFL_XOR(z0, i));
    z1_max = max(z1_max, SHFL_XOR(z1, i));
  }
#endif
}

__forceinline__ __device__ int min_shmem(int val, const int wid,
                                         volatile int *shbuf) {
  shbuf[wid] = val;
  for (int i = 16; i >= 1; i /= 2) {
    int n = shbuf[i ^ wid];
    shbuf[wid] = min(shbuf[wid], n);
  }
  return shbuf[wid];
}

__forceinline__ __device__ int max_shmem(int val, const int wid,
                                         volatile int *shbuf) {
  shbuf[wid] = val;
  for (int i = 16; i >= 1; i /= 2) {
    int n = shbuf[i ^ wid];
    shbuf[wid] = max(shbuf[wid], n);
  }
  return shbuf[wid];
}

__forceinline__ __device__ int bcast_shmem(int val, const int srclane,
                                           const int wid, volatile int *shbuf) {
  if (wid == srclane)
    shbuf[0] = val;
  return shbuf[0];
}

#if __CUDA_ARCH__ >= 300
//
// Checks that the value of integer is the warp, used for debugging
//
__forceinline__ __device__ bool check_int(int val) {
  int val0 = bcast_shfl(val, 0);
  return ALL(val == val0);
}
#endif

__forceinline__ __device__ int incl_scan_shmem(int val, const int wid,
                                               volatile int *shbuf,
                                               const int scansize = warpsize) {
  shbuf[wid] = val;
  for (int i = 1; i < scansize; i *= 2) {
    int n = (wid >= i) ? shbuf[wid - i] : 0;
    shbuf[wid] += n;
  }
  return shbuf[wid];
}

//
// Calculates the sum and places the result in all threads
//
__forceinline__ __device__ int sum_shfl(int val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 16; i >= 1; i /= 2)
    val += SHFL_XOR(val, i);
#else
  val = 0;
#endif
  return val;
}

__forceinline__ __device__ int sum_shmem(int val, const int wid,
                                         volatile int *shbuf) {
  shbuf[wid] = val;
  for (int i = 16; i >= 1; i /= 2) {
    int n = shbuf[i ^ wid];
    shbuf[wid] += n;
  }
  return val;
}

//
// Build neighborlist
// NOTE: One warp takes care of one cell
//
// Shared memory variables. These are variables whos value is the same across
// the warp
struct shVars_t {
  bb_t ibb;
};

template <int tilesize, bool IvsI>
__global__ void buildKernel(
    const bool q_p21, const int cellStart,
    const int4 *__restrict__ cell_xyz_zone, const int *__restrict__ col_ncellz,
    const int *__restrict__ col_cell, const float *__restrict__ cell_bz,
    const float boxx, const float boxy, const float boxz, const float rcut,
    const float rcut2, const bb_t *__restrict__ bb, int *__restrict__ tile_indj,
    ientry_t *__restrict__ ientry, const ZoneParam_t *__restrict__ ZoneParam,
    NlistParam_t *__restrict__ NlistParam) {
  // Shared memory
  extern __shared__ char shbuf[];

  // Warp index
  const int wid = threadIdx.x % warpsize;

  // Index of the i-cell
  const int icell =
      cellStart + (threadIdx.x + blockIdx.x * blockDim.x) / warpsize;
  if (icell - cellStart >= NlistParam->ncell)
    return;

  // Get (icellx, icelly, icellz, izone):
  int4 icell_xyz_zone = cell_xyz_zone[icell];
  int icellz = icell_xyz_zone.z;
  int izone = IvsI ? 0 : icell_xyz_zone.w;

  int n_jzone = IvsI ? 1 : ZoneParam[izone].n_int_zone;
  if (n_jzone == 0)
    return;
  /*
  ----------------------------------------------------------------
  Calculate shared memory pointers:

  Total memory requirement:
  (blockDim.x/warpsize) *
  ( (!IvsI) * n_jzone*sizeof(int2)
    + n_jlist_max * sizeof(int)
    + tilesize * sizeof(float3))

  Required space:
  shflmem:         blockDim.x*sizeof(int)  (Only for __CUDA_ARCH__ < 300)
  sh_jcellxy_min: (blockDim.x/warpsize)*n_jzone*sizeof(int2) (Only IvsI = F)
  sh_jcell: (blockDim.x/warpsize)*n_jlist_max*sizeof(int)
  sh_xyzj: (blockDim.x/warpsize)*tilesize*sizeof(float3)(__CUDA_ARCH__ < 300)
  shVars :         (blockDim.x/warpsize)*sizeof(shVars_t)

  NOTE: Each warp has its own sh_jcellxy_min[]
  ----------------------------------------------------------------
  */
  int shbuf_pos = 0;
#if __CUDA_ARCH__ < 300
  // Shuffle memory buffer
  volatile int *shflmem =
      (int *)&shbuf[(threadIdx.x / warpsize) * warpsize * sizeof(int)];
  shbuf_pos += blockDim.x * sizeof(int);
#endif

  // jcellx and jcelly minimum values
  volatile int2 *sh_jcellxy_min;
  if (!IvsI) {
    sh_jcellxy_min = (int2 *)&shbuf[shbuf_pos + (threadIdx.x / warpsize) *
                                                    n_jzone * sizeof(int2)];
    shbuf_pos += (blockDim.x / warpsize) * n_jzone * sizeof(int2);
  }

  // Temporary j-cell list. Each warp has its own jlist
  volatile int *sh_jcell =
      (int *)&shbuf[shbuf_pos +
                    (threadIdx.x / warpsize) * n_jlist_max * sizeof(int)];
  shbuf_pos += (blockDim.x / warpsize) * n_jlist_max * sizeof(int);

  // Shared memory variables
  shVars_t *shVars = (shVars_t *)&shbuf[shbuf_pos + (threadIdx.x / warpsize) *
                                                        sizeof(shVars_t)];
  shbuf_pos += (blockDim.x / warpsize) * sizeof(shVars_t);

  // Load bounding box
  if (wid == 0)
    shVars->ibb = bb[icell];

  for (int imx = NlistParam->imx_lo; imx <= NlistParam->imx_hi; imx++) {
    float imbbx0 = shVars->ibb.x + imx * boxx;
    int n_jcellx = 0;
    int jcellx_min, jcellx_max;
    if (IvsI) {
      get_cell_bounds_xy<true>(ZoneParam[0].ncellx, ZoneParam[0].min_xyz.x,
                               imbbx0 - shVars->ibb.wx, imbbx0 + shVars->ibb.wx,
                               ZoneParam[0].inv_celldx, rcut, jcellx_min,
                               jcellx_max);
      n_jcellx = max(0, jcellx_max - jcellx_min + 1);
      if (n_jcellx == 0)
        continue;
    } else {
      if (wid < n_jzone) {
        int jzone = ZoneParam[izone].int_zone[wid];
        int jcellx0_t, jcellx1_t;
        get_cell_bounds_xy<false>(
            ZoneParam[jzone].ncellx, ZoneParam[jzone].min_xyz.x,
            imbbx0 - shVars->ibb.wx, imbbx0 + shVars->ibb.wx,
            ZoneParam[jzone].inv_celldx, rcut, jcellx0_t, jcellx1_t);
        n_jcellx = max(0, jcellx1_t - jcellx0_t + 1);
        sh_jcellxy_min[wid].x = jcellx0_t;
      }
      if (ALL(n_jcellx == 0))
        continue;
    }

    for (int imy = NlistParam->imy_lo; imy <= NlistParam->imy_hi; imy++) {
      float imbby0 = shVars->ibb.y + imy * boxy;
      if (q_p21 && imx != 0)
        imbby0 = -shVars->ibb.y + imy * boxy;
      int n_jcelly = 0;
      int jcelly_min, jcelly_max;
      if (IvsI) {
        get_cell_bounds_xy<true>(
            ZoneParam[0].ncelly, ZoneParam[0].min_xyz.y,
            imbby0 - shVars->ibb.wy, imbby0 + shVars->ibb.wy,
            ZoneParam[0].inv_celldy, rcut, jcelly_min, jcelly_max);
        n_jcelly = max(0, jcelly_max - jcelly_min + 1);
        if (n_jcelly == 0)
          continue;
      } else {
        if (wid < n_jzone) {
          int jzone = ZoneParam[izone].int_zone[wid];
          int jcelly0_t, jcelly1_t;
          get_cell_bounds_xy<false>(
              ZoneParam[jzone].ncelly, ZoneParam[jzone].min_xyz.y,
              imbby0 - shVars->ibb.wy, imbby0 + shVars->ibb.wy,
              ZoneParam[jzone].inv_celldy, rcut, jcelly0_t, jcelly1_t);
          n_jcelly = max(0, jcelly1_t - jcelly0_t + 1);
          sh_jcellxy_min[wid].y = jcelly0_t;
        }
        if (ALL(n_jcelly == 0))
          continue;
      }

      for (int imz = NlistParam->imz_lo; imz <= NlistParam->imz_hi; imz++) {
        float imbbz0 = shVars->ibb.z + imz * boxz;
        if (q_p21 && imx != 0)
          imbbz0 = -shVars->ibb.z + imz * boxz;
        int ish = imx + 1 + 3 * (imy + 1 + 3 * (imz + 1));

        int jzone_counter;
        if (!IvsI)
          jzone_counter = 0;
        do {
          int n_jlist = 0;
          int n_jcellx_t = n_jcellx;
          int n_jcelly_t = n_jcelly;
          int jzone;
          if (!IvsI) {
#if __CUDA_ARCH__ >= 300
            n_jcellx_t = bcast_shfl(n_jcellx_t, jzone_counter);
            n_jcelly_t = bcast_shfl(n_jcelly_t, jzone_counter);
#else
            n_jcellx_t = bcast_shmem(n_jcellx_t, jzone_counter, wid, shflmem);
            n_jcelly_t = bcast_shmem(n_jcelly_t, jzone_counter, wid, shflmem);
#endif
            jcellx_min = sh_jcellxy_min[jzone_counter].x;
            jcelly_min = sh_jcellxy_min[jzone_counter].y;
            jzone = ZoneParam[izone].int_zone[jzone_counter];
          }
          int total_xy = n_jcellx_t * n_jcelly_t;
          if (total_xy > 0) {
            int jcellz_min = 1 << 30;
            int jcellz_max = 0;
            for (int ibase = 0; ibase < total_xy; ibase += warpsize) {
              int i = ibase + wid;
              // Find new (jcellz0_t, jcellz1_t) -range
              int jcellz0_t = 1 << 30;
              int jcellz1_t = 0;
              if (i < total_xy) {
                int jcelly = i / n_jcellx_t;
                int jcellx = i - jcelly * n_jcellx_t;
                jcellx += jcellx_min;
                jcelly += jcelly_min;
                int jcol = jcellx +
                           jcelly * ZoneParam[IvsI ? 0 : jzone].ncellx +
                           (IvsI ? 0 : ZoneParam[jzone].zone_col);
                // jcell0 = beginning of cells for column jcol
                int jcell0 = col_cell[jcol];
                if (IvsI) {
                  get_cell_bounds_z<true>(
                      icellz + imz * col_ncellz[jcol], col_ncellz[jcol],
                      ZoneParam[IvsI ? 0 : jzone].min_xyz.z,
                      imbbz0 - shVars->ibb.wz, imbbz0 + shVars->ibb.wz,
                      &cell_bz[jcell0], rcut, jcellz0_t, jcellz1_t);
                } else {
                  get_cell_bounds_z<false>(
                      icellz + imz * col_ncellz[jcol], col_ncellz[jcol],
                      ZoneParam[IvsI ? 0 : jzone].min_xyz.z,
                      imbbz0 - shVars->ibb.wz, imbbz0 + shVars->ibb.wz,
                      &cell_bz[jcell0], rcut, jcellz0_t, jcellz1_t);
                }
              } // if (i < total_xy)
              jcellz_min = min(jcellz_min, jcellz0_t);
              jcellz_max = max(jcellz_max, jcellz1_t);
            } // for (int ibase...)

// Here all threads have their own (jcellz_min, jcellz_max),
// find the minimum and maximum among all threads:
#if __CUDA_ARCH__ >= 300
            jcellz_min = min_shfl(jcellz_min);
            jcellz_max = max_shfl(jcellz_max);
#else
            jcellz_min = min_shmem(jcellz_min, wid, shflmem);
            jcellz_max = max_shmem(jcellz_max, wid, shflmem);
#endif

            int n_jcellz_max = jcellz_max - jcellz_min + 1;
            int total_xyz = total_xy * n_jcellz_max;

            if (total_xyz > 0) {
              //
              // Final loop that goes through the cells
              //
              // Cells are ordered in (y, x, z). (i.e. z first, x
              // second, y third)
              //

              for (int ibase = 0; ibase < total_xyz; ibase += warpsize) {
                int i = ibase + wid;
                int ok = 0;
                int jcell;
                if (i < total_xyz) {
                  // Calculate (jcellx, jcelly, jcellz)
                  int it = i;
                  int jcelly = it / (n_jcellx_t * n_jcellz_max);
                  it -= jcelly * (n_jcellx_t * n_jcellz_max);
                  int jcellx = it / n_jcellz_max;
                  int jcellz = it - jcellx * n_jcellz_max;
                  jcellx += jcellx_min;
                  jcelly += jcelly_min;
                  jcellz += jcellz_min;
                  // Calculate column index "jcol" and final
                  // cell index "jcell"
                  int jcol = jcellx +
                             jcelly * ZoneParam[IvsI ? 0 : jzone].ncellx +
                             (IvsI ? 0 : ZoneParam[jzone].zone_col);
                  jcell = col_cell[jcol] + jcellz;
                  // NOTE: jcellz can be out of bounds here,
                  // so we need to check
                  if (((IvsI && (icell <= jcell)) || !IvsI) && jcellz >= 0 &&
                      jcellz < col_ncellz[jcol]) {
                    // Read bounding box for j-cell
                    // jbb = bb[jcell];
                    // Calculate distance between i-cell and
                    // j-cell bounding boxes
                    float dx = max(0.0f, fabsf(imbbx0 - bb[jcell].x) -
                                             shVars->ibb.wx - bb[jcell].wx);
                    float dy = max(0.0f, fabsf(imbby0 - bb[jcell].y) -
                                             shVars->ibb.wy - bb[jcell].wy);
                    float dz = max(0.0f, fabsf(imbbz0 - bb[jcell].z) -
                                             shVars->ibb.wz - bb[jcell].wz);
                    float r2 = dx * dx + dy * dy + dz * dz;
                    ok = (r2 < rcut2);
                  }
                } // if (i < total_xyz)

                //
                // Add j-cells into temporary list (in shared
                // memory)
                //
                // First reduce to calculate position for each
                // thread in warp
                int pos = binary_excl_scan(ok, wid);
                int n_jlist_add = binary_reduce(ok);

                // Flush if the sh_jcell[] buffer would become
                // full
                if ((n_jlist + n_jlist_add) > n_jlist_max) {
                  // Write sh_jcell[] into global memory
                  int tileStart;
                  if (wid == 0)
                    tileStart = atomicAdd(&NlistParam->n_tile, n_jlist);
#if __CUDA_ARCH__ >= 300
                  tileStart = bcast_shfl(tileStart, 0);
#else
                  tileStart = bcast_shmem(tileStart, 0, wid, shflmem);
#endif
                  // temporarily store jcell here
                  for (int jj = wid; jj < n_jlist; jj += warpsize) {
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

                // Add to list
                if (ok)
                  sh_jcell[n_jlist + pos] = jcell;
                n_jlist += n_jlist_add;

              } // for (int ibase...)

              if (n_jlist > 0) {
                // Write sh_jcell[] into global memory
                int tileStart;
                if (wid == 0) {
                  tileStart = atomicAdd(&NlistParam->n_tile, n_jlist);
                }
#if __CUDA_ARCH__ >= 300
                tileStart = bcast_shfl(tileStart, 0);
#else
                tileStart = bcast_shmem(tileStart, 0, wid, shflmem);
#endif
                // temporarily store jcell here
                for (int jj = wid; jj < n_jlist; jj += warpsize) {
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
            } // if (total_xyz > 0)
          }   // if (total_xy > 0)

          if (!IvsI)
            jzone_counter++;
        } while (!IvsI && (jzone_counter < n_jzone));

        // if (wid == 0) shVars->imz++;
      } // for (int imz=imz_lo;imz <= imz_hi;imz++)
        // if (wid == 0) shVars->imy++;
    }   // for (int imy=imy_lo;imy <= imy_hi;imy++)
        // if (wid == 0) shVars->imx++;
  }     // for (int imx=imx_lo;imx <= imx_hi;imx++)
}

template <int tilesize>
__forceinline__ __device__ void
flushAtomjNew(const int wid, const int min_atomj, const int max_atomj,
              const int n_atomj, const int reg_atomj, const int minExclAtom,
              const int maxExclAtom, const int numExclAtom,
              const int *__restrict__ exclAtom, const int tileStart,
              tile_excl_t<tilesize> *__restrict__ tile_excl,
              const int *__restrict__ tile_indj) {
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

template <int tilesize>
__global__ void buildExclKernel(
    const bool q_p21, const int maxNumExcl, const int cellStart,
    const int *__restrict__ cell_patom, const int *__restrict__ loc2glo,
    const int *__restrict__ glo2loc, const int *__restrict__ atom_excl_pos,
    const int *__restrict__ atom_excl, const float4 *__restrict__ xyzq,
    const float boxx, const float boxy, const float boxz, const float rcut2,
    int *__restrict__ exclAtomHeap, int *__restrict__ tile_indj,
    tile_excl_t<tilesize> *__restrict__ tile_excl,
    ientry_t *__restrict__ ientry, const NlistParam_t *__restrict__ NlistParam,
    int *__restrict__ bucketPos, int *__restrict__ bucketIndex) {
#if __CUDA_ARCH__ < 300
  // Shared memory:
  // shflmem:         blockDim.x*sizeof(int)                           (Only
  // for
  // __CUDA_ARCH__ < 300)
  // sh_xyzj:         (blockDim.x/warpsize)*tilesize*sizeof(float3)    (Only
  // for
  // __CUDA_ARCH__ < 300)
  extern __shared__ char shbuf[];
  int shbuf_pos = 0;
  // Shuffle memory buffer
  volatile int *shflmem = (int *)&shbuf[shbuf_pos + (threadIdx.x / warpsize) *
                                                        warpsize * sizeof(int)];
  shbuf_pos += blockDim.x * sizeof(int);
  volatile float3 *sh_xyzj =
      (float3 *)&shbuf[shbuf_pos +
                       (threadIdx.x / warpsize) * tilesize * sizeof(float3)];
  shbuf_pos += (blockDim.x / warpsize) * tilesize * sizeof(float3);
#endif

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
      &exclAtomHeap[(icell - cellStart) * tilesize * maxNumExcl];
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

  if (q_p21 && imx != 0) {
    yi = -yi + imy * boxy;
    zi = -zi + imz * boxz;
  } else {
    yi += imy * boxy;
    zi += imz * boxz;
  }
  int jlen_excl = jend_excl - jstart_excl + 1;
#if __CUDA_ARCH__ >= 300
  int pos = incl_scan_shfl(jlen_excl, wid);
#else
  int pos = incl_scan_shmem(jlen_excl, wid, shflmem);
#endif
// Get the total number of excluded atoms by broadcasting the last value
// across all threads in the warp
#if __CUDA_ARCH__ >= 300
  int numExclAtom = bcast_shfl(pos, warpsize - 1);
#else
  int numExclAtom = bcast_shmem(pos, warpsize - 1, wid, shflmem);
#endif
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
#if __CUDA_ARCH__ >= 300
  minExclAtom = min_shfl(minExclAtom);
  maxExclAtom = max_shfl(maxExclAtom);
#else
  minExclAtom = min_shmem(minExclAtom, wid, shflmem);
  maxExclAtom = max_shmem(maxExclAtom, wid, shflmem);
#endif

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
#if __CUDA_ARCH__ >= 300
    float xj = xyzq_j.x;
    float yj = xyzq_j.y;
    float zj = xyzq_j.z;
#else
    sh_xyzj[wid].x = xyzq_j.x;
    sh_xyzj[wid].y = xyzq_j.y;
    sh_xyzj[wid].z = xyzq_j.z;
#endif

    bool first = true;
    for (int j = 0; j <= jatomEnd - jatomStart; j++) {
#if __CUDA_ARCH__ >= 300
      float xt = SHFL(xj, j);
      float yt = SHFL(yj, j);
      float zt = SHFL(zj, j);
#else
      float xt = sh_xyzj[j].x;
      float yt = sh_xyzj[j].y;
      float zt = sh_xyzj[j].z;
#endif
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
          int up = (ni >= wid) ? ni - wid : tilesize + ni - wid;
          int dw = (wid >= ni) ? wid - ni : tilesize + wid - ni;
          unsigned int imask = (1 << (tilesize - ni)) - 1;
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
          flushAtomjNew<tilesize>(wid, min_atomj, max_atomj, n_atomj, reg_atomj,
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
    flushAtomjNew<tilesize>(wid, min_atomj, max_atomj, n_atomj, reg_atomj,
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

//----------------------------------------------------------------------------------------
//
// Builds tilex exclusion mask from ijlist[] based on distance and index
// Builds exclusion mask based on atom-atom distance and index (i >= j excluded)
//
// Uses 32 threads to calculate the distances for a single ijlist -entry.
//
const int nwarp_build_excl_dist = 8;

template <int tilesize>
__global__ void
build_excl_kernel(const unsigned int base_tid, const int n_ijlist,
                  const int3 *ijlist, const int *cell_patom, const float4 *xyzq,
                  int *tile_indj, tile_excl_t<tilesize> *tile_excl,
                  const float boxx, const float boxy, const float boxz,
                  const float rcut2) {
  const int num_thread_per_excl = (32 / (num_excl<tilesize>::val));

  // Global thread index
  const unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x + base_tid;
  // Global warp index
  const unsigned int wid = gtid / warpsize;
  // Local thread index (0...warpsize-1)
  const unsigned int tid = gtid % warpsize;
  // local thread index (0...tilesize-1)
  const unsigned int stid = gtid % tilesize;

  // Shared memory
  extern __shared__ char shmem[];
  volatile float3 *sh_xyzi =
      (float3 *)&shmem[0]; // nwarp_build_excl_dist*tilesize
  unsigned int *sh_excl =
      (unsigned int *)&sh_xyzi[nwarp_build_excl_dist * tilesize];

  //  __shared__ float3 sh_xyzi[nwarp_build_excl_dist*tilesize];
  // #if (tilesize == 16)
  //  __shared__ unsigned int sh_excl[nwarp_build_excl_dist*num_excl];
  // #endif

  if (wid >= n_ijlist)
    return;

  // Each warp computes one ijlist entry
  int3 ijlist_val = ijlist[wid];
  int icell = ijlist_val.x - 1;
  int ish = ijlist_val.y;
  int jcell = ijlist_val.z - 1;

  int istart = cell_patom[icell] - 1;
  int iend = cell_patom[icell + 1] - 2;

  int jstart = cell_patom[jcell] - 1;
  int jend = cell_patom[jcell + 1] - 2;

  const unsigned int load_ij = threadIdx.x % tilesize;
  const int sh_start = (threadIdx.x / warpsize) * tilesize;

  // Load atom i coordinates to shared memory
  // NOTE: volatile qualifier in "sh_xyzi" guarantees that values are actually
  // read/written from
  //       shared memory. Therefore, no __syncthreads() is needed.
  float4 xyzq_i;

  if (tilesize == 32 || tid < 16) {
    if (istart + load_ij <= iend) {
      xyzq_i = xyzq[istart + load_ij];
    } else {
      xyzq_i.x = -100000000.0f;
      xyzq_i.y = -100000000.0f;
      xyzq_i.z = -100000000.0f;
    }
    sh_xyzi[sh_start + load_ij].x = xyzq_i.x;
    sh_xyzi[sh_start + load_ij].y = xyzq_i.y;
    sh_xyzi[sh_start + load_ij].z = xyzq_i.z;
  }

  // Load atom j coordinates
  float xj, yj, zj;
  //  const unsigned int loadj = (tid + (tid/TILESIZE)*(TILESIZE-1)) %
  //  TILESIZE; const unsigned int loadj = threadIdx.x % TILESIZE;
  if (jstart + load_ij <= jend) {
    float4 xyzq_j = xyzq[jstart + load_ij];
    xj = xyzq_j.x;
    yj = xyzq_j.y;
    zj = xyzq_j.z;
  } else {
    xj = 100000000.0f;
    yj = 100000000.0f;
    zj = 100000000.0f;
  }

  // Calculate shift
  float shx, shy, shz;
  calc_box_shift<float>(ish, boxx, boxy, boxz, shx, shy, shz);
  xj -= shx;
  yj -= shy;
  zj -= shz;

  // Make sure shared memory has been written
  // NOTE: since we're only operating within the warp, this __syncthreads() is
  // just to make sure
  //       all values are actually written in shared memory and not kept in
  //       registers etc.
  //__syncthreads();

  int q_samecell = (icell == jcell);

  unsigned int excl = 0;
  int t;

  if (tilesize == 32) {
    for (t = 0; t < (num_excl<tilesize>::val); t++) {
      int i = ((threadIdx.x + t) % tilesize);
      int ii = sh_start + i;
      float dx = sh_xyzi[ii].x - xj;
      float dy = sh_xyzi[ii].y - yj;
      float dz = sh_xyzi[ii].z - zj;
      float r2 = dx * dx + dy * dy + dz * dz;
      excl |= ((r2 >= rcut2) | (q_samecell && (tid <= i))) << t;
    }
    tile_indj[wid] = jstart;
    tile_excl[wid].excl[stid] = excl;

  } else {
    for (t = 0; t < (num_excl<tilesize>::val); t++) {
      int load_i = (tid + t * 2 + (tid / tilesize) * (tilesize - 1)) % tilesize;
      int ii = sh_start + load_i;
      float dx = sh_xyzi[ii].x - xj;
      float dy = sh_xyzi[ii].y - yj;
      float dz = sh_xyzi[ii].z - zj;
      float r2 = dx * dx + dy * dy + dz * dz;
      excl |= ((r2 >= rcut2) | (q_samecell && (load_ij <= load_i))) << t;
    }
    // excl is a 8 bit exclusion mask.
    // The full 32 bit exclusion mask is contained in 4 threads:
    // thread 0 contains the lowest 8 bits
    // thread 1 contains the next 8 bits, etc..

    excl <<= (threadIdx.x % num_thread_per_excl) * (num_excl<tilesize>::val);

    // Combine excl using shared memory
    const unsigned int sh_excl_ind =
        (threadIdx.x / warpsize) * (num_excl<tilesize>::val) +
        (threadIdx.x % warpsize) / num_thread_per_excl;

    sh_excl[sh_excl_ind] = 0;
    __syncthreads();

    atomicOr(&sh_excl[sh_excl_ind], excl);

    // Make sure shared memory is written
    __syncthreads();

    // index to tile_excl.excl[] (0...7)
    const unsigned int excl_ind =
        (threadIdx.x % warpsize) / num_thread_per_excl;

    tile_indj[wid] = jstart;

    if ((threadIdx.x % num_thread_per_excl) == 0) {
      tile_excl[wid].excl[excl_ind] = sh_excl[sh_excl_ind];
    }
  }
}

//----------------------------------------------------------------------------------------
//
// Combines tile_excl_top on GPU
// One thread takes care of one integer in the exclusion mask, therefore:
//
// 32x32 tile, 32 integers per tile
// 16x16 tile, 8 integers per tile
//
template <int tilesize>
__global__ void add_tile_top_kernel(const int ntile_top,
                                    const int *tile_ind_top,
                                    const tile_excl_t<tilesize> *tile_excl_top,
                                    tile_excl_t<tilesize> *tile_excl) {
  // Global thread index
  const unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
  // Index to tile_ind_top[]
  const unsigned int i = gtid / (num_excl<tilesize>::val);
  // Index to exclusion mask
  const unsigned int ix = gtid % (num_excl<tilesize>::val);

  if (i < ntile_top) {
    int ind = tile_ind_top[i];
    tile_excl[ind].excl[ix] |= tile_excl_top[i].excl[ix];
  }
}

//
// Sorts ientry with bucket sort using a single thread block
// NOTE: Works for any length array but will not be optimal for very long arrays
//
__global__ void bucketSortShortIentryKernel(const int numBucket, int *bucketPos,
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

//
// Sort ientry using bitonic sort. Keeping it simple.
// Real sorting algorithm should be coded soon.. since this won't work when we
// blow the
// shared memory limit
//
__global__ void sort_ientry_kernel(const int n_ientry,
                                   const ientry_t *__restrict__ ientry_in,
                                   ientry_t *__restrict__ ientry_out) {
  // Shared memory
  // Requires: n_ientry*sizeof(int2)
  extern __shared__ int2 sh_keyval[];

  // Read keys and values into shared memory
  for (int i = threadIdx.x; i < n_ientry; i += blockDim.x) {
    int2 keyval;
    // Note the minus sign here because we want to order these largest first
    keyval.x = -(ientry_in[i].tileEnd - ientry_in[i].tileStart + 1);
    keyval.y = i;
    sh_keyval[i] = keyval;
  }
  __syncthreads();

  for (int k = 2; k < 2 * n_ientry; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      for (int i = threadIdx.x; i < n_ientry; i += blockDim.x) {
        int ixj = i ^ j;
        if (ixj > i && ixj < n_ientry) {
          // asc = true for ascending order
          bool asc = ((i & k) == 0);
          for (int kk = k * 2; kk < 2 * n_ientry; kk *= 2)
            asc = ((i & kk) == 0 ? !asc : asc);

          // Read data
          int2 keyval1 = sh_keyval[i];
          int2 keyval2 = sh_keyval[ixj];

          int lo_key = asc ? keyval1.x : keyval2.x;
          int hi_key = asc ? keyval2.x : keyval1.x;

          if (lo_key > hi_key) {
            // keys are in wrong order => exchange
            sh_keyval[i] = keyval2;
            sh_keyval[ixj] = keyval1;
          }

          // if ((i&k)==0 && get(i)>get(ixj)) exchange(i,ixj);
          // if ((i&k)!=0 && get(i)<get(ixj)) exchange(i,ixj);
        }
      }
      __syncthreads();
    }
  }

  for (int i = threadIdx.x; i < n_ientry; i += blockDim.x) {
    int pos = sh_keyval[i].y;
    ientry_out[i] = ientry_in[pos];
  }
}

// ########################################################################################
// ########################################################################################
// ########################################################################################

//
// Class creator
//
template <int tilesize>
CudaNeighborListBuild<tilesize>::CudaNeighborListBuild(const int n_int_zone_max,
                                                       const int izoneStart,
                                                       const int izoneEnd,
                                                       const bool q_p21_)
    : n_int_zone_max(n_int_zone_max), izoneStart(izoneStart),
      izoneEnd(izoneEnd), q_p21(q_p21_) {
  this->init();
}

template <int tilesize>
CudaNeighborListBuild<tilesize>::CudaNeighborListBuild(const int n_int_zone_max,
                                                       const int izoneStart,
                                                       const int izoneEnd,
                                                       const char *filename)
    : n_int_zone_max(n_int_zone_max), izoneStart(izoneStart),
      izoneEnd(izoneEnd) {
  this->init();
  load(filename);
}

//
// Class destructor
//
template <int tilesize>
CudaNeighborListBuild<tilesize>::~CudaNeighborListBuild() {
  if (tile_excl != NULL)
    deallocate<tile_excl_t<tilesize>>(&tile_excl);
  if (ientry_raw != NULL)
    deallocate<ientry_t>(&ientry_raw);
  if (ientry != NULL)
    deallocate<ientry_t>(&ientry);
  if (tile_indj != NULL)
    deallocate<int>(&tile_indj);
  if (exclAtomHeap != NULL)
    deallocate<int>(&exclAtomHeap);
  deallocate<int>(&bucketPos);
  if (bucketIndex != NULL)
    deallocate<int>(&bucketIndex);
#ifdef USE_SPARSE
  // Sparse
  if (pairs != NULL)
    deallocate<pairs_t<tilesize>>(&pairs);
  if (ientry_sparse != NULL)
    deallocate<ientry_t>(&ientry_sparse);
  if (tile_indj_sparse != NULL)
    deallocate<int>(&tile_indj_sparse);
#endif
}

template <int tilesize> void CudaNeighborListBuild<tilesize>::init() {
  n_ientry = 0;
  n_tile = 0;

  tile_excl = NULL;
  tile_excl_len = 0;

  ientry_raw = NULL;
  ientry_raw_len = 0;

  ientry = NULL;
  ientry_len = 0;

  tile_indj = NULL;
  tile_indj_len = 0;

  exclAtomHeapLen = 0;
  exclAtomHeap = NULL;

  allocate<int>(&bucketPos, n_jlist_max + 1);

  bucketIndexLen = 0;
  bucketIndex = NULL;

#ifdef USE_SPARSE
  // Sparse
  n_ientry_sparse = 0;
  n_tile_sparse = 0;

  pairs_len = 0;
  pairs = NULL;

  ientry_sparse_len = 0;
  ientry_sparse = NULL;

  tile_indj_sparse_len = 0;
  tile_indj_sparse = NULL;
#endif

  test = false;
  // test = true;
}

/*
//
// Copies h_NlistParam (CPU) -> d_NlistParam (GPU)
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::set_NlistParam(cudaStream_t stream) {
  cudaCheck(cudaMemcpyToSymbolAsync(d_NlistParam, h_NlistParam,
sizeof(NeighborListParam_t),
                                    0, cudaMemcpyHostToDevice, stream));
}

//
// Copies d_NlistParam (GPU) -> h_NlistParam (CPU)
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::get_NlistParam() {
  cudaCheck(cudaMemcpyFromSymbol(h_NlistParam, d_NlistParam,
sizeof(NeighborListParam_t),
                                 0, cudaMemcpyDeviceToHost));
}

//
// Copies h_ZoneParam (CPU) -> d_ZoneParam (GPU)
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::setZoneParam(cudaStream_t stream) {
  cudaCheck(cudaMemcpyToSymbolAsync(&d_ZoneParam[izoneStart], h_ZoneParam,
sizeof(ZoneParam_t),
                                    0, cudaMemcpyHostToDevice, stream));
}

//
// Copies d_ZoneParam (GPU) -> h_ZoneParam (CPU)
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::getZoneParam() {
  cudaCheck(cudaMemcpyFromSymbol(h_ZoneParam, d_ZoneParam, sizeof(ZoneParam_t),
                                 0, cudaMemcpyDeviceToHost));
}
*/

/*
//
// Resets n_tile and n_ientry variables for build() -call
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::reset() {
  get_NlistParam();
  cudaCheck(cudaDeviceSynchronize());
  h_NlistParam->n_tile = 0;
  h_NlistParam->n_ientry = 0;
  set_NlistParam(0);
  cudaCheck(cudaDeviceSynchronize());
}
*/

//
// Returns an estimate for the number of tiles
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::calc_tile_ientry_est(
    const ZoneParam_t *h_ZoneParam, const float rcut) {
  n_tile_est = 0;
  // Loop over all zone-zone interactions
  for (int izone = izoneStart; izone <= izoneEnd; izone++) {
    for (int j = 0; j < h_ZoneParam[izone].n_int_zone; j++) {
      int jzone = h_ZoneParam[izone].int_zone[j];
      if (izone != jzone) {
        // Calculate the amount of volume overlap on zone j
        double dx_j, dy_j, dz_j;
        calc_volume_overlap(
            h_ZoneParam[izone].min_xyz.x, h_ZoneParam[izone].min_xyz.y,
            h_ZoneParam[izone].min_xyz.z, h_ZoneParam[izone].max_xyz.x,
            h_ZoneParam[izone].max_xyz.y, h_ZoneParam[izone].max_xyz.z, rcut,
            h_ZoneParam[jzone].min_xyz.x, h_ZoneParam[jzone].min_xyz.y,
            h_ZoneParam[jzone].min_xyz.z, h_ZoneParam[jzone].max_xyz.x,
            h_ZoneParam[jzone].max_xyz.y, h_ZoneParam[jzone].max_xyz.z, dx_j,
            dy_j, dz_j);
        // Calculate the amount of volume overlap on zone i
        double dx_i, dy_i, dz_i;
        calc_volume_overlap(
            h_ZoneParam[jzone].min_xyz.x, h_ZoneParam[jzone].min_xyz.y,
            h_ZoneParam[jzone].min_xyz.z, h_ZoneParam[jzone].max_xyz.x,
            h_ZoneParam[jzone].max_xyz.y, h_ZoneParam[jzone].max_xyz.z, rcut,
            h_ZoneParam[izone].min_xyz.x, h_ZoneParam[izone].min_xyz.y,
            h_ZoneParam[izone].min_xyz.z, h_ZoneParam[izone].max_xyz.x,
            h_ZoneParam[izone].max_xyz.y, h_ZoneParam[izone].max_xyz.z, dx_i,
            dy_i, dz_i);
        // Number of cells in each direction that are needed to fill the
        // overlap volume
        int ncellx_j = (int)ceil(dx_j / h_ZoneParam[jzone].celldx);
        int ncelly_j = (int)ceil(dy_j / h_ZoneParam[jzone].celldy);
        int ncellz_j = (int)ceil(dz_j / h_ZoneParam[jzone].celldz_min);
        int ncell_j = ncellx_j * ncelly_j * ncellz_j;
        int ncellx_i = (int)ceil(dx_i / h_ZoneParam[izone].celldx);
        int ncelly_i = (int)ceil(dy_i / h_ZoneParam[izone].celldy);
        int ncellz_i = (int)ceil(dz_i / h_ZoneParam[izone].celldz_min);
        int ncell_i = ncellx_i * ncelly_i * ncellz_i;
        n_tile_est += 2 * ncell_j * ncell_i;
      } else {
        int ncell_i = h_ZoneParam[izone].ncellx * h_ZoneParam[izone].ncelly *
                      h_ZoneParam[izone].ncellz_max;
        // Estimate the number of neighbors in each direction for the
        // positive direction and multiply by the number of cells
        int n_neigh_ij =
            ((int)ceilf(rcut / h_ZoneParam[izone].celldx) + 1) *
            ((int)ceilf(rcut / h_ZoneParam[izone].celldy) + 1) *
            ((int)ceilf(rcut / h_ZoneParam[izone].celldz_min) + 1) * ncell_i;
        n_tile_est += 2 * n_neigh_ij;
      }
    }
  }

  // Assume every i-j tile is in a separate ientry (worst case)
  n_ientry_est = n_tile_est;
}

//
// Sets ientry from host memory array
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::set_ientry(int n_ientry,
                                                 ientry_t *h_ientry,
                                                 cudaStream_t stream) {
  this->n_ientry = n_ientry;

// Allocate & reallocate d_ientry
#ifdef STRICT_MEMORY_REALLOC
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.0f);
#else
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.4f);
#endif

  // Copy to device
  copy_HtoD<ientry_t>(h_ientry, ientry, n_ientry, stream);
}

//
// Builds neighborlist
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::build(
    const int ncell, const int cellStart, const int maxNumExcl,
    const ZoneParam_t *h_ZoneParam, const ZoneParam_t *d_ZoneParam,
    const float boxx, const float boxy, const float boxz, const float rcut,
    const float4 *xyzq, const int *loc2glo, const int *glo2loc,
    const int *atomExclPos, const int *atomExcl, const int4 *cell_xyz_zone,
    const int *col_ncellz, const int *col_cell, const float *cell_bz,
    const int *cell_patom, const bb_t *bb, NlistParam_t *h_NlistParam,
    NlistParam_t *d_NlistParam, cudaStream_t stream) {
  int nthread, nblock, shmem_size;

  calc_tile_ientry_est(h_ZoneParam, rcut);
  // std::cout << "n_ientry_est = " << n_ientry_est << " n_tile_est = " <<
  // n_tile_est << std::endl;

  if (test) {
    std::cout << "ncell = " << ncell << " n_tile_est = " << n_tile_est
              << std::endl;
    for (int izone = izoneStart; izone <= izoneEnd; izone++) {
      std::cout << izone << ": " << h_ZoneParam[izone].ncellx << " "
                << h_ZoneParam[izone].ncelly << " "
                << h_ZoneParam[izone].ncellz_max << std::endl;
    }
  }

#ifdef STRICT_MEMORY_REALLOC
  reallocate<ientry_t>(&ientry_raw, &ientry_raw_len, n_ientry_est, 1.0f);
  reallocate<tile_excl_t<tilesize>>(&tile_excl, &tile_excl_len, n_tile_est,
                                    1.0f);
  reallocate<int>(&tile_indj, &tile_indj_len, n_tile_est, 1.0f);
  reallocate<int>(&exclAtomHeap, &exclAtomHeapLen,
                  ncell * tilesize * maxNumExcl, 1.0f);
  reallocate<int>(&bucketIndex, &bucketIndexLen, n_ientry_est, 1.0f);
#else
  reallocate<ientry_t>(&ientry_raw, &ientry_raw_len, n_ientry_est, 1.4f);
  reallocate<tile_excl_t<tilesize>>(&tile_excl, &tile_excl_len, n_tile_est,
                                    1.4f);
  reallocate<int>(&tile_indj, &tile_indj_len, n_tile_est, 1.4f);
  reallocate<int>(&exclAtomHeap, &exclAtomHeapLen,
                  ncell * tilesize * maxNumExcl, 1.4f);
  reallocate<int>(&bucketIndex, &bucketIndexLen, n_ientry_est, 1.4f);
#endif

  // Clear bucketPos
  clear_gpu_array<int>(bucketPos, n_jlist_max + 1, stream);

  bool IvsI = (izoneStart == 0 && izoneEnd == 0);

  nthread = 128;
  nblock = (ncell - 1) / (nthread / warpsize) + 1;
  shmem_size = (nthread / warpsize) * n_jlist_max * sizeof(int); // sh_jlist[]
  if (get_cuda_arch() < 300) {
    shmem_size += nthread * sizeof(int); // shflmem[]
  }
  // For !IvsI, shmem_size += (nthread/warpsize)*n_int_zone_max*sizeof(int2)
  if (!IvsI)
    shmem_size += (nthread / warpsize) * n_int_zone_max *
                  sizeof(int2);                          // sh_jcellxy_min[]
  shmem_size += (nthread / warpsize) * sizeof(shVars_t); // shVars

  if (IvsI) {
    buildKernel<tilesize, true><<<nblock, nthread, shmem_size, stream>>>(
        q_p21, cellStart, cell_xyz_zone, col_ncellz, col_cell, cell_bz, boxx,
        boxy, boxz, rcut, rcut * rcut, bb, tile_indj, ientry_raw, d_ZoneParam,
        d_NlistParam);
    cudaCheck(cudaGetLastError());
  } else {
    buildKernel<tilesize, false><<<nblock, nthread, shmem_size, stream>>>(
        q_p21, cellStart, cell_xyz_zone, col_ncellz, col_cell, cell_bz, boxx,
        boxy, boxz, rcut, rcut * rcut, bb, tile_indj, ientry_raw, d_ZoneParam,
        d_NlistParam);
    cudaCheck(cudaGetLastError());
  }

  // Get variables (n_ientry & n_tile) and check for blown arrays
  copy_DtoH<NlistParam_t>(d_NlistParam, h_NlistParam, 1, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  n_ientry = h_NlistParam->n_ientry;
  n_tile = h_NlistParam->n_tile;

  // std::cout << n_tile << " " << n_tile_est << "\n";
  if (n_tile > n_tile_est) {
    std::cout
        << "CudaNeighborListBuild::build, Limit blown: n_tile > n_tile_est"
        << std::endl;
    exit(1);
  } else if (n_tile > n_tile_est / 2) {
    // std::cout << "CudaNeighborListBuild::build, Limit should have blown: "
    //              "n_tile > n_tile_est/2"
    //           << std::endl;
  }
  if (n_ientry > n_ientry_est) {
    std::cout << "CudaNeighborListBuild::build, Limit blown: n_ientry > "
                 "n_ientry_est"
              << std::endl;
    exit(1);
  }
  // std::cout << "n_ientry : " << n_ientry << "\n";

  /*
  std::ofstream ientry_file("ientry_raw.txt", std::ofstream::out);
  ientry_t *h_ientry = new ientry_t[n_ientry];
  copy_DtoH_sync<ientry_t>(ientry_raw, h_ientry, n_ientry);
  int *h_tile_indj = new int[tile_indj_len];
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, tile_indj_len);
  for (int i=0; i < n_ientry; ++i){
      auto element = h_ientry[i];
      for (int jtile = element.tileStart; jtile <=element.tileEnd; jtile++){
        ientry_file << element.iatomStart << " "<< element.tileEnd -
  element.tileStart << " " << element.ish << " "
  <<h_tile_indj[jtile]  << "\n";
      }
  }
  delete[] h_ientry;
  delete[] h_tile_indj;
  ientry_file.close();
  */
  // Build exclusions
  nblock = (n_ientry - 1) / (nthread / warpsize) + 1;
  shmem_size = 0;
  if (get_cuda_arch() < 300) {
    shmem_size += nthread * sizeof(int);                            // shflmem[]
    shmem_size += (nthread / warpsize) * tilesize * sizeof(float3); // sh_xyzj[]
  }
  buildExclKernel<tilesize><<<nblock, nthread, shmem_size, stream>>>(
      q_p21, maxNumExcl, cellStart, cell_patom, loc2glo, glo2loc, atomExclPos,
      atomExcl, xyzq, boxx, boxy, boxz, rcut * rcut, exclAtomHeap, tile_indj,
      tile_excl, ientry_raw, d_NlistParam, bucketPos, bucketIndex);
  cudaCheck(cudaGetLastError());

  /*std::ofstream ientry_file("ientry_raw.txt", std::ofstream::out);
  ientry_t *h_ientry = new ientry_t[n_ientry];
  copy_DtoH_sync<ientry_t>(ientry_raw, h_ientry, n_ientry);
  int *h_tile_indj = new int[tile_indj_len];
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, tile_indj_len);
  for (int i=0; i < n_ientry; ++i){
    //if (h_ientry[i].iatomStart == 398){
      auto element = h_ientry[i];
      for (int jtile = element.tileStart; jtile <=element.tileEnd; jtile++){
        ientry_file << element.iatomStart << " " << element.ish << " "
  <<h_tile_indj[jtile]  << "\n";
      }
      //if (element.tileEnd != element.tileStart -1)
      //  ientry_file << element.iatomStart << " " << element.ish << " "
  <<h_tile_indj[element.tileStart] << " "
  <<h_tile_indj[element.tileEnd] << "\n";
    //}
  }
  delete[] h_ientry;
  delete[] h_tile_indj;
  ientry_file.close();
  */

#ifdef STRICT_MEMORY_REALLOC
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.0f);
#else
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.4f);
#endif

  nthread = get_max_nthread();
  nblock = 1;
  shmem_size = nthread * sizeof(int);
  bucketSortShortIentryKernel<<<nblock, nthread, shmem_size, stream>>>(
      n_jlist_max + 1, bucketPos, bucketIndex, n_ientry, ientry_raw, ientry);
  cudaCheck(cudaGetLastError());

  cudaDeviceSynchronize();
  /*std::ofstream ientry_file("ientry.txt", std::ofstream::out);
  ientry_t *h_ientry = new ientry_t[n_ientry];
  copy_DtoH_sync<ientry_t>(ientry, h_ientry, n_ientry);
  int *h_tile_indj = new int[tile_indj_len];
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, tile_indj_len);
  for (int i=0; i < n_ientry; ++i){
    //if (h_ientry[i].iatomStart == 398){
      auto element = h_ientry[i];
      for (int jtile = element.tileStart; jtile <=element.tileEnd; jtile++){
        ientry_file << element.iatomStart << " "<< element.tileEnd -
  element.tileStart << " " << element.ish << " "
  <<h_tile_indj[jtile]  << "\n";
      }
      //nientry_file << element.iatomStart << " " << element.ish << " "
  <<h_tile_indj[element.tileStart] << " "
  <<h_tile_indj[element.tileEnd] << "\n";
    //}
  }
  delete[] h_ientry;
  delete[] h_tile_indj;
  ientry_file.close();
  */
  /*
  // Sort ientry according to the number of j-entries
  // NOTE: approximate sort is enough here since we don't care about the
  actual order
  //       but the sorting is done to improve non-bonded force kernel
  performance.
  nthread = min(((n_ientry-1)/warpsize + 1)*warpsize, get_max_nthread());
  nblock = 1;
  shmem_size = n_ientry*sizeof(int2);
  if (shmem_size < get_max_shmem_size()) {
    sort_ientry_kernel<<< nblock, nthread, shmem_size, stream >>>
      (n_ientry, ientry_raw, ientry);
    cudaCheck(cudaGetLastError());
  } else {
    // Sorting not possible, just do memcpy
    copy_DtoD<ientry_t>(ientry_raw, ientry, n_ientry, stream);
  }
  */
}

struct tileinfo_t {
  int excl;
  double dx, dy, dz;
  double r2;
};

template <int tilesize>
bool compare(tileinfo_t *tile1, tileinfo_t *tile2, std::vector<int2> &ijvec) {
  ijvec.clear();
  bool ok = true;
  for (int jt = 0; jt < tilesize; jt++) {
    for (int it = 0; it < tilesize; it++) {
      if (tile1[it + jt * tilesize].excl != tile2[it + jt * tilesize].excl) {
        int2 ijval;
        ijval.x = it;
        ijval.y = jt;
        ijvec.push_back(ijval);
        ok = false;
      }
    }
  }
  return ok;
}

template <int tilesize> void set_excl(tileinfo_t *tile1) {
  for (int jt = 0; jt < tilesize; jt++) {
    for (int it = 0; it < tilesize; it++) {
      tile1[it + jt * tilesize].excl = 1;
    }
  }
}

template <int tilesize> void print_excl(tileinfo_t *tile1) {
  for (int jt = 0; jt < tilesize; jt++) {
    for (int it = 0; it < tilesize; it++) {
      fprintf(stderr, "%d ", tile1[it + jt * tilesize].excl);
    }
    fprintf(stderr, "\n");
  }
}

std::ostream &operator<<(std::ostream &o, const bb_t &b) {
  o << "x,y,z= " << b.x << " " << b.y << " " << b.z << " wx,wy,wz= " << b.wx
    << " " << b.wy << " " << b.wz;
  return o;
}

//
// Test neighbor list building with a simple N^2 algorithm
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::test_build(
    const int *zone_patom, const int ncol_tot, const int ncell_tot,
    const ZoneParam_t *h_ZoneParam, const int atomExclPosLen,
    const int *atomExclPos, const int atomExclLen, const int *atomExcl,
    const double boxx, const double boxy, const double boxz, const double rcut,
    const float4 *xyzq, const int *loc2glo, const int *glo2loc,
    const int ncoord_glo, const int *cell_patom, const int *col_cell,
    const float *cell_bz, const bb_t *bb) {
  cudaCheck(cudaDeviceSynchronize());
  // get_NlistParam();

  // Build zone_cell[0...izoneEnd+1] that gives the starting cell for each
  // zone zone_cell[izoneEnd+1] gives the total number of cells
  int *h_col_cell = new int[ncol_tot];
  copy_DtoH_sync<int>(col_cell, h_col_cell, ncol_tot);
  int *zone_cell = new int[izoneEnd + 2];
  for (int izone = 0; izone <= izoneEnd; izone++) {
    // icol = column where this zone starts
    int icol = h_ZoneParam[izone].zone_col;
    zone_cell[izone] = h_col_cell[icol];
  }
  zone_cell[izoneEnd + 1] = ncell_tot;
  if (zone_cell[izoneEnd + 1] < zone_cell[izoneEnd]) {
    // std::cerr << "test_build FAILED, problem setting up zone_cell" <<
    // std::endl;
    throw std::invalid_argument(
        "test_build FAILED, problem setting up zone_cell\n");
    exit(1);
  }
  // for (int izone=0;izone <= izoneEnd+1;izone++) {
  //  std::cout << zone_cell[izone] << " " ;
  //}
  // std::cout << std::endl;
  delete[] h_col_cell;

  int *h_atom_excl_pos = new int[atomExclPosLen];
  int *h_atom_excl = new int[atomExclLen];
  copy_DtoH_sync<int>(atomExclPos, h_atom_excl_pos, atomExclPosLen);
  copy_DtoH_sync<int>(atomExcl, h_atom_excl, atomExclLen);

  // We must take into consideration all atoms up to the end of current last
  // zone
  int ncoord = zone_patom[izoneEnd + 1];

  int *h_loc2glo = new int[ncoord];
  copy_DtoH_sync<int>(loc2glo, h_loc2glo, ncoord);

  int *h_glo2loc = new int[ncoord_glo];
  bool *glo = new bool[ncoord_glo];
  for (int i = 0; i < ncoord_glo; i++)
    glo[i] = false;
  copy_DtoH_sync<int>(glo2loc, h_glo2loc, ncoord_glo);
  for (int i = 0; i < ncoord; i++) {
    int ig = h_loc2glo[i];
    if (ig < 0 || ig >= ncoord_glo) {
      std::stringstream tmpexc;
      tmpexc << "test_build FAILED, value ig=" << ig << " out of bounds"
             << std::endl;
      throw std::invalid_argument(tmpexc.str());
      exit(1);
    }
    if (glo[ig]) {
      std::stringstream tmpexc;
      tmpexc << "test_build FAILED, multiple entries in loc2glo map to same";
      tmpexc << std::endl;
      tmpexc << "i=" << i << " ig=" << ig << std::endl;
      throw std::invalid_argument(tmpexc.str());
      exit(1);
    }
    glo[ig] = true;
    if (i != h_glo2loc[ig]) {
      std::stringstream tmpexc;
      tmpexc << "test_build FAILED, loc2glo/glo2loc do not match" << std::endl;
      tmpexc << "i=" << i << " ig=" << ig << "  h_glo2loc[ig]=" << h_glo2loc[ig]
             << std::endl;
      throw std::invalid_argument(tmpexc.str());
      exit(1);
    }
  }
  delete[] h_glo2loc;

  float4 *h_xyzq = new float4[ncoord];
  copy_DtoH_sync<float4>(xyzq, h_xyzq, ncoord);

  bb_t *h_bb = new bb_t[ncell_tot];
  copy_DtoH_sync<bb_t>(bb, h_bb, ncell_tot);

  float *h_cell_bz = new float[ncell_tot];
  copy_DtoH_sync<float>(cell_bz, h_cell_bz, ncell_tot);

  double rcut2 = rcut * rcut;

  double hboxx = 0.5 * boxx;
  double hboxy = 0.5 * boxy;
  double hboxz = 0.5 * boxz;

  // Calculate number of pairs
  int npair_cpu =
      calc_cpu_pairlist<double>(zone_patom, h_xyzq, h_loc2glo, h_atom_excl_pos,
                                h_atom_excl, boxx, boxy, boxz, rcut);

  std::stringstream tmpexc;
  tmpexc << "npair_cpu=" << npair_cpu << std::endl;
  throw std::invalid_argument(tmpexc.str());

  ientry_t *h_ientry = new ientry_t[n_ientry];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[n_tile];
  int *h_tile_indj = new int[n_tile];
  int *h_cell_patom = new int[ncell_tot + 1];

  copy_DtoH_sync<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH_sync<tile_excl_t<tilesize>>(tile_excl, h_tile_excl, n_tile);
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH_sync<int>(cell_patom, h_cell_patom, ncell_tot + 1);

  // Calculate number of pairs on the GPU list
  int npair_gpu =
      calc_gpu_pairlist<double>(n_ientry, h_ientry, h_tile_indj, h_tile_excl,
                                h_xyzq, boxx, boxy, boxz, rcut);

  // std::cerr << "npair_gpu=" << npair_gpu << std::endl;

  tileinfo_t *tileinfo = new tileinfo_t[tilesize * tilesize];
  tileinfo_t *tileinfo2 = new tileinfo_t[tilesize * tilesize];
  std::vector<int2> ijvec;

  //
  // Go through all cell pairs and check that the gpu caught all of them
  //
  int npair_gpu2 = 0;
  int ncell_pair = 0;
  bool okloop = true;
  for (int izone = 0; izone <= izoneEnd; izone++) {
    for (int jzone = izoneStart; jzone <= izoneEnd; jzone++) {
      if (izone == 1 && jzone != 5)
        continue;
      if (izone == 2 && jzone != 1 && jzone != 6)
        continue;
      if (izone == 4 && jzone != 1 && jzone != 2 && jzone != 3)
        continue;

      int icell_start = zone_cell[izone];
      int icell_end = zone_cell[izone + 1];

      for (int icell = icell_start; icell < icell_end; icell++) {
        int jcell_start = zone_cell[jzone];
        int jcell_end = zone_cell[jzone + 1];
        if (izone == 0 && jzone == 0) {
          jcell_start = icell;
        }
        for (int jcell = jcell_start; jcell < jcell_end; jcell++) {
          int istart = h_cell_patom[icell];
          int iend = h_cell_patom[icell + 1] - 1;
          int jstart = h_cell_patom[jcell];
          int jend = h_cell_patom[jcell + 1] - 1;
          int npair_tile1 = 0;
          bool pair = false;
          double min_diff = 1.0e10;
          set_excl<tilesize>(tileinfo);
          for (int i = istart; i <= iend; i++) {
            double xi = h_xyzq[i].x;
            double yi = h_xyzq[i].y;
            double zi = h_xyzq[i].z;
            int ig = h_loc2glo[i];
            int excl_start = h_atom_excl_pos[ig];
            int excl_end = h_atom_excl_pos[ig + 1] - 1;
            for (int j = jstart; j <= jend; j++) {
              tileinfo_t tileinfo_val;
              tileinfo_val.excl = 1;
              if (icell != jcell || i < j) {
                double xj = h_xyzq[j].x;
                double yj = h_xyzq[j].y;
                double zj = h_xyzq[j].z;
                double dx = xi - xj;
                double dy = yi - yj;
                double dz = zi - zj;
                double shx = 0.0;
                double shy = 0.0;
                double shz = 0.0;
                if (dx > hboxx) {
                  shx = -boxx;
                } else if (dx < -hboxx) {
                  shx = boxx;
                }
                if (dy > hboxy) {
                  shy = -boxy;
                } else if (dy < -hboxy) {
                  shy = boxy;
                }
                if (dz > hboxz) {
                  shz = -boxz;
                } else if (dz < -hboxz) {
                  shz = boxz;
                }
                double xis = xi + shx;
                double yis = yi + shy;
                double zis = zi + shz;
                dx = xis - xj;
                dy = yis - yj;
                dz = zis - zj;
                double r2 = dx * dx + dy * dy + dz * dz;
                min_diff = min(min_diff, fabs(r2 - rcut2));

                int jg = h_loc2glo[j];
                bool excl_flag = false;
                for (int excl = excl_start; excl <= excl_end; excl++) {
                  if (h_atom_excl[excl] == jg) {
                    excl_flag = true;
                    break;
                  }
                }
                if (excl_flag == false) {
                  tileinfo_val.excl = 0;
                } else {
                  tileinfo_val.excl = 1;
                }
                if (r2 < rcut2 && !excl_flag) {
                  npair_gpu2++;
                  npair_tile1++;
                  pair = true;
                }
                tileinfo_val.dx = dx;
                tileinfo_val.dy = dy;
                tileinfo_val.dz = dz;
                tileinfo_val.r2 = r2;
              }
              int it = i - istart;
              int jt = j - jstart;
              tileinfo[it + jt * tilesize] = tileinfo_val;
            }
          } // for (int i=istart;i <= iend;i++)

          if (pair) {
            // Pair of cells with atoms starting at istart and
            // jstart
            bool found_this_pair = false;
            int ind, jtile;
            for (ind = 0; ind < n_ientry; ind++) {
              if (h_ientry[ind].iatomStart != istart &&
                  h_ientry[ind].iatomStart != jstart)
                continue;
              int startj = h_ientry[ind].tileStart;
              int endj = h_ientry[ind].tileEnd;
              for (jtile = startj; jtile <= endj; jtile++) {
                if ((h_ientry[ind].iatomStart == istart &&
                     h_tile_indj[jtile] == jstart) ||
                    (h_ientry[ind].iatomStart == jstart &&
                     h_tile_indj[jtile] == istart)) {
                  found_this_pair = true;
                  break;
                }
              }
              if (found_this_pair)
                break;
            }

            if (found_this_pair) {
              // Check the tile we found (ind, jtile)
              int istart0, jstart0;
              istart0 = h_ientry[ind].iatomStart;
              jstart0 = h_tile_indj[jtile];

              int ish = h_ientry[ind].ish;
              // Calculate shift
              double shx, shy, shz;
              calc_box_shift<double>(ish, boxx, boxy, boxz, shx, shy, shz);

              int npair_tile2 = 0;
              for (int i = istart0; i < istart0 + tilesize; i++) {
                double xi = (double)h_xyzq[i].x + shx;
                double yi = (double)h_xyzq[i].y + shy;
                double zi = (double)h_xyzq[i].z + shz;
                for (int j = jstart0; j < jstart0 + tilesize; j++) {
                  int bitpos =
                      ((i - istart0) - (j - jstart0) + tilesize) % tilesize;
                  unsigned int excl =
                      h_tile_excl[jtile].excl[j - jstart0] >> bitpos;
                  double xj = h_xyzq[j].x;
                  double yj = h_xyzq[j].y;
                  double zj = h_xyzq[j].z;
                  double dx = xi - xj;
                  double dy = yi - yj;
                  double dz = zi - zj;
                  double r2 = dx * dx + dy * dy + dz * dz;

                  int it, jt;
                  if (istart0 == istart) {
                    it = i - istart0;
                    jt = j - jstart0;
                  } else {
                    jt = i - istart0;
                    it = j - jstart0;
                  }

                  tileinfo_t tileinfo_val;
                  tileinfo_val.excl = (excl & 1);
                  tileinfo_val.dx = dx;
                  tileinfo_val.dy = dy;
                  tileinfo_val.dz = dz;
                  tileinfo_val.r2 = r2;
                  tileinfo2[it + jt * tilesize] = tileinfo_val;

                  if (r2 < rcut2 && !(excl & 1)) {
                    npair_tile2++;
                  }
                }
              }

              // if (abs(npair_tile1 - npair_tile2) > 0) {
              if (!compare<tilesize>(tileinfo, tileinfo2, ijvec)) {
                bool ok = true;
                for (int k = 0; k < ijvec.size(); k++) {
                  int it = ijvec.at(k).x;
                  int jt = ijvec.at(k).y;
                  tileinfo_t tileinfo_val;
                  tileinfo_val = tileinfo[it + jt * tilesize];
                  if (tileinfo_val.r2 >= rcut2) {
                    ok = false;
                    break;
                  }
                }
                if (!ok)
                  continue;

                // std::cerr << "tile pair ERROR: icell = " <<
                // icell << " jcell = " << jcell
                //	  << " npair_tile1 = " << npair_tile1 <<
                //" npair_tile2 = " << npair_tile2
                //	  << std::endl;
                // std::cerr << " istart0 = " << istart0 << "
                // jstart0 = " << jstart0
                //	  << " izone = " << izone << " jzone = "
                //<< jzone <<
                // std::endl;
                // std::cerr << " istart,iend  = " << istart <<
                // " " << iend
                //	  << " jstart,jend  = " << jstart << " "
                //<< jend
                //	  << " min_diff=" << min_diff <<
                // std::endl;

                // fprintf(stderr,"tileinfo:\n");
                // print_excl<tilesize>(tileinfo);
                // fprintf(stderr,"tileinfo2:\n");
                // print_excl<tilesize>(tileinfo2);

                for (int k = 0; k < ijvec.size(); k++) {
                  int it = ijvec.at(k).x;
                  int jt = ijvec.at(k).y;

                  tileinfo_t tileinfo_val;
                  tileinfo_val = tileinfo[it + jt * tilesize];
                  tileinfo_t tileinfo2_val;
                  tileinfo2_val = tileinfo2[it + jt * tilesize];

                  if (tileinfo_val.r2 < rcut2) {
                    fprintf(stderr, "------------------------------"
                                    "----------------\n");
                    fprintf(stderr,
                            "it,jt=%d %d dx,dy,dz=%lf %lf %lf "
                            "r2=%lf | %d %d\n",
                            it, jt, tileinfo_val.dx, tileinfo_val.dy,
                            tileinfo_val.dz, tileinfo_val.r2, tileinfo_val.excl,
                            tileinfo2_val.excl);
                    int ig = h_loc2glo[it];
                    int excl_start = h_atom_excl_pos[ig];
                    int excl_end = h_atom_excl_pos[ig + 1] - 1;
                    int jg = h_loc2glo[jt];
                    // bool excl_flag = false;
                    for (int excl = excl_start; excl <= excl_end; excl++) {
                      if (h_atom_excl[excl] == jg) {
                        fprintf(stderr, "======================"
                                        "= EXCLUSION "
                                        "FOUND! "
                                        "==================\n");
                        break;
                      }
                    }
                  }
                }
                exit(1);
              }

            } else {
              std::stringstream tmpexc;
              tmpexc << "tile pair with istart = " << istart
                     << " jstart = " << jstart << " NOT FOUND" << std::endl;
              tmpexc << "min_diff = " << min_diff
                     << " npair_tile1 = " << npair_tile1 << " ind = " << ind
                     << std::endl;
              tmpexc << h_bb[icell] << " | " << icell << std::endl;
              tmpexc << h_bb[jcell] << " | " << jcell << std::endl;
              throw std::invalid_argument(tmpexc.str());
              exit(1);
              okloop = false;
            }
          }

          if (pair)
            ncell_pair++;
        } // for (int jcell...)
      }   // for (int icell...)
    }
  }

  delete[] tileinfo;
  delete[] tileinfo2;

  delete[] h_atom_excl_pos;
  delete[] h_atom_excl;

  delete[] h_loc2glo;

  delete[] h_xyzq;
  delete[] h_ientry;
  delete[] h_tile_excl;
  delete[] h_tile_indj;
  delete[] h_cell_patom;

  delete[] h_bb;
  delete[] h_cell_bz;

  delete[] zone_cell;

  if (npair_cpu != npair_gpu || !okloop) {
    std::stringstream tmpexc;
    tmpexc << "##################################################" << std::endl;
    tmpexc << "test_build FAILED" << std::endl;
    tmpexc << "n_ientry = " << n_ientry << " n_tile = " << n_tile << std::endl;
    tmpexc << "npair_cpu = " << npair_cpu << " npair_gpu = " << npair_gpu
           << " npair_gpu2 = " << npair_gpu2 << std::endl;
    tmpexc << "##################################################" << std::endl;
    throw std::invalid_argument(tmpexc.str());
    exit(1);
  } else {
    std::cout << "test_build OK" << std::endl;
    // std::cout << "n_ientry = " << n_ientry << " n_tile = " << n_tile <<
    // std::endl;
  }

  if (!okloop)
    exit(1);
}

//
// Calculates GPU pair list
//
template <int tilesize>
template <typename T>
int CudaNeighborListBuild<tilesize>::calc_gpu_pairlist(
    const int n_ientry, const ientry_t *ientry, const int *tile_indj,
    const tile_excl_t<tilesize> *tile_excl, const float4 *xyzq,
    const double boxx, const double boxy, const double boxz,
    const double rcut) {
  T rcut2 = rcut * rcut;
  T boxxT = boxx;
  T boxyT = boxy;
  T boxzT = boxz;

  int npair = 0;
  for (int ind = 0; ind < n_ientry; ind++) {
    int istart = ientry[ind].iatomStart;
    int ish = ientry[ind].ish;
    int startj = ientry[ind].tileStart;
    int endj = ientry[ind].tileEnd;

    T shx, shy, shz;
    calc_box_shift<T>(ish, boxxT, boxyT, boxzT, shx, shy, shz);

    for (int jtile = startj; jtile <= endj; jtile++) {
      for (int i = istart; i < istart + tilesize; i++) {
        T xi = (T)xyzq[i].x + shx;
        T yi = (T)xyzq[i].y + shy;
        T zi = (T)xyzq[i].z + shz;
        int jstart = tile_indj[jtile];
        for (int j = jstart; j < jstart + tilesize; j++) {
          int bitpos = ((i - istart) - (j - jstart) + tilesize) % tilesize;
          int excl = tile_excl[jtile].excl[j - jstart] >> bitpos;
          T xj = xyzq[j].x;
          T yj = xyzq[j].y;
          T zj = xyzq[j].z;
          T dx = xi - xj;
          T dy = yi - yj;
          T dz = zi - zj;
          T r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < rcut2 && !(excl & 1))
            npair++;
        }
      }
    }
  }

  return npair;
}

//
// Calculates CPU pair list
//
template <int tilesize>
template <typename T>
int CudaNeighborListBuild<tilesize>::calc_cpu_pairlist(
    const int *zone_patom, const float4 *xyzq, const int *loc2glo,
    const int *atom_excl_pos, const int *atom_excl, const double boxx,
    const double boxy, const double boxz, const double rcut) {
  T rcut2 = rcut * rcut;
  T boxxT = boxx;
  T boxyT = boxy;
  T boxzT = boxz;
  T hboxx = 0.5 * boxx;
  T hboxy = 0.5 * boxy;
  T hboxz = 0.5 * boxz;

  int npair = 0;
  for (int izone = 0; izone <= izoneEnd; izone++) {
    for (int jzone = izoneStart; jzone <= izoneEnd; jzone++) {
      if (izone == 1 && jzone != 5)
        continue;
      if (izone == 2 && jzone != 1 && jzone != 6)
        continue;
      if (izone == 4 && jzone != 1 && jzone != 2 && jzone != 3)
        continue;

      int istart = zone_patom[izone];
      int iend = zone_patom[izone + 1] - 1;
      int jstart = zone_patom[jzone];
      int jend = zone_patom[jzone + 1] - 1;

      for (int i = istart; i <= iend; i++) {
        T xi = xyzq[i].x;
        T yi = xyzq[i].y;
        T zi = xyzq[i].z;
        int ig = loc2glo[i];
        int excl_start = atom_excl_pos[ig];
        int excl_end = atom_excl_pos[ig + 1] - 1;
        if (izone == 0 && jzone == 0)
          jstart = i + 1;
        for (int j = jstart; j <= jend; j++) {
          T xj = xyzq[j].x;
          T yj = xyzq[j].y;
          T zj = xyzq[j].z;
          T dx = xi - xj;
          T dy = yi - yj;
          T dz = zi - zj;
          if (dx > hboxx) {
            dx = (xi - boxxT) - xj;
          } else if (dx < -hboxx) {
            dx = (xi + boxxT) - xj;
          }
          if (dy > hboxy) {
            dy = (yi - boxyT) - yj;
          } else if (dy < -hboxy) {
            dy = (yi + boxyT) - yj;
          }
          if (dz > hboxz) {
            dz = (zi - boxzT) - zj;
          } else if (dz < -hboxz) {
            dz = (zi + boxzT) - zj;
          }
          T r2 = dx * dx + dy * dy + dz * dz;

          if (r2 < rcut2) {
            int jg = loc2glo[j];
            bool excl_flag = false;
            for (int excl = excl_start; excl <= excl_end; excl++) {
              if (atom_excl[excl] == jg) {
                excl_flag = true;
                break;
              }
            }
            if (excl_flag == false)
              npair++;
          }
        }
        //
      }
    }
  }

  return npair;
}

//
// Host wrapper for build_tilex_kernel
// Builds exclusion mask based on atom-atom distance and index (i >= j excluded)
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::build_excl(
    const float boxx, const float boxy, const float boxz, const float rcut,
    const int n_ijlist, const int3 *ijlist, const int *cell_patom,
    const float4 *xyzq, cudaStream_t stream) {
  if (n_ijlist == 0)
    return;

// Allocate & re-allocate (d_tile_indj, d_tile_excl)
#ifdef STRICT_MEMORY_REALLOC
  reallocate<int>(&tile_indj, &tile_indj_len, n_ijlist, 1.0f);
  reallocate<tile_excl_t<tilesize>>(&tile_excl, &tile_excl_len, n_ijlist, 1.0f);
#else
  reallocate<int>(&tile_indj, &tile_indj_len, n_ijlist, 1.2f);
  reallocate<tile_excl_t<tilesize>>(&tile_excl, &tile_excl_len, n_ijlist, 1.2f);
#endif

  float rcut2 = rcut * rcut;

  int nthread = nwarp_build_excl_dist * warpsize;
  int nblock_tot = (n_ijlist - 1) / (nthread / warpsize) + 1;
  size_t shmem_size = nwarp_build_excl_dist * tilesize * sizeof(float3);

  if (tilesize == 16) {
    shmem_size += nwarp_build_excl_dist * (num_excl<tilesize>::val) *
                  sizeof(unsigned int);
  }

  int3 max_nblock3 = get_max_nblock();
  unsigned int max_nblock = max_nblock3.x;
  unsigned int base_tid = 0;

  while (nblock_tot != 0) {
    int nblock = (nblock_tot > max_nblock) ? max_nblock : nblock_tot;
    nblock_tot -= nblock;

    build_excl_kernel<tilesize><<<nblock, nthread, shmem_size, stream>>>(
        base_tid, n_ijlist, ijlist, cell_patom, xyzq, tile_indj, tile_excl,
        boxx, boxy, boxz, rcut2);

    base_tid += nblock * nthread;

    cudaCheck(cudaGetLastError());
  }

  /*
  if (mdsim.q_test != 0) {
    test_excl_dist_index(mdsim.n_ijlist, mdsim.ijlist, mdsim.cell_patom,
                         mdsim.xyzq.xyzq, mdsim.tile_indj, mdsim.tile_excl,
                         boxx, boxy, boxz,
                         rcut2);
  }
  */
}

//
// Host wrapper for add_tile_top_kernel
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::add_tile_top(
    const int ntile_top, const int *tile_ind_top,
    const tile_excl_t<tilesize> *tile_excl_top, cudaStream_t stream) {
  int nthread = 256;
  int nblock = (ntile_top * (num_excl<tilesize>::val) - 1) / nthread + 1;

  add_tile_top_kernel<tilesize><<<nblock, nthread, 0, stream>>>(
      ntile_top, tile_ind_top, tile_excl_top, tile_excl);

  cudaCheck(cudaGetLastError());
}

#ifdef USE_SPARSE
//
// Splits neighbor list into dense and sparse parts
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::split_dense_sparse(int npair_cutoff) {
  ientry_t *h_ientry = new ientry_t[n_ientry];
  int *h_tile_indj = new int[n_tile];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[n_tile];

  ientry_t *h_ientry_dense = new ientry_t[n_ientry];
  int *h_tile_indj_dense = new int[n_tile];
  tile_excl_t<tilesize> *h_tile_excl_dense = new tile_excl_t<tilesize>[n_tile];

  ientry_t *h_ientry_sparse = new ientry_t[n_ientry];
  int *h_tile_indj_sparse = new int[n_tile];
  pairs_t<tilesize> *h_pairs = new pairs_t<tilesize>[n_tile];

  copy_DtoH_sync<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH_sync<tile_excl_t<tilesize>>(tile_excl, h_tile_excl, n_tile);

  int n_ientry_dense = 0;
  int n_tile_dense = 0;
  n_ientry_sparse = 0;
  n_tile_sparse = 0;
  for (int i = 0; i < n_ientry; i++) {
    bool sparse_i_tiles = true;
    int startj_dense = n_tile_dense;
    for (int j = h_ientry[i].tileStart; j <= h_ientry[i].tileEnd; j++) {
      int npair = 0;
      for (int k = 0; k < (num_excl<tilesize>::val); k++) {
        unsigned int n1bit = BitCount(h_tile_excl[j].excl[k]);
        npair += 32 - n1bit;
      }

      if (npair <= npair_cutoff) {
        // Sparse
        for (int k = 0; k < (num_excl<tilesize>::val); k++) {
        }
        h_tile_indj_sparse[n_tile_sparse] = h_tile_indj[j];
        n_tile_sparse++;
      } else {
        // Dense
        for (int k = 0; k < (num_excl<tilesize>::val); k++) {
          h_tile_excl_dense[n_tile_dense].excl[k] = h_tile_excl[j].excl[k];
        }
        h_tile_indj_dense[n_tile_dense] = h_tile_indj[j];
        n_tile_dense++;
        sparse_i_tiles = false;
      }
    }

    if (sparse_i_tiles) {
      // Sparse
    } else {
      h_ientry_dense[n_ientry_dense] = h_ientry[i];
      h_ientry_dense[n_ientry_dense].tileStart = startj_dense;
      h_ientry_dense[n_ientry_dense].tileEnd = n_tile_dense - 1;
      n_ientry_dense++;
    }
  }

  n_ientry = n_ientry_dense;
  n_tile = n_tile_dense;

  copy_HtoD_sync<ientry_t>(h_ientry_dense, ientry, n_ientry);
  copy_HtoD_sync<int>(h_tile_indj_dense, tile_indj, n_tile);
  copy_HtoD_sync<tile_excl_t<tilesize>>(h_tile_excl_dense, tile_excl, n_tile);

  allocate<ientry_t>(&ientry_sparse, n_ientry_sparse);
  allocate<int>(&tile_indj_sparse, n_tile_sparse);
  allocate<pairs_t<tilesize>>(&pairs, n_tile_sparse);
  ientry_sparse_len = n_ientry_sparse;
  tile_indj_sparse_len = n_tile_sparse;
  pairs_len = n_tile_sparse;

  copy_HtoD_sync<ientry_t>(h_ientry_sparse, ientry_sparse, n_ientry_sparse);
  copy_HtoD_sync<int>(h_tile_indj_sparse, tile_indj_sparse, n_tile_sparse);
  copy_HtoD_sync<pairs_t<tilesize>>(h_pairs, pairs, n_tile_sparse);

  delete[] h_ientry;
  delete[] h_tile_indj;
  delete[] h_tile_excl;

  delete[] h_ientry_dense;
  delete[] h_tile_indj_dense;
  delete[] h_tile_excl_dense;

  delete[] h_ientry_sparse;
  delete[] h_tile_indj_sparse;
  delete[] h_pairs;
}
#endif

//
// Removes empty tiles
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::remove_empty_tiles() {
  ientry_t *h_ientry = new ientry_t[n_ientry];
  int *h_tile_indj = new int[n_tile];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[n_tile];

  ientry_t *h_ientry_noempty = new ientry_t[n_ientry];
  int *h_tile_indj_noempty = new int[n_tile];
  tile_excl_t<tilesize> *h_tile_excl_noempty =
      new tile_excl_t<tilesize>[n_tile];

  copy_DtoH_sync<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH_sync<tile_excl_t<tilesize>>(tile_excl, h_tile_excl, n_tile);

  int n_ientry_noempty = 0;
  int n_tile_noempty = 0;
  for (int i = 0; i < n_ientry; i++) {
    bool empty_i_tiles = true;
    int startj_noempty = n_tile_noempty;
    for (int j = h_ientry[i].tileStart; j <= h_ientry[i].tileEnd; j++) {
      bool empty_tile = true;
      for (int k = 0; k < (num_excl<tilesize>::val); k++) {
        unsigned int n1bit = BitCount(h_tile_excl[j].excl[k]);
        if (n1bit != 32)
          empty_tile = false;
      }

      if (!empty_tile) {
        for (int k = 0; k < (num_excl<tilesize>::val); k++) {
          h_tile_excl_noempty[n_tile_noempty].excl[k] = h_tile_excl[j].excl[k];
        }
        h_tile_indj_noempty[n_tile_noempty] = h_tile_indj[j];
        n_tile_noempty++;
        empty_i_tiles = false;
      }
    }

    if (!empty_i_tiles) {
      h_ientry_noempty[n_ientry_noempty] = h_ientry[i];
      h_ientry_noempty[n_ientry_noempty].tileStart = startj_noempty;
      h_ientry_noempty[n_ientry_noempty].tileEnd = n_tile_noempty - 1;
      n_ientry_noempty++;
    }
  }

  n_ientry = n_ientry_noempty;
  n_tile = n_tile_noempty;

  copy_HtoD_sync<ientry_t>(h_ientry_noempty, ientry, n_ientry);
  copy_HtoD_sync<int>(h_tile_indj_noempty, tile_indj, n_tile);
  copy_HtoD_sync<tile_excl_t<tilesize>>(h_tile_excl_noempty, tile_excl, n_tile);

  delete[] h_ientry;
  delete[] h_tile_indj;
  delete[] h_tile_excl;

  delete[] h_ientry_noempty;
  delete[] h_tile_indj_noempty;
  delete[] h_tile_excl_noempty;
}

//
// Analyzes the neighbor list and prints info
//
template <int tilesize> void CudaNeighborListBuild<tilesize>::analyze() {
  std::cout << "Number of i-tiles = " << n_ientry
            << ", total number of tiles = " << n_tile << std::endl;

  ientry_t *h_ientry = new ientry_t[n_ientry];
  int *h_tile_indj = new int[n_tile];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[n_tile];

  copy_DtoH_sync<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH_sync<tile_excl_t<tilesize>>(tile_excl, h_tile_excl, n_tile);

  std::ofstream file_npair("npair.txt", std::ofstream::out);
  std::ofstream file_nj("nj.txt", std::ofstream::out);

  unsigned int nexcl_bit = 0;
  unsigned int nexcl_bit_self = 0;
  unsigned int nempty_tile = 0;
  unsigned int nempty_line = 0;
  unsigned int npair_tot = 0;
  for (int i = 0; i < n_ientry; i++) {
    file_nj << h_ientry[i].tileEnd - h_ientry[i].tileStart + 1 << std::endl;
    for (int j = h_ientry[i].tileStart; j <= h_ientry[i].tileEnd; j++) {
      int npair = 0;
      bool empty_tile = true;
      for (int k = 0; k < (num_excl<tilesize>::val); k++) {
        unsigned int n1bit = BitCount(h_tile_excl[j].excl[k]);

        if (n1bit > 32) {
          std::stringstream tmpexc;
          tmpexc << n1bit << " " << std::hex << h_tile_excl[j].excl[k]
                 << std::endl;
          throw std::invalid_argument(tmpexc.str());
          exit(1);
        }

        if (n1bit == 32)
          nempty_line++;
        else
          empty_tile = false;

        nexcl_bit += n1bit;
        npair += 32 - n1bit;

        if (h_ientry[i].iatomStart == h_tile_indj[j])
          nexcl_bit_self += n1bit;
      }
      if (empty_tile)
        nempty_tile++;
      file_npair << npair << std::endl;
      npair_tot += npair;
    }
  }

  file_npair.close();
  file_nj.close();

  unsigned int n_tile_pairs = n_tile * tilesize * tilesize;
  std::cout << "Total number of pairs = " << npair_tot << " ("
            << (double)npair_tot * 100.0 / (double)n_tile_pairs << "% full)"
            << std::endl;
  std::cout << "Total number of pairs in tiles = " << n_tile_pairs << std::endl;
  std::cout << "Number of excluded pairs = " << nexcl_bit << " ("
            << ((double)nexcl_bit * 100) / (double)n_tile_pairs << "%)"
            << std::endl;
  std::cout << "Number of excluded pairs in self (i==j) tiles = "
            << nexcl_bit_self << " ("
            << ((double)nexcl_bit_self * 100) / (double)n_tile_pairs << "%)"
            << std::endl;
  std::cout << "Number of empty lines = " << nempty_line << " ("
            << ((double)nempty_line * 100) / ((double)(n_tile * tilesize))
            << "%)" << std::endl;
  std::cout << "Number of empty tiles = " << nempty_tile << " ("
            << ((double)nempty_tile * 100) / (double)n_tile << "%)"
            << std::endl;

  delete[] h_ientry;
  delete[] h_tile_indj;
  delete[] h_tile_excl;
}

//
// Load neighbor list from file
//
template <int tilesize>
void CudaNeighborListBuild<tilesize>::load(const char *filename) {
  ientry_t *h_ientry;
  int *h_tile_indj;
  tile_excl_t<tilesize> *h_tile_excl;

  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);

    file >> n_ientry >> n_tile;

    h_ientry = new ientry_t[n_ientry];
    h_tile_indj = new int[n_tile];
    h_tile_excl = new tile_excl_t<tilesize>[n_tile];

    for (int i = 0; i < n_ientry; i++) {
      file >> std::dec >> h_ientry[i].iatomStart >> h_ientry[i].ish >>
          h_ientry[i].tileStart >> h_ientry[i].tileEnd;
      for (int j = h_ientry[i].tileStart; j <= h_ientry[i].tileEnd; j++) {
        file >> std::dec >> h_tile_indj[j];
        for (int k = 0; k < (num_excl<tilesize>::val); k++) {
          file >> std::hex >> h_tile_excl[j].excl[k];
        }
      }
    }

    file.close();
  } catch (std::ifstream::failure e) {
    // std::cerr << "Error opening/reading/closing file " << filename <<
    // std::endl;
    std::stringstream invalar;
    invalar << "Error opening/reading/closing file " << filename << "\n";
    std::string excs = invalar.str();
    throw std::invalid_argument(excs);
    exit(1);
  }

#ifdef STRICT_MEMORY_REALLOC
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.0f);
  reallocate<int>(&tile_indj, &tile_indj_len, n_tile, 1.0f);
  reallocate<tile_excl_t<tilesize>>(&tile_excl, &tile_excl_len, n_tile, 1.0f);
#else
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.2f);
  reallocate<int>(&tile_indj, &tile_indj_len, n_tile, 1.2f);
  reallocate<tile_excl_t<tilesize>>(&tile_excl, &tile_excl_len, n_tile, 1.2f);
#endif

  copy_HtoD_sync<ientry_t>(h_ientry, ientry, n_ientry);
  copy_HtoD_sync<int>(h_tile_indj, tile_indj, n_tile);
  copy_HtoD_sync<tile_excl_t<tilesize>>(h_tile_excl, tile_excl, n_tile);

  delete[] h_ientry;
  delete[] h_tile_indj;
  delete[] h_tile_excl;
}

//
// Explicit instances of CudaNeighborList
//
// template class CudaNeighborList<16>;
template class CudaNeighborListBuild<32>;

template int CudaNeighborListBuild<32>::calc_gpu_pairlist<double>(
    const int n_ientry, const ientry_t *ientry, const int *tile_indj,
    const tile_excl_t<32> *tile_excl, const float4 *xyzq, const double boxx,
    const double boxy, const double boxz, const double rcut);

template int CudaNeighborListBuild<32>::calc_cpu_pairlist<double>(
    const int *zone_patom, const float4 *xyzq, const int *loc2glo,
    const int *atom_excl_pos, const int *atom_excl, const double boxx,
    const double boxy, const double boxz, const double rcut);

#endif // NOCUDAC
