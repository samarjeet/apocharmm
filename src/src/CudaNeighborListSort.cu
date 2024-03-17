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
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cassert>
#include <iostream>
#include <stdio.h>
#include <sstream>
//#define BUCKET_SORT_IN_USE
#include "CudaNeighborListSort.h"

// IF defined, uses strict (Factor = 1.0f) memory reallocation. Used for
// debuggin memory problems.
#define STRICT_MEMORY_REALLOC

//
// Sort atoms into z-columns
//
// col_natom[0..ncellx*ncelly-1] = number of atoms in each column
// atom_icol[istart..iend]     = column index for atoms
//
__global__ void calc_z_column_index_kernel(
    const int izoneStart, const int izoneEnd,
    const ZoneParam_t *__restrict__ ZoneParam, const float4 *__restrict__ xyzq,
    int *__restrict__ col_natom, int *__restrict__ atom_icol,
    int3 *__restrict__ col_xy_zone) {
  // Atom index
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  int ind0 = 0;
  int istart = 0;
  int iend = 0;
  for (int izone = izoneStart; izone <= izoneEnd; izone++) {
    istart = iend;
    iend = istart + ZoneParam[izone].ncoord;
    if (i >= istart && i < iend) {
      float4 xyzq_val = xyzq[i];
      float x = xyzq_val.x;
      float y = xyzq_val.y;
      float3 min_xyz = ZoneParam[izone].min_xyz;
      int ncellx = ZoneParam[izone].ncellx;
      int ncelly = ZoneParam[izone].ncelly;
      int ix =
          min((int)((x - min_xyz.x) * ZoneParam[izone].inv_celldx), ncellx - 1);
      int iy =
          min((int)((y - min_xyz.y) * ZoneParam[izone].inv_celldy), ncelly - 1);
      int ind = ind0 + ix + iy * ncellx;
      atomicAdd(&col_natom[ind], 1);
      atom_icol[i] = ind;
      int3 col_xy_zone_val;
      col_xy_zone_val.x = ix;
      col_xy_zone_val.y = iy;
      col_xy_zone_val.z = izone;
      col_xy_zone[ind] = col_xy_zone_val;
      break;
    }
    ind0 += ZoneParam[izone].ncellx * ZoneParam[izone].ncelly;
  }
}

//
// For each zone, calculates the maximum number of atoms per z-column
// This information is used in the sorting of z-columns
//
// Thread block per zone
//
__global__ void calcZoneMaxZColNatomKernel(
    const int izoneStart, const ZoneParam_t *__restrict__ ZoneParam,
    const int *__restrict__ col_natom, int *__restrict__ zoneMaxZColNatom) {
  // Shared memory
  // Requires: blockDim.x*sizeof(int)
  extern __shared__ int shZoneMaxZColNatom[];

  // Number of columns in this zone
  const int ncol = ZoneParam[izoneStart + blockIdx.x].ncellx *
                   ZoneParam[izoneStart + blockIdx.x].ncelly;
  // Starting column index in this zone
  const int col0 = ZoneParam[izoneStart + blockIdx.x].zone_col -
                   ZoneParam[izoneStart].zone_col;

  // Result, only used by threadIdx.x = 0
  int max_val = 0;
  for (int base = 0; base < ncol; base += blockDim.x) {
    // Load blockDim.x number of values into shared memory
    shZoneMaxZColNatom[threadIdx.x] =
        (base + threadIdx.x < ncol) ? col_natom[col0 + base + threadIdx.x] : 0;
    __syncthreads();

    // Find maximum of blockDim.x number of values
    for (int d = 1; d < blockDim.x; d *= 2) {
      int val = (threadIdx.x >= d) ? shZoneMaxZColNatom[threadIdx.x - d] : 0;
      __syncthreads();
      shZoneMaxZColNatom[threadIdx.x] =
          max(shZoneMaxZColNatom[threadIdx.x], val);
      __syncthreads();
    }

    // Compare maximum found to the global maximum
    if (threadIdx.x == 0)
      max_val = max(max_val, shZoneMaxZColNatom[blockDim.x - 1]);
  }

  // Write result into global memory
  if (threadIdx.x == 0) {
    zoneMaxZColNatom[blockIdx.x] = max_val;
  }
}

//
// Computes z column position using parallel exclusive prefix sum
// Also computes the cell_patom, col_ncellz, col_cell, and ncell
//
// NOTE: Must have nblock = 1, we loop over buckets to avoid multiple kernel
// calls
//
__global__ void calc_z_column_pos_kernel(
    const int tilesize, const int ncol_tot, const int atomStart,
    const int cellStart, const int3 *__restrict__ col_xy_zone,
    int *__restrict__ col_natom, int *__restrict__ col_patom,
    int *__restrict__ cell_patom, int *__restrict__ col_ncellz,
    int4 *__restrict__ cell_xyz_zone, int *__restrict__ col_cell,
    NlistParam_t *__restrict__ d_NlistParam) {
  // Shared memory
  // Requires: blockDim.x*sizeof(int2)
  // This shared memory buffer is used for computing positions (cumulative
  // sums) of columns, size blockDim.x*sizeof(int2)
  extern __shared__ int2 sh_pos[];

  if (threadIdx.x == 0) {
    col_patom[0] = 0;
  }

  int2 offset = make_int2(0, 0);
  for (int base = 0; base < ncol_tot; base += blockDim.x) {
    // i = column index
    int i = base + threadIdx.x;
    int2 tmpval;
    tmpval.x =
        (i < ncol_tot) ? col_natom[i] : 0; // Number of atoms in this column
    tmpval.y = (i < ncol_tot) ? (tmpval.x - 1) / tilesize + 1
                              : 0; // Number of z-cells in this column (NOTE:
                                   // this needs to be 0 when tmpval.x = 0)
    if (i < ncol_tot)
      col_ncellz[i] = tmpval.y; // Set col_ncellz[icol]
    sh_pos[threadIdx.x] = tmpval;
    if (i < ncol_tot)
      col_natom[i] = 0;
    __syncthreads();

    for (int d = 1; d < blockDim.x; d *= 2) {
      int2 posd =
          (threadIdx.x >= d) ? sh_pos[threadIdx.x - d] : make_int2(0, 0);
      __syncthreads();
      int2 pos0 = sh_pos[threadIdx.x];
      pos0.x += posd.x;
      pos0.y += posd.y;
      sh_pos[threadIdx.x] = pos0;
      __syncthreads();
    }

    if (i < ncol_tot) {
      // shpos[threadIdx.x].x = cumulative sum of number of atoms in this
      // column shpos[threadIdx.x].y = cumulative sum of number of z-cells
      // in this column Write col_patom in global memory
      int2 val1 = sh_pos[threadIdx.x];
      val1.x += offset.x;
      val1.y += offset.y;
      col_patom[i + 1] = val1.x;
      // Write cell_patom in global memory
      // OPTIMIZATION NOTE: Is this looping too slow? Should we move this
      // into a separate kernel?
      int2 val0 = (threadIdx.x > 0) ? sh_pos[threadIdx.x - 1] : make_int2(0, 0);
      val0.x += offset.x;
      val0.y += offset.y;
      int icell0 = val0.y;
      int icell1 = val1.y;
      int iatom = val0.x + atomStart;
      // Write col_cell
      col_cell[i] = icell0 + cellStart;
      // col_xy_zone[icol].x = x coordinate for each column
      // col_xy_zone[icol].y = y coordinate for each column
      // col_xy_zone[icol].z = zone for each column
      int4 cell_xyz_zone_val;
      int3 col_xy_zone_val = col_xy_zone[i];
      cell_xyz_zone_val.x = col_xy_zone_val.x; // icellx
      cell_xyz_zone_val.y = col_xy_zone_val.y; // icelly
      cell_xyz_zone_val.z = 0;                 // icellz (set in the loop below)
      cell_xyz_zone_val.w = col_xy_zone_val.z; // izone
      for (int icell = icell0; icell < icell1;
           icell++, iatom += tilesize, cell_xyz_zone_val.z++) {
        cell_patom[icell] = iatom;
        cell_xyz_zone[icell] = cell_xyz_zone_val;
      }
    }

    // Add the last value to the offset for the next block
    int2 lastval = sh_pos[blockDim.x - 1];
    offset.x += lastval.x;
    offset.y += lastval.y;

    // Sync threads so that the next iteration can start writing in shared
    // memory
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    // Cap off cell_patom
    cell_patom[offset.y] = offset.x + atomStart;
    // Write ncell into global GPU buffer
    d_NlistParam->ncell = offset.y;
    // Clear nexcl
    d_NlistParam->nexcl = 0;
    // Zero n_ientry and n_tile -counters in preparation for neighbor list
    // build
    d_NlistParam->n_ientry = 0;
    d_NlistParam->n_tile = 0;
  }

  // Set zone_cell = starting cell for each zone
  // if (threadIdx.x < maxNumZone) {
  //  int icol = d_ZoneParam[threadIdx.x].zone_col;
  //  d_NlistParam.zone_cell[threadIdx.x] = (icol < ncol_tot) ? col_cell[icol]
  //  : d_NlistParam.ncell;
  //}
}

/*
//
// Calculates ncellz_max[izone].
//
// blockDim.x = max number of columns over all zones
// Each thread block calculates one zone (blockIdx.x = izone)
//
__global__ void calc_ncellz_max_kernel(const int* __restrict__ col_ncellz) {

  // Shared memory
  // Requires: blockDim.x*sizeof(int)
  extern __shared__ int sh_col_ncellz[];

  // ncol[izone] gives the cumulative sum of ncellx[izone]*ncelly[izone]
  // NOTE: we can use zone_col & ncol_tot to replace ncol
  int start = d_NlistParam.ncol[blockIdx.x];
  int end   = d_NlistParam.ncol[blockIdx.x+1] - 1;

  int ncellz_max = 0;

  for (;start <= end;start += blockDim.x) {
    // Load col_ncellz into shared memory
    int pos = start + threadIdx.x;
    int col_ncellz_val = 0;
    if (pos <= end) col_ncellz_val = col_ncellz[pos];
    sh_col_ncellz[threadIdx.x] = col_ncellz_val;
    __syncthreads();

    // Reduce
    int n = end - start;
    for (int d=1;d < n;d *= 2) {
      int t = threadIdx.x + d;
      int val = (t < n) ? sh_col_ncellz[t] : 0;
      __syncthreads();
      sh_col_ncellz[threadIdx.x] = max(sh_col_ncellz[threadIdx.x], val);
      __syncthreads();
    }

    // Store into register
    if (threadIdx.x == 0) ncellz_max = max(ncellz_max, sh_col_ncellz[0]);
  }

  // Write into global memory
  if (threadIdx.x == 0) d_ZoneParam[blockIdx.x].ncellz_max = ncellz_max;
}
*/

/*
//
// Calculates celldz_min[izone], where izone = blockIdx.x = 0...7
//
__global__ void calc_celldz_min_kernel() {

  // Shared memory
  // Requires: blockDim.x*sizeof(float)
  extern __shared__ float sh_celldz_min[];

  // ncol[izone] gives the cumulative sum of ncellx[izone]*ncelly[izone]
  int start = d_NlistParam.ncell[blockIdx.x];
  int end   = d_NlistParam.ncell[blockIdx.x+1] - 1;

  float celldz_min = (float)(1.0e20);

  for (;start <= end;start += blockDim.x) {
    // Load value into shared memory
    float celldz_min_val = (float)(1.0e20);
    int pos = start + threadIdx.x;
    if (pos <= end) celldz_min_val = ;
    sh_celldz_min[threadIdx.x] = celldz_min_val;
    __synthreads();

    // Reduce
    int n = end - start;
    for (int d=1;d < n;d *= 2) {
      int t = threadIdx.x + d;
      float val = (t < n) ? sh_celldz_min[t] : (float)(1.0e20);
      __syncthreads();
      sh_celldz_min[threadIdx.x] = min(sh_celldz_min[threadIdx.x], val);
      __syncthreads();
    }

    // Store into register
    if (threadIdx.x == 0) celldz_min = min(celldz_min, sh_celldz_min[0]);
  }

  // Write into global memory
  if (threadIdx.x == 0) d_NlistParam.celldz_min[blockIdx.x] = celldz_min;

}
*/

//
// Finds the min_xyz and max_xyz for zone "izone"
//
__global__ void calc_minmax_xyz_kernel(const int ncoord, const int izone,
                                       const float4 *__restrict__ xyzq,
                                       ZoneParam_t *__restrict__ d_ZoneParam) {
  // Shared memory
  // Requires: 6*blockDim.x*sizeof(float)
  extern __shared__ float sh_minmax_xyz[];
  volatile float *sh_min_x = &sh_minmax_xyz[0];
  volatile float *sh_min_y = &sh_minmax_xyz[blockDim.x];
  volatile float *sh_min_z = &sh_minmax_xyz[blockDim.x * 2];
  volatile float *sh_max_x = &sh_minmax_xyz[blockDim.x * 3];
  volatile float *sh_max_y = &sh_minmax_xyz[blockDim.x * 4];
  volatile float *sh_max_z = &sh_minmax_xyz[blockDim.x * 5];

  // Load data into shared memory
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  float4 xyzq_i = xyzq[min(i, ncoord - 1)];
  float x = xyzq_i.x;
  float y = xyzq_i.y;
  float z = xyzq_i.z;

  sh_min_x[threadIdx.x] = x;
  sh_min_y[threadIdx.x] = y;
  sh_min_z[threadIdx.x] = z;
  sh_max_x[threadIdx.x] = x;
  sh_max_y[threadIdx.x] = y;
  sh_max_z[threadIdx.x] = z;
  __syncthreads();

  // Reduce
  for (int d = 1; d < blockDim.x; d *= 2) {
    int t = threadIdx.x + d;
    float min_x = (t < blockDim.x) ? sh_min_x[t] : (float)(1.0e20);
    float min_y = (t < blockDim.x) ? sh_min_y[t] : (float)(1.0e20);
    float min_z = (t < blockDim.x) ? sh_min_z[t] : (float)(1.0e20);
    float max_x = (t < blockDim.x) ? sh_max_x[t] : (float)(-1.0e20);
    float max_y = (t < blockDim.x) ? sh_max_y[t] : (float)(-1.0e20);
    float max_z = (t < blockDim.x) ? sh_max_z[t] : (float)(-1.0e20);
    __syncthreads();
    sh_min_x[threadIdx.x] = min(sh_min_x[threadIdx.x], min_x);
    sh_min_y[threadIdx.x] = min(sh_min_y[threadIdx.x], min_y);
    sh_min_z[threadIdx.x] = min(sh_min_z[threadIdx.x], min_z);
    sh_max_x[threadIdx.x] = max(sh_max_x[threadIdx.x], max_x);
    sh_max_y[threadIdx.x] = max(sh_max_y[threadIdx.x], max_y);
    sh_max_z[threadIdx.x] = max(sh_max_z[threadIdx.x], max_z);
    __syncthreads();
  }

  // Store into global memory
  if (threadIdx.x == 0) {
    atomicMin(&d_ZoneParam[izone].min_xyz.x, sh_min_x[0]);
    atomicMin(&d_ZoneParam[izone].min_xyz.y, sh_min_y[0]);
    atomicMin(&d_ZoneParam[izone].min_xyz.z, sh_min_z[0]);
    atomicMax(&d_ZoneParam[izone].max_xyz.x, sh_max_x[0]);
    atomicMax(&d_ZoneParam[izone].max_xyz.y, sh_max_y[0]);
    atomicMax(&d_ZoneParam[izone].max_xyz.z, sh_max_z[0]);
  }
}

//
// Re-order atoms according to pos. Non-deterministic version (because of
// atomicAdd())
//
__global__ void reorder_atoms_z_column_kernel(
    const int ncoord, const int *atom_icol, int *col_natom,
    const int *col_patom, const float4 *__restrict__ xyzq_in,
    float4 *__restrict__ xyzq_out, int *__restrict__ ind_sorted) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < ncoord) {
    // Column index
    int icol = atom_icol[i];
    int pos = col_patom[icol];
    int n = atomicAdd(&col_natom[icol], 1);
    // new position = pos + n
    int newpos = pos + n;
    ind_sorted[newpos] = i;
    xyzq_out[newpos] = xyzq_in[i];
  }
}

//
// Reorders loc2glo using ind_sorted and shifts ind_sorted by atomStart
//
__global__ void reorder_loc2glo_kernel(const int ncoord, const int atomStart,
                                       int *__restrict__ ind_sorted,
                                       const int *__restrict__ loc2glo_in,
                                       int *__restrict__ loc2glo_out) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < ncoord) {
    int j = ind_sorted[i];
    ind_sorted[i] = j + atomStart;
    loc2glo_out[i] = loc2glo_in[j];
  }
}

//
// Builds glo2loc using loc2glo
//
__global__ void build_glo2loc_kernel(const int ncoord, const int atomStart,
                                     const int *__restrict__ loc2glo,
                                     int *__restrict__ glo2loc) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < ncoord) {
    int ig = loc2glo[i];
    glo2loc[ig] = i + atomStart;
  }
}

/*
//
// Builds atom_pcell. Single warp takes care of single cell
//
__global__ void build_atom_pcell_kernel(const int* __restrict__ cell_patom,
                                        int* __restrict__ atom_pcell) {
  const int icell = (threadIdx.x + blockIdx.x*blockDim.x)/warpsize;
  const int wid = threadIdx.x % warpsize;

  if (icell < d_NlistParam.ncell) {
    int istart = cell_patom[icell];
    int iend   = cell_patom[icell+1] - 1;
    if (istart + wid <= iend) atom_pcell[istart + wid] = icell;
  }

}
*/

//
// Sorts atoms according to z coordinate
//
// Uses bitonic sort, see:
// http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
//
// Each thread block sorts a single z column.
//
// Shared memory version (use_shmem = true)
// Length of z-columns is limited by the shared memory size.
// max_n not used
// glo_keyvalBase = NULL
//
// Global memory version (use_shmem = false)
// No limitations on the length of the z-columns
// max_n = maximum number of atoms across all z-columns
// glo_keyvalBase = buffer of size max_n*blockDim.x
//
template <bool use_shmem>
__global__ void
bitonicSortZColKernel(const int max_n, const int *__restrict__ col_patom,
                      float4 *__restrict__ xyzq, float4 *__restrict__ xyzqTmp,
                      int *__restrict__ indSorted,
                      keyval_t *__restrict__ glo_keyvalBase) {
  // Shared memory:
  // Requires max_n*sizeof(keyval_t) for the shared memory version
  extern __shared__ keyval_t sh_keyval[];

  const int col_patom0 = col_patom[blockIdx.x];
  const int n = col_patom[blockIdx.x + 1] - col_patom0;

  // Get pointer for this column (global memory version only)
  keyval_t *__restrict__ glo_keyval;
  if (!use_shmem)
    glo_keyval = &glo_keyvalBase[max_n * blockIdx.x];

  // Read keys and values into shared/global memory
  // key = z-coordinate
  // val = index
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    keyval_t keyval;
    keyval.key = xyzq[i + col_patom0].z;
    keyval.val = i + col_patom0;
    if (use_shmem) {
      sh_keyval[i] = keyval;
    } else {
      glo_keyval[i] = keyval;
    }
  }
  __syncthreads();

  for (int k = 2; k < 2 * n; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int ixj = i ^ j;
        if (ixj > i && ixj < n) {
          // asc = true for ascending order
          bool asc = ((i & k) == 0);
          for (int kk = k * 2; kk < 2 * n; kk *= 2)
            asc = ((i & kk) == 0 ? !asc : asc);

          // Read data
          keyval_t keyval1 = (use_shmem) ? sh_keyval[i] : glo_keyval[i];
          keyval_t keyval2 = (use_shmem) ? sh_keyval[ixj] : glo_keyval[ixj];

          float lo_key = asc ? keyval1.key : keyval2.key;
          float hi_key = asc ? keyval2.key : keyval1.key;

          if (lo_key > hi_key) {
            // keys are in wrong order => exchange
            if (use_shmem) {
              sh_keyval[i] = keyval2;
              sh_keyval[ixj] = keyval1;
            } else {
              glo_keyval[i] = keyval2;
              glo_keyval[ixj] = keyval1;
            }
          }
        }
      }
      __syncthreads();
    }
  }

  // sh_keyval[threadIdx.x].val gives the mapping:
  //
  // xyzq_new[threadIdx.x + col_patom0]        =
  // xyzq[sh_keyval[threadIdx.x].val]
  // loc2glo_new[threadIdx.x + col_patom0] =
  // loc2glo[sh_keyval[threadIdx.x].val]
  //

  // keys are not needed anymore, store index into that memory location
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (use_shmem) {
      sh_keyval[i].ind = indSorted[i + col_patom0];
    } else {
      glo_keyval[i].ind = indSorted[i + col_patom0];
    }
  }
  __syncthreads();

  for (int ibase = 0; ibase < n; ibase += blockDim.x) {
    int i = ibase + threadIdx.x;
    float4 xyzq_val;
    int ind_val;
    if (i < n) {
      int pos = (use_shmem) ? sh_keyval[i].val : glo_keyval[i].val;
      ind_val = (use_shmem) ? sh_keyval[pos - col_patom0].ind
                            : glo_keyval[pos - col_patom0].ind;
      xyzq_val = xyzq[pos];
    }
    __syncthreads();
    if (i < n) {
      int newpos = i + col_patom0;
      indSorted[newpos] = ind_val;
      xyzqTmp[newpos] = xyzq_val;
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    xyzq[i + col_patom0] = xyzqTmp[i + col_patom0];
  }
}

#ifdef BUCKET_SORT_IN_USE
//
// Sort atoms in z-columns using bucket sort
// One block does one z-column
//
// min_z = minimum z-coordinate for this zone
// invBucketWidth = 1/(bucket width)
// numBucket = number of buckets per column
// bucketPos[]   = for each bucket, position
// bucketIndex[] = for each atom, bucket index
//
__global__ void bucketSortZColKernel(
    const int *__restrict__ col_patom, const float min_z,
    const float invBucketWidth, const int numBucket,
    int *__restrict__ bucketPosBase, int *__restrict__ bucketIndexBase,
    float4 *__restrict__ xyzqBase, float4 *__restrict__ xyzqTmpBase,
    int *__restrict__ indSortedBase, int *__restrict__ indSortedTmpBase) {
  // Shared memory, requires: blockDim.x*sizeof(int)
  extern __shared__ int shBucketPos[];

  // Get atom starting index and number of coordinates
  const int atomStart = col_patom[blockIdx.x];
  const int ncoord = col_patom[blockIdx.x + 1] - atomStart;

  // Pointers for this column
  int *__restrict__ bucketPos = &bucketPosBase[numBucket * blockIdx.x];
  int *__restrict__ bucketIndex = &bucketIndexBase[atomStart];
  float4 *__restrict__ xyzq = &xyzqBase[atomStart];
  float4 *__restrict__ xyzqTmp = &xyzqTmpBase[atomStart];
  int *__restrict__ indSorted = &indSortedBase[atomStart];
  int *__restrict__ indSortedTmp = &indSortedTmpBase[atomStart];

  // Before starting, clear bucketPos
  for (int i = threadIdx.x; i < numBucket; i += blockDim.x)
    bucketPos[i] = 0;
  __syncthreads();

  // Assign atoms into buckets
  for (int i = threadIdx.x; i < ncoord; i += blockDim.x) {
    int ibucket =
        min((int)((xyzq[i].z - min_z) * invBucketWidth), numBucket - 1);
    atomicAdd(&bucketPos[ibucket], 1);
    bucketIndex[i] = ibucket;
  }
  __syncthreads();

  // Compute position of buckets
  int pos0 = 0;
  for (int base = 0; base < ncoord; base += blockDim.x) {
    // Load bucketPos into shared memory
    shBucketPos[threadIdx.x] =
        (base + threadIdx.x < ncoord) ? bucketPos[base + threadIdx.x] : 0;
    __syncthreads();
    // Reduce in shared memory, result is inclusive cumsum
    for (int d = 1; d < blockDim.x; d *= 2) {
      int val = (threadIdx.x >= d) ? shBucketPos[threadIdx.x - d] : 0;
      __syncthreads();
      shBucketPos[threadIdx.x] += val;
      __syncthreads();
    }
    // Write result back into global memory and shift to get exclusive
    // cumsum
    if (base + threadIdx.x < ncoord)
      bucketPos[base + threadIdx.x] =
          pos0 + (threadIdx.x >= 1) ? shBucketPos[threadIdx.x - 1] : 0;
    // Broadcast last value to pos0
    pos0 = shBucketPos[blockDim.x - 1];
  }
  __syncthreads();

  // Re-order according to buckets, result in (xyzqTmp, indSortedTmp)
  for (int i = threadIdx.x; i < ncoord; i += blockDim.x) {
    int ibucket = bucketIndex[i];
    int pos = atomicAdd(&bucketPos[ibucket], 1);
    xyzqTmp[pos] = xyzq[i];
    indSortedTmp[pos] = indSorted[i];
  }
  __syncthreads();

  // Sort within buckets
  THIS PART NOT IMPLEMENTED !

      // Copy final result back to xyzq
      for (int i = threadIdx.x; i < ncoord; i += blockDim.x) {
    xyzq[i] = xyzqTmp[i];
    indSorted[i] = indSortedTmp[i];
  }
}
#endif

//
// Calculates bounding box (bb) and cell z-boundaries (cell_bz)
// NOTE: Each thread calculates one bounding box
//
__global__ void calc_bb_cell_bz_kernel(const int ncell,
                                       const int *__restrict__ cell_patom,
                                       const float4 *__restrict__ xyzq,
                                       bb_t *__restrict__ bb,
                                       float *__restrict__ cell_bz) {
  const int icell = threadIdx.x + blockIdx.x * blockDim.x;

  if (icell < ncell) {
    int istart = cell_patom[icell];
    int iend = cell_patom[icell + 1] - 1;
    float4 xyzq_val = xyzq[istart];
    float minx = xyzq_val.x;
    float miny = xyzq_val.y;
    float minz = xyzq_val.z;
    float maxx = xyzq_val.x;
    float maxy = xyzq_val.y;
    float maxz = xyzq_val.z;

    for (int i = istart + 1; i <= iend; i++) {
      xyzq_val = xyzq[i];
      minx = min(minx, xyzq_val.x);
      miny = min(miny, xyzq_val.y);
      minz = min(minz, xyzq_val.z);
      maxx = max(maxx, xyzq_val.x);
      maxy = max(maxy, xyzq_val.y);
      maxz = max(maxz, xyzq_val.z);
    }
    // Set the cell z-boundary equal to the z-coordinate of the last atom
    cell_bz[icell] = xyzq_val.z;
    bb_t bb_val;
    bb_val.x = 0.5f * (minx + maxx);
    bb_val.y = 0.5f * (miny + maxy);
    bb_val.z = 0.5f * (minz + maxz);
    bb_val.wx = 0.5f * (maxx - minx);
    bb_val.wy = 0.5f * (maxy - miny);
    bb_val.wz = 0.5f * (maxz - minz);
    bb[icell] = bb_val;
  }
}

//########################################################################################
//########################################################################################
//########################################################################################

//
// Class creator
//
CudaNeighborListSort::CudaNeighborListSort(const int tilesize,
                                           const int izoneStart,
                                           const int izoneEnd)
    : tilesize(tilesize), izoneStart(izoneStart), izoneEnd(izoneEnd) {
  col_natom_len = 0;
  col_natom = NULL;

  col_patom_len = 0;
  col_patom = NULL;

  atom_icol_len = 0;
  atom_icol = NULL;

  col_xy_zone_len = 0;
  col_xy_zone = NULL;

  loc2gloTmp_len = 0;
  loc2gloTmp = NULL;

  xyzqTmpLen = 0;
  xyzqTmp = NULL;

#ifdef BUCKET_SORT_IN_USE
  bucketPosLen = 0;
  bucketPos = NULL;

  bucketIndexLen = 0;
  bucketIndex = NULL;

  indSortedTmpLen = 0;
  indSortedTmp = NULL;
#else
  keyvalBufferLen = 0;
  keyvalBuffer = NULL;
#endif

  test = false;

  // Allocate pinned host memory
  allocate_host<int>(&h_ncell, 1);
  allocate_host<int>(&h_zoneMaxZColNatom, izoneEnd - izoneStart + 1);

  // Allocate device memory
  allocate<int>(&d_zoneMaxZColNatom, izoneEnd - izoneStart + 1);

  // Create events
  cudaCheck(cudaEventCreate(&ncell_copy_event));
  cudaCheck(cudaEventCreate(&zoneMaxZColNatom_copy_event));
}

//
// Class destructor
//
CudaNeighborListSort::~CudaNeighborListSort() {
  if (col_natom != NULL)
    deallocate<int>(&col_natom);
  if (col_patom != NULL)
    deallocate<int>(&col_patom);
  if (atom_icol != NULL)
    deallocate<int>(&atom_icol);
  if (col_xy_zone != NULL)
    deallocate<int3>(&col_xy_zone);
  if (loc2gloTmp != NULL)
    deallocate<int>(&loc2gloTmp);
  if (xyzqTmp != NULL)
    deallocate<float4>(&xyzqTmp);
#ifdef BUCKET_SORT_IN_USE
  if (bucketPos != NULL)
    deallocate<int>(&bucketPos);
  if (bucketIndex != NULL)
    deallocate<int>(&bucketIndex);
  if (indSortedTmp != NULL)
    deallocate<int>(&indSortedTmp);
#else
  if (keyvalBuffer != NULL)
    deallocate<keyval_t>(&keyvalBuffer);
#endif
  deallocate_host<int>(&h_ncell);
  deallocate_host<int>(&h_zoneMaxZColNatom);
  deallocate<int>(&d_zoneMaxZColNatom);
  cudaCheck(cudaEventDestroy(ncell_copy_event));
  cudaCheck(cudaEventDestroy(zoneMaxZColNatom_copy_event));
}

//
// Copies h_ZoneParam (CPU) -> d_ZoneParam (GPU)
//
void CudaNeighborListSort::setZoneParam(ZoneParam_t *h_ZoneParam,
                                        ZoneParam_t *d_ZoneParam,
                                        cudaStream_t stream) {
  copy_HtoD<ZoneParam_t>(h_ZoneParam + izoneStart, d_ZoneParam + izoneStart,
                         izoneEnd - izoneStart + 1, stream);
}

//
// Copies d_ZoneParam (GPU) -> h_ZoneParam (CPU)
//
void CudaNeighborListSort::getZoneParam(ZoneParam_t *h_ZoneParam,
                                        ZoneParam_t *d_ZoneParam,
                                        cudaStream_t stream) {
  cudaCheck(cudaStreamSynchronize(stream));
  copy_DtoH_sync<ZoneParam_t>(d_ZoneParam + izoneStart,
                              h_ZoneParam + izoneStart,
                              izoneEnd - izoneStart + 1);
}

//
// Calculate min_xyz and max_xyz
//
void CudaNeighborListSort::calc_min_max_xyz(const int *zone_patom,
                                            const float4 *xyzq,
                                            ZoneParam_t *h_ZoneParam,
                                            ZoneParam_t *d_ZoneParam,
                                            cudaStream_t stream) {
  for (int izone = izoneStart; izone <= izoneEnd; izone++) {
    h_ZoneParam[izone].min_xyz.x = (float)1.0e20;
    h_ZoneParam[izone].min_xyz.y = (float)1.0e20;
    h_ZoneParam[izone].min_xyz.z = (float)1.0e20;
    h_ZoneParam[izone].max_xyz.x = (float)(-1.0e20);
    h_ZoneParam[izone].max_xyz.y = (float)(-1.0e20);
    h_ZoneParam[izone].max_xyz.z = (float)(-1.0e20);
  }

  setZoneParam(h_ZoneParam, d_ZoneParam, stream);

  for (int izone = izoneStart; izone <= izoneEnd; izone++) {
    int nstart = zone_patom[izone];
    int ncoord_zone = zone_patom[izone + 1] - nstart;
    if (ncoord_zone > 0) {
      int nthread = 512;
      int nblock = (ncoord_zone - 1) / nthread + 1;
      int shmem_size = 6 * nthread * sizeof(float);
      calc_minmax_xyz_kernel<<<nblock, nthread, shmem_size, stream>>>(
          ncoord_zone, izone, &xyzq[nstart], d_ZoneParam);
    }
  }

  getZoneParam(h_ZoneParam, d_ZoneParam, stream);
}

//
// Setups sort parameters: NlistParam, ncol_tot, ncell_max
//
// NOTE: ncell_max is an approximate upper bound for the number of cells,
//       it is possible to blow this bound, so we should check for it
void CudaNeighborListSort::sort_setup(const int *zone_patom,
                                      ZoneParam_t *h_ZoneParam,
                                      ZoneParam_t *d_ZoneParam,
                                      cudaStream_t stream) {
  //
  // Setup ZoneParam and calculate ncol_tot, ncoord_tot, and ncell_max
  //
  ncol_tot = 0;
  ncoord_tot = 0;
  ncell_max = 0;
  if (izoneStart == 0)
    h_ZoneParam[0].zone_col = 0;
  for (int izone = izoneStart; izone <= izoneEnd; izone++) {
    h_ZoneParam[izone].ncoord = zone_patom[izone + 1] - zone_patom[izone];
    if (h_ZoneParam[izone].ncoord > 0) {
      // NOTE: we increase the cell sizes here by 0.001 to make sure no
      // atoms drop outside cells
      float xsize =
          h_ZoneParam[izone].max_xyz.x - h_ZoneParam[izone].min_xyz.x + 0.001f;
      float ysize =
          h_ZoneParam[izone].max_xyz.y - h_ZoneParam[izone].min_xyz.y + 0.001f;
      float zsize =
          h_ZoneParam[izone].max_xyz.z - h_ZoneParam[izone].min_xyz.z + 0.001f;
      float delta = powf(xsize * ysize * zsize * tilesize /
                             (float)h_ZoneParam[izone].ncoord,
                         1.0f / 3.0f);
      h_ZoneParam[izone].ncellx = max(1, (int)(xsize / delta));
      h_ZoneParam[izone].ncelly = max(1, (int)(ysize / delta));
      // Approximation for ncellz = 2 x "uniform distribution of atoms"
      h_ZoneParam[izone].ncellz_max =
          max(1, 2 * h_ZoneParam[izone].ncoord /
                     (h_ZoneParam[izone].ncellx * h_ZoneParam[izone].ncelly *
                      tilesize));
      /*
      fprintf(stderr,"%d %d %d | %f %f %f | %f %f %f | %f %f %f\n",
              h_ZoneParam[izone].ncellx, h_ZoneParam[izone].ncelly,
      h_ZoneParam[izone].ncellz_max,
              h_ZoneParam[izone].min_xyz.x, h_ZoneParam[izone].min_xyz.y,
      h_ZoneParam[izone].min_xyz.z,
              h_ZoneParam[izone].max_xyz.x, h_ZoneParam[izone].max_xyz.y,
      h_ZoneParam[izone].max_xyz.z,
              xsize, ysize, zsize);
      */
      h_ZoneParam[izone].celldx = xsize / (float)(h_ZoneParam[izone].ncellx);
      h_ZoneParam[izone].celldy = ysize / (float)(h_ZoneParam[izone].ncelly);
      h_ZoneParam[izone].celldz_min =
          zsize / (float)(h_ZoneParam[izone].ncellz_max);
      if (test) {
         std::stringstream tmpexc; 
        tmpexc << izone << ": " << h_ZoneParam[izone].min_xyz.z << " ... "
                  << h_ZoneParam[izone].max_xyz.z << std::endl;
        throw std::invalid_argument(tmpexc.str());
      }
    } else {
      h_ZoneParam[izone].ncellx = 0;
      h_ZoneParam[izone].ncelly = 0;
      h_ZoneParam[izone].ncellz_max = 0;
      h_ZoneParam[izone].celldx = 1.0f;
      h_ZoneParam[izone].celldy = 1.0f;
      h_ZoneParam[izone].celldz_min = 1.0f;
    }
    h_ZoneParam[izone].inv_celldx = 1.0f / h_ZoneParam[izone].celldx;
    h_ZoneParam[izone].inv_celldy = 1.0f / h_ZoneParam[izone].celldy;
    if (izone > 0) {
      h_ZoneParam[izone].zone_col =
          h_ZoneParam[izone - 1].zone_col +
          h_ZoneParam[izone - 1].ncellx * h_ZoneParam[izone - 1].ncelly;
    }
    int ncellxy = h_ZoneParam[izone].ncellx * h_ZoneParam[izone].ncelly;
    ncol_tot += ncellxy;
    ncoord_tot += h_ZoneParam[izone].ncoord;
    ncell_max += ncellxy * h_ZoneParam[izone].ncellz_max;
  }

  // Copy h_ZoneParam => d_ZoneParam
  setZoneParam(h_ZoneParam, d_ZoneParam, stream);

  // Wait till setZoneParam finishes
  cudaCheck(cudaStreamSynchronize(stream));
}

//
// Allocates / Re-allocates memory for sort
//
void CudaNeighborListSort::sort_realloc() {
#ifdef STRICT_MEMORY_REALLOC
  float fac = 1.0f;
#else
  float fac = 1.2f;
#endif

  reallocate<int>(&col_natom, &col_natom_len, ncol_tot, fac);
  reallocate<int>(&col_patom, &col_patom_len, ncol_tot + 1, fac);
  reallocate<int3>(&col_xy_zone, &col_xy_zone_len, ncol_tot, fac);

  reallocate<int>(&atom_icol, &atom_icol_len, ncoord_tot, fac);

  reallocate<int>(&loc2gloTmp, &loc2gloTmp_len, ncoord_tot, fac);
  reallocate<float4>(&xyzqTmp, &xyzqTmpLen, ncoord_tot, fac);
}

//
// Sorts atoms, core subroutine.
//
void CudaNeighborListSort::sort_core(
    const int *zone_patom, const int cellStart, const int colStart,
    ZoneParam_t *h_ZoneParam, ZoneParam_t *d_ZoneParam,
    NlistParam_t *d_NlistParam, int *cell_patom, int *col_ncellz,
    int4 *cell_xyz_zone, int *col_cell, int *ind_sorted, const float4 *xyzq,
    float4 *xyzq_sorted, cudaStream_t stream) {
  int nthread, nblock, shmem_size;

  int atomStart = zone_patom[izoneStart];

  // Clear col_natom
  clear_gpu_array<int>(col_natom, ncol_tot, stream);

  //
  // Calculate number of atoms in each z-column (col_natom)
  // and the column index for each atom (atom_icol)
  //
  nthread = 512;
  nblock = (ncoord_tot - 1) / nthread + 1;
  calc_z_column_index_kernel<<<nblock, nthread, 0, stream>>>(
      izoneStart, izoneEnd, d_ZoneParam, xyzq + atomStart, col_natom, atom_icol,
      col_xy_zone);
  cudaCheck(cudaGetLastError());

  //
  // For each zone, calculate maximum number of atoms in columns
  //
  nthread = 0;
  for (int izone = izoneStart; izone <= izoneEnd; izone++) {
    nthread =
        max(nthread, h_ZoneParam[izone].ncellx * h_ZoneParam[izone].ncelly);
  }
  nthread = min(nthread, get_max_nthread());
  shmem_size = nthread * sizeof(int);
  calcZoneMaxZColNatomKernel<<<izoneEnd - izoneStart + 1, nthread, shmem_size,
                               stream>>>(izoneStart, d_ZoneParam, col_natom,
                                         d_zoneMaxZColNatom);
  cudaCheck(cudaGetLastError());
  copy_DtoH<int>(d_zoneMaxZColNatom, h_zoneMaxZColNatom,
                 izoneEnd - izoneStart + 1, stream);
  cudaCheck(cudaEventRecord(zoneMaxZColNatom_copy_event, stream));

  //
  // Calculate positions in z columns
  // NOTE: Clears col_natom and sets (col_patom, cell_patom, col_ncellz,
  // d_NlistParam.ncell)
  //
  nthread = min(((ncol_tot - 1) / tilesize + 1) * tilesize, get_max_nthread());
  shmem_size = nthread * sizeof(int2);
  if (shmem_size > get_max_shmem_size()) {
     std::stringstream tmpexc; 
    tmpexc << "CudaNeighborListSort::sort_core, Device maximum reached: "
                 "shmem_size="
              << shmem_size << std::endl;
    throw std::invalid_argument(tmpexc.str());
    exit(1);
  }
  calc_z_column_pos_kernel<<<1, nthread, shmem_size, stream>>>(
      tilesize, ncol_tot, atomStart, cellStart, col_xy_zone, col_natom,
      col_patom, cell_patom + cellStart, col_ncellz + colStart,
      cell_xyz_zone + cellStart, col_cell + colStart, d_NlistParam);
  cudaCheck(cudaGetLastError());

  // This copying is done to get value of ncell and col_max_natom
  copy_DtoH<int>(&d_NlistParam->ncell, h_ncell, 1, stream);
  cudaCheck(cudaEventRecord(ncell_copy_event, stream));

  //
  // Reorder atoms into z-columns
  // NOTE: also sets up startcell_zone[izone]
  //
  nthread = 512;
  nblock = (ncoord_tot - 1) / nthread + 1;
  reorder_atoms_z_column_kernel<<<nblock, nthread, 0, stream>>>(
      ncoord_tot, atom_icol, col_natom, col_patom, xyzq + atomStart,
      xyzq_sorted + atomStart, ind_sorted + atomStart);
  cudaCheck(cudaGetLastError());

  // Test z columns
  if (test) {
    cudaCheck(cudaDeviceSynchronize());
    test_z_columns(zone_patom, h_ZoneParam, xyzq + atomStart,
                   xyzq_sorted + atomStart, col_patom, ind_sorted + atomStart);
  }

  // Wait until zoneMaxZColNatom is received on host
  cudaCheck(cudaEventSynchronize(zoneMaxZColNatom_copy_event));

  //
  // Now sort according to z coordinate. We sort each zone separately
  //
  int colStartZone = 0;
  for (int izone = izoneStart; izone <= izoneEnd; izone++) {
    int maxZColNatom = h_zoneMaxZColNatom[izone - izoneStart];
    int ncellxy = h_ZoneParam[izone].ncellx * h_ZoneParam[izone].ncelly;
    nblock = ncellxy;
    nthread = min(maxZColNatom, get_max_nthread());
    shmem_size = maxZColNatom * sizeof(keyval_t);
    // Amount of shared memory is the limiting factor here:
    // For a simple bitonic sort, values and keys must fit into shared
    // memory
    if (shmem_size <= get_max_shmem_size()) {
      // Bitonic sort in shared memory
      bitonicSortZColKernel<true><<<nblock, nthread, shmem_size, stream>>>(
          0, col_patom + colStartZone, xyzq_sorted + atomStart, xyzqTmp,
          ind_sorted + atomStart, NULL);
      cudaCheck(cudaGetLastError());
    } else {
#ifdef BUCKET_SORT_IN_USE
      // Bucket sort
      float min_z = h_ZoneParam[izone].min_xyz.z;
      float max_z = h_ZoneParam[izone].max_xyz.z;
      int ncoord = h_ZoneParam[izone].ncoord;
      int numBucket = maxZColNatom;
      float invBucketWidth = ((float)numBucket) / (max_z - min_z);
      reallocate<int>(&bucketPos, &bucketPosLen, numBucket * ncellxy, 1.5f);
      reallocate<int>(&bucketIndex, &bucketIndexLen, ncoord, 1.5f);
      reallocate<int>(&indSortedTmp, &indSortedTmpLen, ncoord, 1.5f);
      shmem_size = nthread * sizeof(int);
      bucketSortZColKernel<<<nblock, nthread, shmem_size, stream>>>(
          col_patom + colStartZone, min_z, invBucketWidth, numBucket, bucketPos,
          bucketIndex, xyzq_sorted + atomStart, xyzqTmp, ind_sorted + atomStart,
          indSortedTmp);
      cudaCheck(cudaGetLastError());
#else
      // Bitonic sort in global memory
      reallocate<keyval_t>(&keyvalBuffer, &keyvalBufferLen,
                           maxZColNatom * nblock, 1.5f);
      bitonicSortZColKernel<false><<<nblock, nthread, 0, stream>>>(
          maxZColNatom, col_patom + colStartZone, xyzq_sorted + atomStart,
          xyzqTmp, ind_sorted + atomStart, keyvalBuffer);
      cudaCheck(cudaGetLastError());
#endif
    }
    colStartZone += ncellxy;
  }

  /*
  nblock = 0;
  for (int izone=izoneStart;izone <= izoneEnd;izone++) {
    nblock += h_ZoneParam[izone].ncellx*h_ZoneParam[izone].ncelly;
  }
  nthread = min(col_max_natom, get_max_nthread());
  shmem_size = col_max_natom*sizeof(keyval_t);
  if (shmem_size > get_max_shmem_size()) {
    std::cerr << "CudaNeighborListSort::sort_core, Device maximum reached:
  shmem_size="
              << shmem_size << " nthread=" << nthread << std::endl;
    exit(1);
  } else {
    sort_z_column_kernel<<< nblock, nthread, shmem_size, stream >>>
      (col_patom, xyzq_sorted+atomStart, xyzqTmp, ind_sorted+atomStart);
    cudaCheck(cudaGetLastError());
  }
  */
}

//
// Builds indices etc. after sort. xyzq is the sorted array
//
void CudaNeighborListSort::sort_build_indices(
    const int *zone_patom, const int cellStart, int *cell_patom,
    const float4 *xyzq, int *loc2glo, int *glo2loc, int *ind_sorted, bb_t *bb,
    float *cell_bz, cudaStream_t stream) {
  int nthread, nblock, shmem_size;

  int atomStart = zone_patom[izoneStart];

  //
  // Build loc2glo (really we are reordering it with ind_sorted)
  //
  // Make a copy of loc2glo to a temporary array
  copy_DtoD<int>(loc2glo + atomStart, loc2gloTmp, ncoord_tot, stream);
  nthread = 512;
  nblock = (ncoord_tot - 1) / nthread + 1;
  reorder_loc2glo_kernel<<<nblock, nthread, 0, stream>>>(
      ncoord_tot, atomStart, ind_sorted + atomStart, loc2gloTmp,
      loc2glo + atomStart);
  cudaCheck(cudaGetLastError());

  // Build glo2loc
  nthread = 512;
  nblock = (ncoord_tot - 1) / nthread + 1;
  build_glo2loc_kernel<<<nblock, nthread, 0, stream>>>(
      ncoord_tot, atomStart, loc2glo + atomStart, glo2loc);
  cudaCheck(cudaGetLastError());

  /*
  // Build atom_pcell
  nthread = 1024;
  nblock = (ncell_max - 1)/(nthread/warpsize) + 1;
  build_atom_pcell_kernel<<< nblock, nthread, 0, stream >>>(cell_patom,
  atom_pcell);
  cudaCheck(cudaGetLastError());
  */

  // Wait until value of ncell is received on host
  cudaCheck(cudaEventSynchronize(ncell_copy_event));
  this->ncell = h_ncell[0];

  // Build bounding box (bb) and cell boundaries (cell_bz)
  nthread = 512;
  nblock = (ncell_max - 1) / nthread + 1;
  shmem_size = 0;
  calc_bb_cell_bz_kernel<<<nblock, nthread, shmem_size, stream>>>(
      ncell, cell_patom + cellStart, xyzq, bb + cellStart, cell_bz + cellStart);
  cudaCheck(cudaGetLastError());

  /*
  //
  // Calculate ncellz_max[izone]
  // NOTE: This is only needed in order to get a better estimate for
  n_tile_est
  //
  nthread = min(((max_ncellxy-1)/warpsize+1)*warpsize, get_max_nthread());
  nblock = maxNumZone;
  shmem_size = nthread*sizeof(int);
  calc_ncellz_max_kernel<<< nblock, nthread, shmem_size, stream
  >>>(col_ncellz);
  */
}

//
// Tests for z columns
//
bool CudaNeighborListSort::test_z_columns(
    const int *zone_patom, const ZoneParam_t *h_ZoneParam, const float4 *xyzq,
    const float4 *xyzq_sorted, const int *col_patom, const int *ind_sorted) {
  int atomStart = zone_patom[izoneStart];

  float4 *h_xyzq = new float4[ncoord_tot];
  copy_DtoH_sync<float4>(xyzq, h_xyzq, ncoord_tot);

  float4 *h_xyzq_sorted = new float4[ncoord_tot];
  copy_DtoH_sync<float4>(xyzq_sorted, h_xyzq_sorted, ncoord_tot);

  int *h_col_patom = new int[ncol_tot + 1];
  copy_DtoH_sync<int>(col_patom, h_col_patom, ncol_tot + 1);

  int *h_ind_sorted = new int[ncoord_tot];
  copy_DtoH_sync<int>(ind_sorted, h_ind_sorted, ncoord_tot);

  bool ok = true;

  int ind0 = 0;
  int prev_ind = 0;
  for (int izone = izoneStart; izone <= izoneEnd; izone++) {
    int istart = zone_patom[izone] - atomStart;
    int iend = zone_patom[izone + 1] - 1 - atomStart;
    if (iend >= istart) {
      float x0 = h_ZoneParam[izone].min_xyz.x;
      float y0 = h_ZoneParam[izone].min_xyz.y;
      for (int i = istart; i <= iend; i++) {
        float x = h_xyzq_sorted[i].x;
        float y = h_xyzq_sorted[i].y;
        int ix = (int)((x - x0) * h_ZoneParam[izone].inv_celldx);
        int iy = (int)((y - y0) * h_ZoneParam[izone].inv_celldy);
        int ind = ind0 + ix + iy * h_ZoneParam[izone].ncellx;
        // Check that the column indices are in increasing order
        if (ind < prev_ind) {
          std::cout << "test_z_columns FAILED at i=" << i + atomStart
                    << " prev_ind=" << prev_ind << " ind=" << ind << std::endl;
          exit(1);
        }
        prev_ind = ind;
        int lo_ind = h_col_patom[ind];
        int hi_ind = h_col_patom[ind + 1] - 1;
        if (i < lo_ind || i > hi_ind) {
          std::cout << "test_z_columns FAILED at i=" << i + atomStart
                    << " izone = " << izone << std::endl;
          std::cout << "ind, lo_ind, hi_ind = " << ind << " " << lo_ind << " "
                    << hi_ind << std::endl;
          std::cout << "x,y = " << x << " " << y << " x0,y0 = " << x0 << " "
                    << y0 << std::endl;
          std::cout << "inv_celldx/y = " << h_ZoneParam[izone].inv_celldx << " "
                    << h_ZoneParam[izone].inv_celldy << std::endl;
          std::cout << "ix,iy =" << ix << " " << iy << " ind0 = " << ind0
                    << " ncellx = " << h_ZoneParam[izone].ncellx
                    << " ncelly = " << h_ZoneParam[izone].ncelly << std::endl;
          exit(1);
        }
      }
      for (int i = istart; i <= iend; i++) {
        int j = h_ind_sorted[i];
        float x = h_xyzq_sorted[i].x;
        float y = h_xyzq_sorted[i].y;
        float xj = h_xyzq[j].x;
        float yj = h_xyzq[j].y;
        if (x != xj || y != yj) {
          std::cout << "test_z_columns FAILED at i=" << i + atomStart
                    << std::endl;
          std::cout << "x,y   =" << x << " " << y << std::endl;
          std::cout << "xj,yj =" << xj << " " << yj << std::endl;
          exit(1);
        }
      }
      ind0 += h_ZoneParam[izone].ncellx * h_ZoneParam[izone].ncelly;
    }
  }

  if (ok)
    std::cout << "test_z_columns OK" << std::endl;

  delete[] h_xyzq;
  delete[] h_xyzq_sorted;
  delete[] h_col_patom;
  delete[] h_ind_sorted;

  return ok;
}

//
// Tests sort
//
bool CudaNeighborListSort::test_sort(const int *zone_patom, const int cellStart,
                                     const ZoneParam_t *h_ZoneParam,
                                     const float4 *xyzq,
                                     const float4 *xyzq_sorted,
                                     const int *ind_sorted,
                                     const int *cell_patom) {
  cudaCheck(cudaDeviceSynchronize());

  int atomStart = zone_patom[izoneStart];

  float4 *h_xyzq = new float4[ncoord_tot];
  copy_DtoH_sync<float4>(xyzq + atomStart, h_xyzq, ncoord_tot);

  float4 *h_xyzq_sorted = new float4[ncoord_tot];
  copy_DtoH_sync<float4>(xyzq_sorted + atomStart, h_xyzq_sorted, ncoord_tot);

  int *h_col_patom = new int[ncol_tot + 1];
  copy_DtoH_sync<int>(col_patom, h_col_patom, ncol_tot + 1);

  int *h_ind_sorted = new int[ncoord_tot];
  copy_DtoH_sync<int>(ind_sorted + atomStart, h_ind_sorted, ncoord_tot);

  int *h_cell_patom = new int[ncell];
  copy_DtoH_sync<int>(cell_patom + cellStart, h_cell_patom, ncell);

  bool ok = true;

  int izone, i, j, k, prev_ind;
  float x, y, z, prev_z;
  float xj, yj, zj;
  int ix, iy, ind, lo_ind, hi_ind;

  k = 0;
  // Loop through columns
  for (i = 1; i < ncol_tot + 1; i++) {
    // Loop through cells
    for (j = h_col_patom[i - 1]; j < h_col_patom[i]; j += tilesize) {
      if (j + atomStart != h_cell_patom[k]) {
        std::cout << "test_sort FAILED at i=" << i << " k=" << k
                  << " atomStart=" << atomStart << " cellStart=" << cellStart
                  << " ncell=" << ncell << std::endl;
        std::cout << "j+atomStart=" << j + atomStart
                  << " cell_patom[k]=" << h_cell_patom[k] << std::endl;
        exit(1);
      }
      k++;
    }
  }
  int ind0 = 0;
  for (izone = izoneStart; izone <= izoneEnd; izone++) {
    int istart = zone_patom[izone] - atomStart;
    int iend = zone_patom[izone + 1] - 1 - atomStart;
    if (iend >= istart) {
      float x0 = h_ZoneParam[izone].min_xyz.x;
      float y0 = h_ZoneParam[izone].min_xyz.y;
      prev_z = h_ZoneParam[izone].min_xyz.z;
      prev_ind = ind0;
      for (i = istart; i <= iend; i++) {
        x = h_xyzq_sorted[i].x;
        y = h_xyzq_sorted[i].y;
        z = h_xyzq_sorted[i].z;

        ix = (int)((x - x0) * h_ZoneParam[izone].inv_celldx);
        iy = (int)((y - y0) * h_ZoneParam[izone].inv_celldy);
        ind = ind0 + ix + iy * h_ZoneParam[izone].ncellx;

        if (prev_ind != ind) {
          prev_z = h_ZoneParam[izone].min_xyz.z;
        }

        lo_ind = h_col_patom[ind];
        hi_ind = h_col_patom[ind + 1] - 1;
        if (i < lo_ind || i > hi_ind) {
          std::cout << "test_sort FAILED at i=" << i << " ind=" << ind
                    << " istart=" << istart << " iend=" << iend
                    << " lo_ind= " << lo_ind << " hi_ind=" << hi_ind
                    << std::endl;
          std::cout << "x=" << x << " x0=" << x0 << " y=" << x << " y0=" << y0
                    << " z=" << z << " ix=" << ix << " iy=" << iy << std::endl;
          std::cout << "celldx=" << h_ZoneParam[izone].celldx
                    << " celldy=" << h_ZoneParam[izone].celldy
                    << " atomStart=" << atomStart
                    << " ncoord_tot=" << ncoord_tot << std::endl;
          exit(1);
        }
        if (z < prev_z) {
          std::cout << "test_sort FAILED at i=" << i << std::endl;
          std::cout << "prev_z, z = " << prev_z << " " << z << std::endl;
          exit(1);
        }
        prev_z = z;
        prev_ind = ind;
      }

      for (i = istart; i <= iend; i++) {
        j = h_ind_sorted[i] - atomStart;
        x = h_xyzq_sorted[i].x;
        y = h_xyzq_sorted[i].y;
        z = h_xyzq_sorted[i].z;
        xj = h_xyzq[j].x;
        yj = h_xyzq[j].y;
        zj = h_xyzq[j].z;
        if (x != xj || y != yj || z != zj) {
          std::cout << "test_sort FAILED at i=" << i + atomStart << std::endl;
          std::cout << "x,y,z   =" << x << " " << y << " " << z << std::endl;
          std::cout << "xj,yj,zj=" << xj << " " << yj << " " << zj << std::endl;
          exit(1);
        }
      }
      ind0 += h_ZoneParam[izone].ncellx * h_ZoneParam[izone].ncelly;
    }
  }

  if (ok)
    std::cout << "test_sort OK" << std::endl;

  delete[] h_xyzq;
  delete[] h_xyzq_sorted;
  delete[] h_col_patom;
  delete[] h_cell_patom;
  delete[] h_ind_sorted;

  return ok;
}

/*
//
// Sorts atoms. Assumes h_ZoneParam is already setup correctly
//
void CudaNeighborListSort::sort(const int *zone_patom,
                                ZoneParam_t *h_ZoneParam, ZoneParam_t
*d_ZoneParam,
                                float4 *xyzq,
                                float4 *xyzq_sorted,
                                int *loc2glo,
                                cudaStream_t stream) {
  const int ncoord = zone_patom[maxNumZone];
  assert(ncoord <= max_ncoord);

  if (ncoord > topExcl.get_ncoord()) {
    std::cerr << "CudaNeighborList::sort(1), Invalid value for ncoord" <<
std::endl;
    exit(1);
  }

  // -------------------------- Setup -----------------------------
  //sort_setup(zone_patom, ncol_tot, stream);
  // --------------------------------------------------------------

  // ------------------ Allocate/Reallocate memory ----------------
  //sort_alloc_realloc(ncoord);
  // --------------------------------------------------------------

  // ---------------------- Do actual sorting ---------------------
  sort_core(ncoord, cell_patom, col_ncellz, cell_xyz_zone, xyzq, xyzq_sorted,
stream);
  // --------------------------------------------------------------

  // ------------------ Build indices etc. after sort -------------
  sort_build_indices(ncoord, xyzq_sorted, loc2glo, stream);
  // --------------------------------------------------------------

  // Test sort
  if (test) {
    test_sort(h_NlistParam->zone_patom, h_NlistParam->ncellx,
h_NlistParam->ncelly,
              ncell_max, h_NlistParam->min_xyz,
              h_NlistParam->inv_celldx, h_NlistParam->inv_celldy,
              xyzq, xyzq_sorted, col_patom, cell_patom);
  }

}
*/

/*
//
// Sorts atoms, when minimum and maximum coordinate values are known
//
void CudaNeighborListSort::sort(const int *zone_patom,
                                const float3 *max_xyz, const float3 *min_xyz,
                                float4 *xyzq,
                                float4 *xyzq_sorted,
                                int *loc2glo,
                                cudaStream_t stream) {
  int ncoord = zone_patom[maxNumZone];
  assert(ncoord <= max_ncoord);
  int ncol_tot;

  if (ncoord > topExcl.get_ncoord()) {
    std::cerr << "CudaNeighborList::sort(1), Invalid value for ncoord" <<
std::endl;
    exit(1);
  }

  for (int izone=izoneStart;izone <= izoneEnd;izone++) {
    h_ZoneParam[izone].min_xyz = min_xyz[izone];
    h_ZoneParam[izone].max_xyz = max_xyz[izone];
  }

  // -------------------------- Setup -----------------------------
  sort_setup(izoneStart, izoneEnd, zone_patom, ncol_tot, stream);
  // --------------------------------------------------------------

  // ------------------ Allocate/Reallocate memory ----------------
  sort_alloc_realloc(ncol_tot, ncoord);
  // --------------------------------------------------------------

  // ---------------------- Do actual sorting ---------------------
  sort_core(ncol_tot, ncoord, xyzq, xyzq_sorted, stream);
  // --------------------------------------------------------------

  // ------------------ Build indices etc. after sort -------------
  sort_build_indices(ncoord, xyzq_sorted, loc2glo, stream);
  // --------------------------------------------------------------

  // Test sort
  if (test) {
    test_sort(h_NlistParam->zone_patom, h_NlistParam->ncellx,
h_NlistParam->ncelly,
              ncol_tot, ncell_max, min_xyz, h_NlistParam->inv_celldx,
h_NlistParam->inv_celldy,
              xyzq, xyzq_sorted, col_patom, cell_patom);
  }

}
*/
#endif // NOCUDAC
