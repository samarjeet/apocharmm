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

#include "CudaNeighborListBuild.h"
#include "CudaNeighborListStruct.h"
//#include "gpu_utils.h"
#include <cuda_runtime.h>

class CudaP21NeighborListBuild {
public:
  CudaP21NeighborListBuild();
  void build(const int ncell, const int maxNumExcl,
             const ZoneParam_t *__restrict__ h_zoneParam,
             const ZoneParam_t *__restrict__ d_zoneParam, const float boxx,
             const float boxy, const float boxz, const float rcut,
             const float4 *xyzq, const int *loc2glo, const int *glo2loc,
             const int *atomExclPos, const int *atomExcl,
             const int4 *cell_xyz_zone, const int *col_ncellz,
             const int *col_cell, const float *cell_bz, const int *cell_patom,
             const bb_t *bb, NlistParam_t *h_NlistParam,
             NlistParam_t *d_NlistParam, cudaStream_t stream);

  int get_n_ientry() { return n_ientry; }
  int get_n_ientry_est() { return n_ientry_est; }
  ientry_t *get_ientry() { return ientry; }
  int get_n_tile() { return n_tile; }
  int get_n_tile_est() { return n_tile_est; }
  int *get_tile_indj() { return tile_indj; }
  tile_excl_t<32> *get_tile_excl() { return tile_excl; }

  int get_n_ientry() const { return n_ientry; }
  int get_n_ientry_est() const { return n_ientry_est; }
  const ientry_t *get_ientry() const { return ientry; }
  int get_n_tile() const { return n_tile; }
  int get_n_tile_est() const { return n_tile_est; }
  const int *get_tile_indj() const { return tile_indj; }
  const tile_excl_t<32> *get_tile_excl() const { return tile_excl; }
private:
  int imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi;

  // Number of i tiles
  int n_ientry;

  // Total number of tiles
  int n_tile;

  // Estimates to number of tiles
  int n_ientry_est;
  int n_tile_est;

  int tile_excl_len;
  tile_excl_t<32> *tile_excl;

  int ientry_raw_len;
  ientry_t *ientry_raw;

  int ientry_len;
  ientry_t *ientry;

  int tile_indj_len;
  int *tile_indj;

  // Atom-Atom exclusion heap
  int exclAtomHeapLen;
  int *exclAtomHeap;

  // ------------------------------------
  // Used for sorting ientry:
  // bucketPos[n_jlist_max+1]
  int bucketPosLen;
  int *bucketPos;

  // bucketIndex[n_ientry]
  int bucketIndexLen;
  int *bucketIndex;

  void estimateIentry(const ZoneParam_t *__restrict__ h_zoneParam, float rcut);
};
