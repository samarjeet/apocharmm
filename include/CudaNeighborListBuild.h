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
#ifndef CUDANEIGHBORLISTBUILD_H
#define CUDANEIGHBORLISTBUILD_H
//
// Neighbor list build class
//
// (c) Antti-Pekka Hynninen 2014
// aphynninen@hotmail.com
//
#include "CudaNeighborListStruct.h"
#include <cuda.h>
#include <iostream>
#include <vector>

template <int tilesize> struct num_excl {
  static const int val = ((tilesize * tilesize - 1) / 32 + 1);
};

template <int tilesize> struct tile_excl_t {
  unsigned int excl[num_excl<tilesize>::val]; // Exclusion mask
};

struct ientry_t {
  int iatomStart;
  int ish;
  int tileStart;
  int tileEnd;
};

#ifdef USE_SPARSE
template <int tilesize> struct pairs_t { int i[tilesize]; };
#endif

template <int tilesize> class CudaNeighborListBuild {
private:
  // Disable copy constructor
  CudaNeighborListBuild(const CudaNeighborListBuild &);

  bool q_p21;

  // zone range for this sort
  int izoneStart;
  int izoneEnd;

  // Number of i tiles
  int n_ientry;

  // Total number of tiles
  int n_tile;

  // Estimates to number of tiles
  int n_ientry_est;
  int n_tile_est;

  int tile_excl_len;
  tile_excl_t<tilesize> *tile_excl;

  // Raw un-sorted ientry
  int ientry_raw_len;
  ientry_t *ientry_raw;

  // Final sorted ientry
  int ientry_len;
  ientry_t *ientry;

  int tile_indj_len;
  int *tile_indj;

#ifdef USE_SPARSE
  // Sparse:
  int n_ientry_sparse;
  int n_tile_sparse;

  int pairs_len;
  pairs_t<tilesize> *pairs;

  int ientry_sparse_len;
  ientry_t *ientry_sparse;

  int tile_indj_sparse_len;
  int *tile_indj_sparse;
#endif

  // Approximate upper bound for number of cells
  // int ncell_max;

  // Maximum value of n_int_zone[]
  int n_int_zone_max;

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
  // ------------------------------------

  // Flag for testing neighborlist build
  bool test;

  void calc_tile_ientry_est(const ZoneParam_t *h_ZoneParam, const float rcut);

  template <typename T>
  int calc_gpu_pairlist(const int n_ientry, const ientry_t *ientry,
                        const int *tile_indj,
                        const tile_excl_t<tilesize> *tile_excl,
                        const float4 *xyzq, const double boxx,
                        const double boxy, const double boxz,
                        const double rcut);

  template <typename T>
  int calc_cpu_pairlist(const int *zone_patom, const float4 *xyzq,
                        const int *loc2glo, const int *atom_excl_pos,
                        const int *atom_excl, const double boxx,
                        const double boxy, const double boxz,
                        const double rcut);

  // void set_NlistParam(cudaStream_t stream);
  // void get_NlistParam();

  void init();
  void load(const char *filename);

public:
  CudaNeighborListBuild(const int n_int_zone_max, const int izoneStart,
                        const int izoneEnd, const bool q_p21_);
  CudaNeighborListBuild(const int n_int_zone_max, const int izoneStart,
                        const int izoneEnd, const char *filename);
  ~CudaNeighborListBuild();

  void build(const int ncell, const int cellStart, const int maxNumExcl,
             const ZoneParam_t *h_ZoneParam, const ZoneParam_t *d_ZoneParam,
             const float boxx, const float boxy, const float boxz,
             const float rcut, const float4 *xyzq, const int *loc2glo,
             const int *glo2loc, const int *atomExclPos, const int *atomExcl,
             const int4 *cell_xyz_zone, const int *col_ncellz,
             const int *col_cell, const float *cell_bz, const int *cell_patom,
             const bb_t *bb, NlistParam_t *h_NlistParam,
             NlistParam_t *d_NlistParam, cudaStream_t stream);

  void test_build(const int *zone_patom, const int ncol_tot,
                  const int ncell_tot, const ZoneParam_t *h_ZoneParam,
                  const int atomExclPosLen, const int *atomExclPos,
                  const int atomExclLen, const int *atomExcl, const double boxx,
                  const double boxy, const double boxz, const double rcut,
                  const float4 *xyzq, const int *loc2glo, const int *glo2loc,
                  const int ncoord_glo, const int *cell_patom,
                  const int *col_cell, const float *cell_bz, const bb_t *bb);
  // void reset();

  void build_excl(const float boxx, const float boxy, const float boxz,
                  const float rcut, const int n_ijlist, const int3 *ijlist,
                  const int *cell_patom, const float4 *xyzq,
                  cudaStream_t stream = 0);

  void add_tile_top(const int ntile_top, const int *tile_ind_top,
                    const tile_excl_t<tilesize> *tile_excl_top,
                    cudaStream_t stream = 0);

  void set_ientry(int n_ientry, ientry_t *h_ientry, cudaStream_t stream);

#ifdef SPARSE
  void split_dense_sparse(int npair_cutoff);
#endif
  void remove_empty_tiles();
  void analyze();

  void set_test(bool test_in) { this->test = test_in; }

  int get_n_ientry() { return n_ientry; }
  int get_n_ientry_est() { return n_ientry_est; }
  ientry_t *get_ientry() { return ientry; }
  int get_n_tile() { return n_tile; }
  int get_n_tile_est() { return n_tile_est; }
  int *get_tile_indj() { return tile_indj; }
  tile_excl_t<tilesize> *get_tile_excl() { return tile_excl; }

  int get_n_ientry() const { return n_ientry; }
  int get_n_ientry_est() const { return n_ientry_est; }
  const ientry_t *get_ientry() const { return ientry; }
  int get_n_tile() const { return n_tile; }
  int get_n_tile_est() const { return n_tile_est; }
  const int *get_tile_indj() const { return tile_indj; }
  const tile_excl_t<tilesize> *get_tile_excl() const { return tile_excl; }
};

#endif // CUDANEIGHBORLISTBUILD_H
#endif // NOCUDAC
