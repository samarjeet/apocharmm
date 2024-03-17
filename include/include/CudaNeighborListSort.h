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
#ifndef CUDANEIGHBORLISTSORT_H
#define CUDANEIGHBORLISTSORT_H
//
// Neighbor list sort class
//
// (c) Antti-Pekka Hynninen 2014
// aphynninen@hotmail.com
//
#include "CudaNeighborListStruct.h"
#include <cuda.h>

struct keyval_t {
  union {
    float key;
    int ind;
  };
  int val;
};

class CudaNeighborListSort {
private:
  // Disable copy constructor
  CudaNeighborListSort(const CudaNeighborListSort &);

  // Tile size
  int tilesize;

  // zone range for this sort
  int izoneStart;
  int izoneEnd;

  // Total number of columns
  int ncol_tot;

  // Max. estimate for the total number of cells
  int ncell_max;

  // Total number of cells
  int ncell;

  // Maximum number of atoms in all columns
  int col_max_natom;

  // Total number of coordinates
  int ncoord_tot;

  // Number of atoms in each column
  int col_natom_len;
  int *col_natom;

  // Cumulative number of atoms in each column
  int col_patom_len;
  int *col_patom;

  // x and y coordinates and zone of each column
  int col_xy_zone_len;
  int3 *col_xy_zone;

  // Column index of each atom
  int atom_icol_len;
  int *atom_icol;

  // Temporary array used for constructing loc2glo
  int loc2gloTmp_len;
  int *loc2gloTmp;

  // Temporary array used for constructing xyzq
  int xyzqTmpLen;
  float4 *xyzqTmp;

#ifdef BUCKET_SORT_IN_USE
  //-------------------------------------------------
  // These are required by the z-column bucket sort
  int bucketPosLen;
  int *bucketPos;

  int bucketIndexLen;
  int *bucketIndex;

  int indSortedTmpLen;
  int *indSortedTmp;
//-------------------------------------------------
#else
  int keyvalBufferLen;
  keyval_t *keyvalBuffer;
#endif

  // Pinned memory host-buffer for ncell (single value)
  int *h_ncell;

  // Pinned memory host and device buffers for zoneMaxZColNatom
  // (izoneEnd-izoneStart+1 values)
  int *h_zoneMaxZColNatom;
  int *d_zoneMaxZColNatom;

  // Events
  cudaEvent_t ncell_copy_event;
  cudaEvent_t zoneMaxZColNatom_copy_event;

  // Flag for testing neighborlist build
  bool test;

  bool test_z_columns(const int *zone_patom, const ZoneParam_t *h_ZoneParam,
                      const float4 *xyzq, const float4 *xyzq_sorted,
                      const int *col_patom, const int *ind_sorted);

  void setZoneParam(ZoneParam_t *h_ZoneParam, ZoneParam_t *d_ZoneParam,
                    cudaStream_t stream);

  void getZoneParam(ZoneParam_t *h_ZoneParam, ZoneParam_t *d_ZoneParam,
                    cudaStream_t stream);

  // void set_NlistParam(cudaStream_t stream);
  // void get_NlistParam();

public:
  CudaNeighborListSort(const int tilesize, const int izoneStart,
                       const int izoneEnd);
  ~CudaNeighborListSort();

  void calc_min_max_xyz(const int *zone_patom, const float4 *xyzq,
                        ZoneParam_t *h_ZoneParam, ZoneParam_t *d_ZoneParam,
                        cudaStream_t stream);

  void sort_setup(const int *zone_patom, ZoneParam_t *h_ZoneParam,
                  ZoneParam_t *d_ZoneParam, cudaStream_t stream);

  void sort_realloc();

  void sort_core(const int *zone_patom, const int cellStart, const int colStart,
                 ZoneParam_t *h_ZoneParam, ZoneParam_t *d_ZoneParam,
                 NlistParam_t *d_NlistParam, int *cell_patom, int *col_ncellz,
                 int4 *cell_xyz_zone, int *col_cell, int *ind_sorted,
                 const float4 *xyzq, float4 *xyzq_sorted, cudaStream_t stream);

  void sort_build_indices(const int *zone_patom, const int cellStart,
                          int *cell_patom, const float4 *xyzq, int *loc2glo,
                          int *glo2loc, int *ind_sorted, bb_t *bb,
                          float *cell_bz, cudaStream_t stream);

  bool test_sort(const int *zone_patom, const int cellStart,
                 const ZoneParam_t *h_ZoneParam, const float4 *xyzq,
                 const float4 *xyzq_sorted, const int *ind_sorted,
                 const int *cell_patom);

  // void reset();

  void set_test(bool test_in) { this->test = test_in; }

  // int getIzoneStart() {return izoneStart;}
  // int getIzoneEnd() {return izoneEnd;}

  int get_ncol_tot() { return ncol_tot; }
  int get_ncoord_tot() { return ncoord_tot; }
  int get_ncell_max() { return ncell_max; }
  int get_ncell() { return ncell; }

  /*
  void sort(const int *zone_patom,
            const float3 *max_xyz, const float3 *min_xyz,
            float4 *xyzq,
            float4 *xyzq_sorted,
            int *loc2glo,
            cudaStream_t stream=0);

  void sort(const int *zone_patom,
            float4 *xyzq,
            float4 *xyzq_sorted,
            int *loc2glo,
            cudaStream_t stream=0);
  */
};

#endif // CUDANEIGHBORLISTSORT_H
#endif // NOCUDAC
