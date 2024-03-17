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
#include "CudaNeighborList.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

// IF defined, uses strict (Factor = 1.0f) memory reallocation. Used for
// debuggin memory problems.
#define STRICT_MEMORY_REALLOC

// static const int numNlistParam=2;
// static __device__ NeighborListParam_t d_NlistParam[numNlistParam];

// static __device__ ZoneParam_t d_ZoneParam[maxNumZone];

//########################################################################################
//########################################################################################
//########################################################################################

//
// Dummy constructo
//
/*
template <int tilesize>
CudaNeighborList<tilesize>::CudaNeighborList(){

}
*/

//
// Class creator
//
template <int tilesize>
CudaNeighborList<tilesize>::CudaNeighborList(const CudaTopExcl &topExcl,
                                             const int nx, const int ny,
                                             const int nz)
    : topExcl(topExcl) {
  numZone = 0;

  numList = 0;

  ind_sorted_len = 0;
  ind_sorted = NULL;

  col_ncellz_len = 0;
  col_ncellz = NULL;

  col_cell_len = 0;
  col_cell = NULL;

  cell_patom_len = 0;
  cell_patom = NULL;

  cell_xyz_zone_len = 0;
  cell_xyz_zone = NULL;

  cell_bz_len = 0;
  cell_bz = NULL;

  bb_len = 0;
  bb = NULL;

  d_ZoneParam_len = 0;
  d_ZoneParam = NULL;

  h_ZoneParam_len = 0;
  h_ZoneParam = NULL;

  imx_lo = 0;
  imx_hi = 0;
  imy_lo = 0;
  imy_hi = 0;
  imz_lo = 0;
  imz_hi = 0;
  if (nx == 1) {
    imx_lo = -1;
    imx_hi = 1;
  }
  if (ny == 1) {
    imy_lo = -1;
    imy_hi = 1;
  }
  if (nz == 1) {
    imz_lo = -1;
    imz_hi = 1;
  }

  cudaCheck(cudaEventCreate(&glo2loc_reset_event));

  test = false;
}

//
// Class destructor
//
template <int tilesize> CudaNeighborList<tilesize>::~CudaNeighborList() {
  // Neighbor list building
  if (ind_sorted != NULL)
    deallocate<int>(&ind_sorted);
  if (cell_patom != NULL)
    deallocate<int>(&cell_patom);
  if (col_ncellz != NULL)
    deallocate<int>(&col_ncellz);
  if (col_cell != NULL)
    deallocate<int>(&col_cell);
  if (cell_xyz_zone != NULL)
    deallocate<int4>(&cell_xyz_zone);
  if (cell_bz != NULL)
    deallocate<float>(&cell_bz);
  if (bb != NULL)
    deallocate<bb_t>(&bb);
  for (int i = 0; i < d_NlistParam.size(); i++) {
    deallocate<NlistParam_t>(&d_NlistParam.at(i));
    deallocate_host<NlistParam_t>(&h_NlistParam.at(i));
  }
  for (int i = 0; i < sorter.size(); i++) {
    delete sorter.at(i);
    delete builder.at(i);
  }
  if (h_ZoneParam != NULL)
    deallocate_host<ZoneParam_t>(&h_ZoneParam);
  if (d_ZoneParam != NULL)
    deallocate<ZoneParam_t>(&d_ZoneParam);
  for (int i = 0; i < build_event.size(); i++) {
    cudaCheck(cudaEventDestroy(build_event.at(i)));
  }
  cudaCheck(cudaEventDestroy(glo2loc_reset_event));
}

//
// Register List
//
template <int tilesize>
void CudaNeighborList<tilesize>::registerList(
    std::vector<int> &numIntZone, std::vector<std::vector<int>> &intZones,
    const char *filename) {
  assert(numIntZone.size() == intZones.size());
  assert(numIntZone.size() <= maxNumZone);

  numList++;

  // Get izoneStart and izoneEnd
  int izoneStart = numZone + 1;
  int izoneEnd = -1;
  for (int izone = 0; izone < numIntZone.size(); izone++) {
    if (numIntZone.at(izone) > 0) {
      izoneStart = min(izoneStart, izone);
      izoneEnd = max(izoneEnd, izone);
    }
  }
  numZone = max(numZone, izoneEnd + 1);

  resize_host<ZoneParam_t>(&h_ZoneParam, &h_ZoneParam_len, h_ZoneParam_len,
                           numZone, 1.0f);
  resize<ZoneParam_t>(&d_ZoneParam, &d_ZoneParam_len, d_ZoneParam_len, numZone,
                      1.0f);

  // ----------------------
  // Setup h_ZoneParam
  // ----------------------
  int n_int_zone_max = 0;
  for (int izone = izoneStart; izone <= izoneEnd; izone++) {
    h_ZoneParam[izone].n_int_zone = numIntZone.at(izone);
    if (h_ZoneParam[izone].n_int_zone > 0) {
      assert(intZones.at(izone).size() <= maxNumIntZone);
      std::copy(intZones.at(izone).begin(), intZones.at(izone).end(),
                h_ZoneParam[izone].int_zone);
      n_int_zone_max = max(n_int_zone_max, h_ZoneParam[izone].n_int_zone);
    }
  }

  // Create list sorter and builder
  sorter.push_back(new CudaNeighborListSort(tilesize, izoneStart, izoneEnd));
  if (filename != NULL) {
    builder.push_back(new CudaNeighborListBuild<tilesize>(
        n_int_zone_max, izoneStart, izoneEnd, filename));
  } else {
    builder.push_back(new CudaNeighborListBuild<tilesize>(
        n_int_zone_max, izoneStart, izoneEnd));
  }

  // Create new NlistParam
  NlistParam_t *d_tmp;
  allocate<NlistParam_t>(&d_tmp, 1);
  NlistParam_t *h_tmp;
  allocate_host<NlistParam_t>(&h_tmp, 1);
  d_NlistParam.push_back(d_tmp);
  h_NlistParam.push_back(h_tmp);

  int indList = numList - 1;

  h_NlistParam.at(indList)->imx_lo = imx_lo;
  h_NlistParam.at(indList)->imx_hi = imx_hi;
  h_NlistParam.at(indList)->imy_lo = imy_lo;
  h_NlistParam.at(indList)->imy_hi = imy_hi;
  h_NlistParam.at(indList)->imz_lo = imz_lo;
  h_NlistParam.at(indList)->imz_hi = imz_hi;

  copy_HtoD_sync<NlistParam_t>(h_NlistParam.at(indList),
                               d_NlistParam.at(indList), 1);

  // Create events
  build_event.resize(numList);
  cudaCheck(cudaEventCreate(&build_event.at(indList)));
}

//
// Sort
//
template <int tilesize>
void CudaNeighborList<tilesize>::sort(const int indList, const int *zone_patom,
                                      const float4 *xyzq, float4 *xyzq_sorted,
                                      int *loc2glo, cudaStream_t stream) {
  assert(indList >= 0 && indList < numList);

  // Reset glo2loc or wait for the reset to be done
  if (indList == 0) {
    set_gpu_array<int>(topExcl.get_glo2loc(), topExcl.get_ncoord(), -1, stream);
    cudaCheck(cudaEventRecord(glo2loc_reset_event, stream));
  } else {
    cudaCheck(cudaStreamWaitEvent(stream, glo2loc_reset_event, 0));
  }

  // ------------------------ min/max -----------------------------
  sorter.at(indList)->calc_min_max_xyz(zone_patom, xyzq, h_ZoneParam,
                                       d_ZoneParam, stream);
  // --------------------------------------------------------------

  // -------------------------- Setup -----------------------------
  // NOTE: After this call, ncol_tot, ncoord_tot, and ncell_max are up to date
  sorter.at(indList)->sort_setup(zone_patom, h_ZoneParam, d_ZoneParam, stream);
  // --------------------------------------------------------------

  // ------------------------ Realloc -----------------------------
  sort_realloc(indList);
  // --------------------------------------------------------------

  // ---------------------- Do actual sorting ---------------------
  int cellStart = 0;
  int colStart = 0;
  for (int i = 0; i < indList; i++) {
    cellStart += sorter.at(i)->get_ncell();
    colStart += sorter.at(i)->get_ncol_tot();
  }
  sorter.at(indList)->sort_core(zone_patom, cellStart, colStart, h_ZoneParam,
                                d_ZoneParam, d_NlistParam.at(indList),
                                cell_patom, col_ncellz, cell_xyz_zone, col_cell,
                                ind_sorted, xyzq, xyzq_sorted, stream);
  // --------------------------------------------------------------

  // ------------------ Build indices etc. after sort -------------
  // NOTE: After this call, ncell is up to date
  sorter.at(indList)->sort_build_indices(
      zone_patom, cellStart, cell_patom, xyzq_sorted, loc2glo,
      topExcl.get_glo2loc(), ind_sorted, bb, cell_bz, stream);
  // --------------------------------------------------------------

  // Test sort
  if (test) {
    sorter.at(indList)->test_sort(zone_patom, cellStart, h_ZoneParam, xyzq,
                                  xyzq_sorted, ind_sorted, cell_patom);
  }
}

//
// Allocates / Re-allocates memory for sort
//
template <int tilesize>
void CudaNeighborList<tilesize>::sort_realloc(const int indList) {
  sorter.at(indList)->sort_realloc();

#ifdef STRICT_MEMORY_REALLOC
  float fac = 1.0f;
#else
  float fac = 1.2f;
#endif

  // Compute ncol_tot, ncoord_tot, ncell_max.
  // NOTE: these variables are quaranteed to be up to date because
  //       sort is only launched after launching the previous build
  // Current content
  int ncol_cur = 0;
  int ncoord_cur = 0;
  int ncell_cur = 0;
  for (int i = 0; i < indList; i++) {
    ncol_cur += sorter.at(i)->get_ncol_tot();
    ncoord_cur += sorter.at(i)->get_ncoord_tot();
    // NOTE: here we're using ncell instead of ncell_max, because this value
    //       from previous launch is now up to date
    ncell_cur += sorter.at(i)->get_ncell();
  }
  // Total
  int ncol_tot = ncol_cur + sorter.at(indList)->get_ncol_tot();
  int ncoord_tot = ncoord_cur + sorter.at(indList)->get_ncoord_tot();
  int ncell_tot = ncell_cur + sorter.at(indList)->get_ncell_max();

  bool q_resize = (cell_patom_len < ncell_tot + 1) ||
                  (cell_xyz_zone_len < ncell_tot) ||
                  (cell_bz_len < ncell_tot) || (bb_len < ncell_tot) ||
                  (col_ncellz_len < ncol_tot) || (col_cell_len < ncol_tot) ||
                  (ind_sorted_len < ncoord_tot);

  if (q_resize) {
    // We must wait here until all preceding neighbor list builds have
    // finished before reallocating these arrays.
    for (int i = 0; i < indList; i++) {
      cudaCheck(cudaEventSynchronize(build_event.at(i)));
    }

    resize<int>(&cell_patom, &cell_patom_len, ncell_cur + 1, ncell_tot + 1,
                fac);
    resize<int4>(&cell_xyz_zone, &cell_xyz_zone_len, ncell_cur, ncell_tot, fac);
    resize<float>(&cell_bz, &cell_bz_len, ncell_cur, ncell_tot, fac);
    resize<bb_t>(&bb, &bb_len, ncell_cur, ncell_tot, fac);

    resize<int>(&col_ncellz, &col_ncellz_len, ncol_cur, ncol_tot, fac);
    resize<int>(&col_cell, &col_cell_len, ncol_cur, ncol_tot, fac);

    resize<int>(&ind_sorted, &ind_sorted_len, ncoord_cur, ncoord_tot, fac);
  }
}

//
// Build
//
template <int tilesize>
void CudaNeighborList<tilesize>::build(const int indList, const int *zone_patom,
                                       const float boxx, const float boxy,
                                       const float boxz, const float rcut,
                                       const float4 *xyzq, const int *loc2glo,
                                       cudaStream_t stream) {
  assert(indList >= 0 && indList < numList);

  int cellStart = 0;
  for (int i = 0; i < indList; i++) {
    cellStart += sorter.at(i)->get_ncell();
  }

  builder.at(indList)->build(
      sorter.at(indList)->get_ncell(), cellStart, topExcl.getMaxNumExcl(),
      h_ZoneParam, d_ZoneParam, boxx, boxy, boxz, rcut, xyzq, loc2glo,
      topExcl.get_glo2loc(), topExcl.getAtomExclPos(), topExcl.getAtomExcl(),
      cell_xyz_zone, col_ncellz, col_cell, cell_bz, cell_patom, bb,
      h_NlistParam.at(indList), d_NlistParam.at(indList), stream);
  cudaCheck(cudaEventRecord(build_event.at(indList), stream));

  /*
  // Check for blown arrays
  copy_DtoH<NlistParam_t>(d_NlistParam.at(indList), h_NlistParam.at(indList),
  1, stream); cudaCheck(cudaStreamSynchronize(stream));

  int n_ientry = h_NlistParam.at(indList)->n_ientry;
  int n_tile = h_NlistParam.at(indList)->n_tile;

  if (n_tile > builder.at(indList)->get_n_tile_est()) {
    std::cout << "CudaNeighborListBuild::build, Limit blown: n_tile >
  n_tile_est"<< std::endl; exit(1);
  }

  if (n_ientry > builder.at(indList)->get_n_ientry_est()) {
    std::cout << "CudaNeighborListBuild::build, Limit blown: n_ientry >
  n_ientry_est"<< std::endl; exit(1);
  }
  */

  if (test) {
    int ncol_tot = 0;
    int ncell_tot = 0;
    for (int i = 0; i <= indList; i++) {
      ncol_tot += sorter.at(i)->get_ncol_tot();
      ncell_tot += sorter.at(i)->get_ncell();
    }
    builder.at(indList)->test_build(
        zone_patom, ncol_tot, ncell_tot, h_ZoneParam,
        topExcl.getAtomExclPosLen(), topExcl.getAtomExclPos(),
        topExcl.getAtomExclLen(), topExcl.getAtomExcl(), boxx, boxy, boxz, rcut,
        xyzq, loc2glo, topExcl.get_glo2loc(), topExcl.get_ncoord(), cell_patom,
        col_cell, cell_bz, bb);
    builder.at(indList)->analyze();
  }
}

/*
//
// Copies h_NlistParam (CPU) -> d_NlistParam (GPU)
//
template <int tilesize>
void CudaNeighborList<tilesize>::set_NlistParam(cudaStream_t stream) {
  cudaCheck(cudaMemcpyToSymbolAsync(d_NlistParam, h_NlistParam,
sizeof(NeighborListParam_t), 0, cudaMemcpyHostToDevice, stream));
}

//
// Copies d_NlistParam (GPU) -> h_NlistParam (CPU)
//
template <int tilesize>
void CudaNeighborList<tilesize>::NlistParam(cudaStream_t stream) {
  //cudaCheck(cudaMemcpyFromSymbol(h_NlistParam, d_NlistParam,
sizeof(NeighborListParam_t),
  //				 0, cudaMemcpyDeviceToHost));
}
*/

/*
//
// Copies h_ZoneParam (CPU) -> d_ZoneParam (GPU)
//
template <int tilesize>
void CudaNeighborList<tilesize>::setZoneParam(cudaStream_t stream) {
  cudaCheck(cudaMemcpyToSymbolAsync(&d_ZoneParam[izoneStart], h_ZoneParam,
sizeof(ZoneParam_t), 0, cudaMemcpyHostToDevice, stream));
}

//
// Copies d_ZoneParam (GPU) -> h_ZoneParam (CPU)
//
template <int tilesize>
void CudaNeighborList<tilesize>::getZoneParam() {
  cudaCheck(cudaMemcpyFromSymbol(h_ZoneParam, d_ZoneParam, sizeof(ZoneParam_t),
                                 0, cudaMemcpyDeviceToHost));
}

//
// Resets n_tile and n_ientry variables for build() -call
//
template <int tilesize>
void CudaNeighborList<tilesize>::reset() {
  get_NlistParam();
  cudaCheck(cudaDeviceSynchronize());
  h_NlistParam->n_tile = 0;
  h_NlistParam->n_ientry = 0;
  set_NlistParam(0);
  cudaCheck(cudaDeviceSynchronize());
}
*/

//
// Explicit instances of CudaNeighborList
//
template class CudaNeighborList<32>;
#endif // NOCUDAC
