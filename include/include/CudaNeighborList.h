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
#ifndef CUDANEIGHBORLIST_H
#define CUDANEIGHBORLIST_H
//
// Neighbor list class
//
// (c) Antti-Pekka Hynninen 2014
// aphynninen@hotmail.com
//
#include "CudaNeighborListBuild.h"
#include "CudaNeighborListSort.h"
#include "CudaP21NeighborListBuild.h"
#include "CudaTopExcl.h"
#include "NeighborListSort.h"
#include "PBC.h"
#include <cassert>
#include <cuda.h>
#include <iostream>
#include <memory>
#include <vector>

template <int tilesize> class CudaNeighborList {
private:
  // Number of zones
  int numZone;

  // Number of lists
  int numList;

  // Zone parameters
  int h_ZoneParam_len;
  ZoneParam_t *h_ZoneParam;

  int d_ZoneParam_len;
  ZoneParam_t *d_ZoneParam;

  // Topological exclusions
  const CudaTopExcl &topExcl;

  // List sorting
  std::vector<CudaNeighborListSort *> sorter;

  // List building
  std::vector<CudaNeighborListBuild<tilesize> *> builder;

  std::shared_ptr<CudaP21NeighborListBuild> p21builder;

  // Total number of columns for each list
  // std::vector<int> ncol_tot;

  // NlistParam lists
  std::vector<NlistParam_t *> d_NlistParam;
  std::vector<NlistParam_t *> h_NlistParam;

  // Events
  std::vector<cudaEvent_t> build_event;
  cudaEvent_t glo2loc_reset_event;

  // Image boundaries: -1, 0, 1
  int imx_lo, imx_hi;
  int imy_lo, imy_hi;
  int imz_lo, imz_hi;

  // Number of z-cells in each column
  int col_ncellz_len;
  int *col_ncellz;

  // Starting cell index for each column
  int col_cell_len;
  int *col_cell;

  // Index sorting performed by this class
  // Usage: ind_sorted[i] = j : Atom j belongs to position i
  int ind_sorted_len;
  int *ind_sorted;

  // Atom indices where each cell start
  int cell_patom_len;
  int *cell_patom;

  // (icellx, icelly, icellz, izone) for each cell
  int cell_xyz_zone_len;
  int4 *cell_xyz_zone;

  // Cell z-boundaries
  int cell_bz_len;
  float *cell_bz;

  // Approximate upper bound for number of cells
  int ncell_max;

  // Bounding boxes
  int bb_len;
  bb_t *bb;

  // Flag for testing neighborlist build
  bool test;
  PBC pbc;

  void set_NlistParam(cudaStream_t stream);
  void get_NlistParam();

  void sort_realloc(const int indList);

  std::shared_ptr<CellParam_t> h_cellParam;
  CellParam_t *d_cellParam;
  std::unique_ptr<NeighborListSort> neighborListSorter;

public:
  CudaNeighborList();
  CudaNeighborList(const CudaTopExcl &topExcl, const int nx, const int ny,
                   const int nz);
  ~CudaNeighborList();
  void setPBC(PBC _pbc) { pbc = _pbc; }
  void registerList(std::vector<int> &numIntZone,
                    std::vector<std::vector<int>> &intZones,
                    const char *filename = NULL);

  void sort(const int indList, const int *zone_patom, const float4 *xyzq,
            float4 *xyzq_sorted, int *loc2glo, cudaStream_t stream = 0);

  void build(const int indList, const int *zone_patom, const float boxx,
             const float boxy, const float boxz, const float rcut,
             const float4 *xyzq, const int *loc2glo, cudaStream_t stream = 0);

  // void reset();

  int *get_glo2loc() { return topExcl.get_glo2loc(); }

  int *get_ind_sorted() { return ind_sorted; }

  int getNumList() { return numList; }

  void set_test(bool test_in) {
    this->test = test_in;
    for (int indList = 0; indList < numList; indList++) {
      sorter.at(indList)->set_test(test_in);
      builder.at(indList)->set_test(test_in);
    }
  }

  CudaNeighborListBuild<tilesize> &getBuilder(const int indList) {
    assert(indList >= 0 && indList < numList);
    //assert(pbc == PBC::P1);

    return *builder.at(indList);
  }

  /*CudaP21NeighborListBuild &getP21Builder() {
    assert(pbc == PBC::P21);
    return *p21builder;
  }
  */

  void analyze() {
    for (int indList = 0; indList < numList; indList++) {
      builder.at(indList)->analyze();
    }
  }
};

#endif // CUDANEIGHBORLIST_H
#endif // NOCUDAC
