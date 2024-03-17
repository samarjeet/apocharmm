#pragma once

#include "CudaNeighborListStruct.h"
#include <cuda.h>
#include <memory>
#include <vector>

class NeighborListSort {
public:
  NeighborListSort();
  void calcMinMaxXYZ(int numAtoms, const float4 *xyzq,
                     std::shared_ptr<CellParam_t> h_cellParam,
                     CellParam_t *d_cellParam, cudaStream_t stream);
  void sortSetup(std::shared_ptr<CellParam_t> h_cellParam,
                 CellParam_t *d_cellParam);
  void sortCore();

private:
  // Total number of cols
  int nColTotal;

  // Max estimate for the total # of cells
  int nCellMax;

  // Toal number of cells
  int nCell;

  // Number of atoms in each colum
  // Should this be CudaContainer<int> ?
  std::vector<int> numAtomsInCol;

  // Cumulative number of atoms in column
  std::vector<int> pAtomsInCol;

  // (x,y) coord of a column
  std::vector<std::pair<int, int>> colXY;

  // Column index of each atom
  std::vector<int> colIndexOfAtom;

  int tileSize;
};
