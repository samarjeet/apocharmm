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

#include "CudaTopExcl.h"
#include "cuda_utils.h"
#include <cuda.h>
#include <fstream>
#include <iostream>

//
// Dummy Constructor
//
CudaTopExcl::CudaTopExcl() {
  atomExclPosLen = 0;
  atomExclPos = NULL;
  atomExclLen = 0;
  atomExcl = NULL;
}

void CudaTopExcl::setFromFile(const int numAtoms, std::string iblo14File,
                              std::string inb14File) {
  ncoord = numAtoms;
  int *iblo14, *inb14;
  iblo14 = new int[ncoord];
  std::ifstream iblo14Stream(iblo14File.c_str());
  for (int i = 0; i < ncoord; ++i) {
    iblo14Stream >> iblo14[i];
  }
  // for (int i=0; i<ncoord; ++i) std::cout << iblo14[i] << "\n";

  inb14 = new int[iblo14[ncoord - 1]];
  std::ifstream inb14Stream(inb14File.c_str());
  for (int i = 0; i < iblo14[ncoord - 1]; ++i) {
    inb14Stream >> inb14[i];
  }
  // for (int i=0; i < iblo14[ncoord-1]; ++i) std::cout << inb14[i] << "\n";
  setup(iblo14, inb14);

  allocate<int>(&glo2loc, ncoord);
  set_gpu_array<int>(glo2loc, ncoord, -1);
}

void CudaTopExcl::setFromVector(int numAtoms, const std::vector<int> &iblo14,
                                const std::vector<int> &inb14) {
  ncoord = numAtoms;
  setup(iblo14.data(), inb14.data());
  allocate<int>(&glo2loc, ncoord);
  set_gpu_array<int>(glo2loc, ncoord, -1);
}
//
// Class creator
//
CudaTopExcl::CudaTopExcl(const int ncoord, const int *iblo14, const int *inb14)
    : ncoord(ncoord) {
  atomExclPosLen = 0;
  atomExclPos = NULL;
  atomExclLen = 0;
  atomExcl = NULL;
  setup(iblo14, inb14);

  allocate<int>(&glo2loc, ncoord);
  set_gpu_array<int>(glo2loc, ncoord, -1);
}

CudaTopExcl::CudaTopExcl(const int ncoord, std::string iblo14File,
                         std::string inb14File)
    : ncoord(ncoord) {
  atomExclPosLen = 0;
  atomExclPos = NULL;
  atomExclLen = 0;
  atomExcl = NULL;
  int *iblo14, *inb14;
  iblo14 = new int[ncoord];
  std::ifstream iblo14Stream(iblo14File.c_str());
  for (int i = 0; i < ncoord; ++i) {
    iblo14Stream >> iblo14[i];
  }
  // for (int i=0; i<ncoord; ++i) std::cout << iblo14[i] << "\n";

  inb14 = new int[iblo14[ncoord - 1]];
  std::ifstream inb14Stream(inb14File.c_str());
  for (int i = 0; i < iblo14[ncoord - 1]; ++i) {
    inb14Stream >> inb14[i];
  }
  // for (int i=0; i < iblo14[ncoord-1]; ++i) std::cout << inb14[i] << "\n";
  setup(iblo14, inb14);

  allocate<int>(&glo2loc, ncoord);
  set_gpu_array<int>(glo2loc, ncoord, -1);
}
//
// Class destructor
//
CudaTopExcl::~CudaTopExcl() {
  if (atomExclPos != NULL)
    deallocate<int>(&atomExclPos);
  if (atomExcl != NULL)
    deallocate<int>(&atomExcl);
  deallocate<int>(&glo2loc);
}

/**
 * @brief Setups topological exclusions from data structure used in CHARMM
 *
 * @param iblo14
 *
 * @param inb14
 */
void CudaTopExcl::setup(const int *iblo14, const int *inb14) {
  int *nexcl = new int[ncoord];

  for (int i = 0; i < ncoord; i++)
    nexcl[i] = 0;

  // Count the number of exclusions to nexcl[0 ... ncoord-1]
  for (int i = 0; i < ncoord; i++) {
    int excl_start;
    if (i > 0) {
      excl_start = iblo14[i - 1];
    } else {
      excl_start = 0;
    }
    int excl_end = iblo14[i] - 1;
    // add i-j exclusions to atom i
    nexcl[i] += excl_end - excl_start + 1;
    for (int excl_i = excl_start; excl_i <= excl_end; excl_i++) {
      int j = abs(inb14[excl_i]) - 1;
      // add i-j exclusion to atom j
      nexcl[j]++;
    }
  }

  // Find out maximum number of atom-atom exclusions per atom
  maxNumExcl = 0;
  for (int i = 0; i < ncoord; i++)
    maxNumExcl = max(maxNumExcl, nexcl[i]);

  int *h_atomExclPos = new int[ncoord + 1];

  // Use exclusive cumulative sum to calculate positions
  int nexcl_tot = 0;
  for (int i = 0; i < ncoord; i++) {
    h_atomExclPos[i] = nexcl_tot;
    nexcl_tot += nexcl[i];
  }
  h_atomExclPos[ncoord] = nexcl_tot;

  int *h_atomExcl = new int[nexcl_tot];

  for (int i = 0; i < ncoord; i++)
    nexcl[i] = 0;

  for (int i = 0; i < ncoord; i++) {
    int excl_start;
    if (i > 0) {
      excl_start = iblo14[i - 1];
    } else {
      excl_start = 0;
    }
    int excl_end = iblo14[i] - 1;

    int pos_starti = h_atomExclPos[i];
    int ni = nexcl[i];
    for (int excl_i = excl_start; excl_i <= excl_end; excl_i++) {
      int j = abs(inb14[excl_i]) - 1;
      // Add i-j exclusion to atom j
      int pos_startj = h_atomExclPos[j];
      int nj = nexcl[j];
      if (pos_startj + nj >= h_atomExclPos[j + 1]) {
        //std::cerr << "CudaTopExcl::setup, overflow in j" << std::endl;
        throw std::invalid_argument("CudaTopExcl::setup, overflow in j\n" );
        exit(1);
      }
      h_atomExcl[pos_startj + nj] = i;
      nj++;
      nexcl[j] = nj;
      // Add i-j exclusion to atom i
      if (pos_starti + ni >= h_atomExclPos[i + 1]) {
        //std::cerr << "CudaTopExcl::setup, overflow in i" << std::endl;
        throw std::invalid_argument("CudaTopExcl::setup, overflow in i\n");
        exit(1);
      }
      h_atomExcl[pos_starti + ni] = j;
      ni++;
    }

    nexcl[i] = ni;
  }

// Allocate GPU memory and copy results to GPU
#ifdef STRICT_MEMORY_REALLOC
  reallocate<int>(&atomExclPos, &atomExclPosLen, ncoord + 1, 1.0f);
  reallocate<int>(&atomExcl, &atomExclLen, nexcl_tot, 1.0f);
#else
  reallocate<int>(&atomExclPos, &atomExclPosLen, ncoord + 1, 1.1f);
  reallocate<int>(&atomExcl, &atomExclLen, nexcl_tot, 1.1f);
#endif
  copy_HtoD_sync<int>(h_atomExclPos, atomExclPos, ncoord + 1);
  copy_HtoD_sync<int>(h_atomExcl, atomExcl, nexcl_tot);

  delete[] h_atomExcl;
  delete[] h_atomExclPos;
  delete[] nexcl;
}
#endif // NOCUDAC
