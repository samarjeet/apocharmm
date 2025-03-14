// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE


#include "CudaEMap.h"
#include <iostream>
#include <chrono>

/*
__global__ void cuda_mass_spread(const double *__restrict__ masses,
                                 int numAtoms) {
  __shared__ int sh_ix[32];
  __shared__ int sh_iy[32];
  __shared__ int sh_iz[32];
  __shared__ float sh_m[32];

  // Process atoms pos to pos_end-1
  const unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x,
                     pos_end = min((blockIdx.x + 1) * blockDim.x, numAtoms);
}
*/

void CudaEMap::generate() {
  std::cout << "Generating EMap" << std::endl;

  /*
  int numAtoms = context->getNumAtoms();
  // Spread the mass of the particles to the grid
  dim3 nthread, nblock;

  // Only doing the case of 4 right now
  nthread.x = 32;  
  nthread.y = 4;   
  nthread.z = 1; 

  nblock.x = (numAtoms + nthread.x - 1) / nthread.x;
  nblock.y = 1; 
  nblock.z = 1;
  

  cuda_mass_spread<<<nblock, nthread, 0, stream>>>(
      atomicMasses.getDeviceArray().data(), numAtoms);cudaCheck(cudaGetLastError());
  // reduce the mass of the particles to the grid
  // this is similar to the spread of pme
  // how is it now
}
