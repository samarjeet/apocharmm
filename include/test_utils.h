// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

//#pragma once
#include <cuda_runtime.h>
#include <iostream>

__global__ void testAtomics(double *address) {
  double val = 0.15;
  *address = 0;
  double newVal = atomicAdd(address, val);
}
/*
void testCompiler(){
#ifdef __CUDACC__
  std::cout << "is compiler nvcc : " << __CUDACC__ << "\n";
#else
  std::cout << "not compiled via nvcc\n";
#endif
}
*/
