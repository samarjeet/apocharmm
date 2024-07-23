// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Andrew Simmonett, Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include "CharmmContext.h"
#include "CudaContainer.h"
#include <cuda_runtime.h>
#include <memory>

class CudaEMap {
public:
  CudaEMap(std::shared_ptr<CharmmContext> context) : context(context) {
    cudaCheck(cudaStreamCreate(&stream));

    // for now we are assuming that the grid is cubic
    // and set to 100x100x100.
    // this can be changed later
    nx = 100;
    ny = 100;
    nz = 100;

    auto masses = context->getForceManager()->getPSF()->getAtomMasses();
    // atomicMasses = CudaContainer<double>(masses.size());
    atomicMasses.allocate(masses.size());
    atomicMasses.set(masses);
  }

  ~CudaEMap() { cudaCheck(cudaStreamDestroy(stream)); }
  void generate();

private:
  std::shared_ptr<CharmmContext> context;

  CudaContainer<double> atomicMasses;

  // cudastream
  cudaStream_t stream;
  // fft

  int nx, ny, nz;
};