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
#include "CudaBlock.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cassert>
#include <iostream>
#ifndef USE_TEXTURE_OBJECTS
#include "CudaDirectForceKernels.h"
#endif

//#ifndef USE_TEXTURE_OBJECTS
// VdW parameter texture reference
// texture<float, 1, cudaReadModeElementType> blockParamTexRef;
//#endif

//
// Class creator
//
CudaBlock::CudaBlock(const int numBlock, const int use_softcore,
                     const int use_PMEL)
    : numBlock(numBlock) {
  assert(numBlock >= 1);
  blockTypeLen = 0;
  blockType = NULL;
  useSoftcore = use_softcore;
  usePMEL = use_PMEL;
  allocate<float>(&d_blockParam, numBlock * (numBlock + 1) / 2);
  allocate_host<float>(&h_blockParam, numBlock * (numBlock + 1) / 2);
  if (usePMEL >= 2) {
    allocate<float>(&d_blockParamEx, numBlock * (numBlock + 1) / 2);
    allocate_host<float>(&h_blockParamEx, numBlock * (numBlock + 1) / 2);
  }
  if (useSoftcore == 2) {
    allocate<double>(&d_DSoftDFscale, numBlock * (numBlock + 1) / 2);
  }
  allocate<float>(&bixlam, numBlock);
  allocate<double>(&biflam, numBlock);
  allocate<double>(&biflam2, numBlock);
  allocate<int>(&siteMLD, numBlock);
#ifdef USE_TEXTURE_OBJECTS
  // Use texture objects
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = d_blockParam;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = sizeof(float) * 8;
  resDesc.res.linear.sizeInBytes =
      numBlock * (numBlock + 1) / 2 * sizeof(float);
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  cudaCheck(
      cudaCreateTextureObject(&blockParamTexObj, &resDesc, &texDesc, NULL));
#else
  assert(!getBlockParamTexRefBound());
  // Bind blockparam texture
  memset(getBlockParamTexRef(), 0,
         sizeof(texture<float, 1, cudaReadModeElementType>));
  getBlockParamTexRef()->normalized = 0;
  getBlockParamTexRef()->filterMode = cudaFilterModePoint;
  getBlockParamTexRef()->addressMode[0] = cudaAddressModeClamp;
  getBlockParamTexRef()->channelDesc.x = 32;
  getBlockParamTexRef()->channelDesc.y = 0;
  getBlockParamTexRef()->channelDesc.z = 0;
  getBlockParamTexRef()->channelDesc.w = 0;
  getBlockParamTexRef()->channelDesc.f = cudaChannelFormatKindFloat;
  cudaCheck(cudaBindTexture(NULL, *getBlockParamTexRef(), d_blockParam,
                            numBlock * (numBlock + 1) / 2 * sizeof(float)));
  setBlockParamTexRefBound(true);
#endif
}

//
// Class destructor
//
CudaBlock::~CudaBlock() {
#ifdef USE_TEXTURE_OBJECTS
  cudaCheck(cudaDestroyTextureObject(blockParamTexObj));
#else
  cudaCheck(cudaUnbindTexture(*getBlockParamTexRef()));
  setBlockParamTexRefBound(false);
#endif
  if (blockType != NULL)
    deallocate<int>(&blockType);
  deallocate<float>(&d_blockParam);
  deallocate_host<float>(&h_blockParam);
  if (usePMEL >= 2) {
    deallocate<float>(&d_blockParamEx);
    deallocate_host<float>(&h_blockParamEx);
  }
  if (useSoftcore == 2) {
    deallocate<double>(&d_DSoftDFscale);
  }
  deallocate<float>(&bixlam);
  deallocate<double>(&biflam);
  deallocate<double>(&biflam2);
  deallocate<int>(&siteMLD);
}

//
// Sets blocktype array from host memory
//
void CudaBlock::setBlockType(const int ncoord, const int *h_blockType) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord - 1) / warpsize + 1) * warpsize;
  reallocate<int>(&blockType, &blockTypeLen, ncoord_aligned, 1.2f);
  copy_HtoD_sync<int>(h_blockType, blockType, ncoord);
}

//
// Sets block parameters by copying them from CPU
// NOTE: The CPU buffer is in full-matrix form
//
void CudaBlock::setBlockParam(const float *h_blockParamFull) {
  int k = 0;
  for (int i = 0; i < numBlock; i++) {
    for (int j = 0; j <= i; j++) {
      h_blockParam[k] = h_blockParamFull[j * numBlock + i];
      if (usePMEL == 2) {
        h_blockParamEx[k] = h_blockParamFull[j * numBlock + i];
        if (i == j) {
          h_blockParamEx[k] *= h_blockParamFull[j * numBlock + i];
        }
      } else if (usePMEL == 3) {
        h_blockParamEx[k] = h_blockParamFull[i] * h_blockParamFull[j];
      }
      k++;
    }
  }
  copy_HtoD_sync<float>(h_blockParam, d_blockParam,
                        numBlock * (numBlock + 1) / 2);
  if (usePMEL >= 2) {
    copy_HtoD_sync<float>(h_blockParamEx, d_blockParamEx,
                          numBlock * (numBlock + 1) / 2);
  }
}

//
// Sets bixlam by copying them from CPU
//
void CudaBlock::setBixlam(const float *h_bixlam) {
  copy_HtoD_sync<float>(h_bixlam, bixlam, numBlock);
}

//
// Set siteMLD
//
void CudaBlock::setSiteMLD(const int *h_siteMLD) {
  copy_HtoD_sync<int>(h_siteMLD, siteMLD, numBlock);
}

//
// Copies biflam and biflam2 to CPU arrays
//
void CudaBlock::getBiflam(double *h_biflam, double *h_biflam2) {
  copy_DtoH_sync<double>((double *)biflam, h_biflam, numBlock);
  copy_DtoH_sync<double>((double *)biflam2, h_biflam2, numBlock);
}

//
// Sets biflam and biflam2 GPU arrays
//
void CudaBlock::setBiflam(double *h_biflam, double *h_biflam2) {
  copy_HtoD_sync<double>(h_biflam, (double *)biflam, numBlock);
  copy_HtoD_sync<double>(h_biflam2, (double *)biflam2, numBlock);
}
#endif // NOCUDAC
