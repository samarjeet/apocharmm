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
#ifndef CUDABLOCK_H
#define CUDABLOCK_H
//
// BLOCK / lambda dynamics storage class
//
// (c) Antti-Pekka Hynninen 2015
//
#include <cuda.h>

class CudaBlock {
private:
  // Number of blocks
  const int numBlock;

  int useSoftcore;
  int usePMEL;

  // Block type for each atom (ncoord -size), values in range 0...numBlock-1
  // blockType = (sitemld[ibl] << 16) | (ibl-1), ibl=0...numBlock-1
  int blockTypeLen;
  int *blockType;

  // parameter (lambda) for each block pair, size numBlock*(numBlock+1)/2
  float *d_blockParam;
  float *d_blockParamEx;
  float *h_blockParam;
  float *h_blockParamEx;
#ifdef USE_TEXTURE_OBJECTS
  cudaTextureObject_t blockParamTexObj;
#endif

  // 1-4 soft-core interactions
  double *d_DSoftDFscale;

  // dimensions of each Site in Multi-site L-dynamics (size numBlock)
  int *siteMLD;

  // Coupling coefficients for sites (size numBlock)
  float *bixlam;

  // Results (size numBlock each)
  double *biflam;
  double *biflam2;

public:
  CudaBlock(const int numBlock, const int use_softcore, const int use_PMEL);
  ~CudaBlock();

  void setBlockType(const int ncoord, const int *h_blockType);
  void setBlockParam(const float *h_blockParamFull);
  void setBixlam(const float *h_bixlam);
  void setSiteMLD(const int *h_siteMLD);

  int getNumBlock() { return numBlock; }
  int *getBlockType() { return blockType; }
  float *getBixlam() { return bixlam; }
  double *getBiflam() { return biflam; }
  double *getBiflam2() { return biflam2; }
  void getBiflam(double *h_biflam, double *h_biflam2);
  void setBiflam(double *h_biflam, double *h_biflam2);
#ifdef USE_TEXTURE_OBJECTS
  cudaTextureObject_t *getBlockParamTexObj() { return &blockParamTexObj; }
#endif
  float *getBlockParam() { return d_blockParam; }
  float *getBlockParamEx() { return d_blockParamEx; }
  double *getDSoftDFscale() { return d_DSoftDFscale; }
  float getBlockParamValue(const int k) { return h_blockParam[k]; }
  float getBlockParamExValue(const int k) { return h_blockParamEx[k]; }
  int *getSiteMLD() { return siteMLD; }
  int getUseSoftcore() { return useSoftcore; }
  int getUsePMEL() { return usePMEL; }
};

#endif // CUDABLOCK_H
#endif // NOCUDAC
