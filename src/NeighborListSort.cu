#include "NeighborListSort.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cassert>
#include <iostream>

NeighborListSort::NeighborListSort() {
  tileSize = 32;
  nColTotal = 0;
}

__global__ void calcMinMaxXYZKernel(int numAtoms, CellParam_t *cellParam,
                                    const float4 *xyzq) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Shared memory
  // Requires: 6*blockDim.x*sizeof(float)
  extern __shared__ float sh_minmax_xyz[];
  volatile float *sh_min_x = &sh_minmax_xyz[0];
  volatile float *sh_min_y = &sh_minmax_xyz[blockDim.x];
  volatile float *sh_min_z = &sh_minmax_xyz[blockDim.x * 2];
  volatile float *sh_max_x = &sh_minmax_xyz[blockDim.x * 3];
  volatile float *sh_max_y = &sh_minmax_xyz[blockDim.x * 4];
  volatile float *sh_max_z = &sh_minmax_xyz[blockDim.x * 5];

  if (i < numAtoms) {
    float4 xyzq_i = xyzq[min(i, numAtoms - 1)];
    float x = xyzq_i.x;
    float y = xyzq_i.y;
    float z = xyzq_i.z;

    sh_min_x[threadIdx.x] = x;
    sh_min_y[threadIdx.x] = y;
    sh_min_z[threadIdx.x] = z;
    sh_max_x[threadIdx.x] = x;
    sh_max_y[threadIdx.x] = y;
    sh_max_z[threadIdx.x] = z;
    __syncthreads();

    // Reduce
    for (int d = 1; d < blockDim.x; d *= 2) {
      int t = threadIdx.x + d;
      float min_x = (t < blockDim.x) ? sh_min_x[t] : (float)(1.0e20);
      float min_y = (t < blockDim.x) ? sh_min_y[t] : (float)(1.0e20);
      float min_z = (t < blockDim.x) ? sh_min_z[t] : (float)(1.0e20);
      float max_x = (t < blockDim.x) ? sh_max_x[t] : (float)(-1.0e20);
      float max_y = (t < blockDim.x) ? sh_max_y[t] : (float)(-1.0e20);
      float max_z = (t < blockDim.x) ? sh_max_z[t] : (float)(-1.0e20);
      __syncthreads();
      sh_min_x[threadIdx.x] = min(sh_min_x[threadIdx.x], min_x);
      sh_min_y[threadIdx.x] = min(sh_min_y[threadIdx.x], min_y);
      sh_min_z[threadIdx.x] = min(sh_min_z[threadIdx.x], min_z);
      sh_max_x[threadIdx.x] = max(sh_max_x[threadIdx.x], max_x);
      sh_max_y[threadIdx.x] = max(sh_max_y[threadIdx.x], max_y);
      sh_max_z[threadIdx.x] = max(sh_max_z[threadIdx.x], max_z);
      __syncthreads();
    }

    // Store into global memory
    if (threadIdx.x == 0) {
      atomicMin(&cellParam->min_xyz.x, sh_min_x[0]);
      atomicMin(&cellParam->min_xyz.y, sh_min_y[0]);
      atomicMin(&cellParam->min_xyz.z, sh_min_z[0]);
      atomicMax(&cellParam->max_xyz.x, sh_max_x[0]);
      atomicMax(&cellParam->max_xyz.y, sh_max_y[0]);
      atomicMax(&cellParam->max_xyz.z, sh_max_z[0]);
    }
  }
}

void NeighborListSort::calcMinMaxXYZ(int numAtoms, const float4 *xyzq,
                                     std::shared_ptr<CellParam_t> h_cellParam,
                                     CellParam_t *d_cellParam,
                                     cudaStream_t stream) {
  h_cellParam->numAtoms = numAtoms;
  h_cellParam->min_xyz.x = (float)1.0e20;
  h_cellParam->min_xyz.y = (float)1.0e20;
  h_cellParam->min_xyz.z = (float)1.0e20;
  h_cellParam->max_xyz.x = (float)(-1.0e20);
  h_cellParam->max_xyz.y = (float)(-1.0e20);
  h_cellParam->max_xyz.z = (float)(-1.0e20);

  copy_HtoD_T(h_cellParam.get(), d_cellParam, 1, sizeof(CellParam_t));

  int numThreads = 512;
  int numBlocks = (numAtoms - 1) / numThreads + 1;
  int sharedMemSize = 6 * numThreads * sizeof(float);

  calcMinMaxXYZKernel<<<numBlocks, numThreads, sharedMemSize, stream>>>(
      numAtoms, d_cellParam, xyzq);
  cudaDeviceSynchronize();
  copy_DtoH_T(d_cellParam, h_cellParam.get(), 1, sizeof(CellParam_t));
  cudaCheck(cudaDeviceSynchronize());
}

void NeighborListSort::sortSetup(std::shared_ptr<CellParam_t> h_cellParam,
                                 CellParam_t *d_cellParam) {
  nColTotal = 0;

  float xSize = h_cellParam->max_xyz.x - h_cellParam->min_xyz.x + 0.001f;
  float ySize = h_cellParam->max_xyz.y - h_cellParam->min_xyz.y + 0.001f;
  float zSize = h_cellParam->max_xyz.z - h_cellParam->min_xyz.z + 0.001f;

  float delta =
      powf(xSize * ySize * zSize * tileSize / (float)h_cellParam->numAtoms,
           1.0f / 3.0f);

  h_cellParam->nCellx = max(1, (int)(xSize / delta));
  h_cellParam->nCelly = max(1, (int)(ySize / delta));

  h_cellParam->nCellzMax =
      max(1, 2 * h_cellParam->numAtoms /
                 (h_cellParam->nCellx * h_cellParam->nCelly * tileSize));

  h_cellParam->cellDx = xSize / (float)(h_cellParam->nCellx);
  h_cellParam->cellDy = ySize / (float)(h_cellParam->nCelly);
  h_cellParam->cellDzMin = zSize / (float)(h_cellParam->nCellzMax);

  h_cellParam->invCellDx = 1.0f / h_cellParam->cellDx;
  h_cellParam->invCellDy = 1.0f / h_cellParam->cellDy;

  int nCellxy = h_cellParam->nCellx * h_cellParam->nCelly;
  nColTotal = nCellxy;
  nCellMax = nCellxy * h_cellParam->nCellzMax;
  copy_HtoD_T(h_cellParam.get(), d_cellParam, 1, sizeof(CellParam_t));
}

void NeighborListSort::sortCore() {}
