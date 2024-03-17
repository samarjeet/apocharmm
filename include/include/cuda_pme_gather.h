// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

// gather_force_ORDER_ortho_kernel body
// Gathers forces from the grid
// blockDim.x            = Number of atoms each block loads
// blockDim.x*blockDim.y = Total number of threads per block
#define HALF_ORDER (ORDER / 2)
#define SQUARE_HALF (HALF_ORDER * HALF_ORDER)
#define CUBE_HALF (HALF_ORDER * SQUARE_HALF)
(const float4 *xyzq, const int ncoord, const int nfftx, const int nffty,
 const int nfftz, const int xsize, const int ysize, const int zsize,
 const float recip1, const float recip2, const float recip3, const float ccelec,
#ifdef USE_TEXTURE_OBJECTS
 const cudaTextureObject_t gridTexObj,
#endif
 const int stride, FT *force
#ifdef DOMDEC_MSLDPME
 ,
 const float *bixlam, double *biflam, const int *blockIndexes
#endif
) {

  const int tid = threadIdx.x + threadIdx.y * blockDim.x; // 0...63

  // Shared memory
  __shared__ gather_t<CT, ORDER> shmem[32];
#if __CUDA_ARCH__ < 300
#ifdef DOMDEC_MSLDPME
  __shared__ float4 shred_buf[32 * 2];
  volatile float4 *shred = &shred_buf[(tid / 8) * 8];
#else
  __shared__ float3 shred_buf[32 * 2];
  volatile float3 *shred = &shred_buf[(tid / 8) * 8];
#endif
#endif

#ifdef DOMDEC_MSLDPME
  __shared__ double sh_flambda[32];
#endif

  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int pos_end = min((blockIdx.x + 1) * blockDim.x, ncoord);

  // Load atom data into shared memory
  if (pos < pos_end && threadIdx.y == 0) {
    float4 xyzqi = xyzq[pos];
    float x = xyzqi.x;
    float y = xyzqi.y;
    float z = xyzqi.z;
    float q = xyzqi.w;

    float w;

    w = x * recip1 + 2.0f;
    float frx = (float)(nfftx * (w - (floorf(w + 0.5f) - 0.5f)));

    w = y * recip2 + 2.0f;
    float fry = (float)(nffty * (w - (floorf(w + 0.5f) - 0.5f)));

    w = z * recip3 + 2.0f;
    float frz = (float)(nfftz * (w - (floorf(w + 0.5f) - 0.5f)));

    int frxi = (int)frx;
    int fryi = (int)fry;
    int frzi = (int)frz;

    shmem[threadIdx.x].ix = frxi;
    shmem[threadIdx.x].iy = fryi;
    shmem[threadIdx.x].iz = frzi;
    shmem[threadIdx.x].charge = q;

    float wx = frx - (float)frxi;
    float wy = fry - (float)fryi;
    float wz = frz - (float)frzi;

    float3 theta_tmp[ORDER];
    float3 dtheta_tmp[ORDER];
    calc_theta_dtheta<float, float3, ORDER>(wx, wy, wz, theta_tmp, dtheta_tmp);

#pragma unroll
    for (int i = 0; i < ORDER; i++)
      shmem[threadIdx.x].thetax[i] = theta_tmp[i].x;

#pragma unroll
    for (int i = 0; i < ORDER; i++)
      shmem[threadIdx.x].thetay[i] = theta_tmp[i].y;

#pragma unroll
    for (int i = 0; i < ORDER; i++)
      shmem[threadIdx.x].thetaz[i] = theta_tmp[i].z;

#pragma unroll
    for (int i = 0; i < ORDER; i++)
      shmem[threadIdx.x].dthetax[i] = dtheta_tmp[i].x;

#pragma unroll
    for (int i = 0; i < ORDER; i++)
      shmem[threadIdx.x].dthetay[i] = dtheta_tmp[i].y;

#pragma unroll
    for (int i = 0; i < ORDER; i++)
      shmem[threadIdx.x].dthetaz[i] = dtheta_tmp[i].z;
  }
  __syncthreads();

  // We divide the 6x6x6 cube into 8 3x3x3 sub-cubes.
  // These sub-cubes are taken care by a single thread
  //
  // Calculate the index this thread is calculating
  // tid = 0...63
  const int t = (tid % 8); // 0...7
  // t = (tx0 + ty0*2 + tz0*4)/3
  // (tx0, ty0, tz0) gives the starting index of the 3x3x3 sub-cube
  const int tz0 = (t / 4) * HALF_ORDER;
  const int ty0 = ((t / 2) % 2) * HALF_ORDER;
  const int tx0 = (t % 2) * HALF_ORDER;

  //
  // Calculate forces for 32 atoms. We have 32*2 = 64 threads
  // Loop is iterated 4 times:
  //                         (iterations)
  // Threads 0...7   = atoms 0, 8,  16, 24
  // Threads 8...15  = atoms 1, 9,  17, 25
  // Threads 16...31 = atoms 2, 10, 18, 26
  //                ...
  // Threads 56...63 = atoms 7, 15, 23, 31
  //

  int base = tid / 8;
  const unsigned int shfl_mask = 0xFF << ((base % 4) * 8);
  const int base_end = pos_end - blockIdx.x * blockDim.x;
  while (base < base_end) {
#ifdef DOMDEC_MSLDPME
    double flambda = 0.0;
#endif

    float f1 = 0.0f;
    float f2 = 0.0f;
    float f3 = 0.0f;
    int ix0 = shmem[base].ix;
    int iy0 = shmem[base].iy;
    int iz0 = shmem[base].iz;

// Each thread calculates a 3x3x3 sub-cube
#pragma unroll
    for (int i = 0; i < CUBE_HALF; i++) {
      int tz = tz0 + (i / SQUARE_HALF);
      int ty = ty0 + ((i / HALF_ORDER) % HALF_ORDER);
      int tx = tx0 + (i % HALF_ORDER);

      int ix = ix0 + tx;
      int iy = iy0 + ty;
      int iz = iz0 + tz;

      if (ix >= nfftx)
        ix -= nfftx;
      if (iy >= nffty)
        iy -= nffty;
      if (iz >= nfftz)
        iz -= nfftz;

#ifdef USE_TEXTURE_OBJECTS
      float q0 = tex1Dfetch<float>(gridTexObj, ix + (iy + iz * ysize) * xsize);
#else
      float q0 = tex1Dfetch(gridTexRef, ix + (iy + iz * ysize) * xsize);
#endif

      // float q0 = datap[];
      float thx0 = shmem[base].thetax[tx];
      float thy0 = shmem[base].thetay[ty];
      float thz0 = shmem[base].thetaz[tz];

      float dthx0 = shmem[base].dthetax[tx];
      float dthy0 = shmem[base].dthetay[ty];
      float dthz0 = shmem[base].dthetaz[tz];

      f1 += dthx0 * thy0 * thz0 * q0;
      f2 += thx0 * dthy0 * thz0 * q0;
      f3 += thx0 * thy0 * dthz0 * q0;

#ifdef DOMDEC_MSLDPME
      flambda += thx0 * thy0 * thz0 * q0;
#endif
    }

//-------------------------

// Reduce
#if __CUDA_ARCH__ >= 300
    const int i = threadIdx.x & 7;

    // f1 += SHFL(f1, i + 4, 8);
    // f2 += SHFL(f2, i + 4, 8);
    // f3 += SHFL(f3, i + 4, 8);
    f1 += __shfl_sync(shfl_mask, f1, i + 4, 8);
    f2 += __shfl_sync(shfl_mask, f2, i + 4, 8);
    f3 += __shfl_sync(shfl_mask, f3, i + 4, 8);
#ifdef DOMDEC_MSLDPME
    // flambda += SHFL(flambda, i + 4, 8);
    flambda += __shfl_sync(shfl_mask, flambda, i + 4, 8);
#endif

    // f1 += SHFL(f1, i + 2, 8);
    // f2 += SHFL(f2, i + 2, 8);
    // f3 += SHFL(f3, i + 2, 8);
    f1 += __shfl_sync(shfl_mask, f1, i + 2, 8);
    f2 += __shfl_sync(shfl_mask, f2, i + 2, 8);
    f3 += __shfl_sync(shfl_mask, f3, i + 2, 8);
#ifdef DOMDEC_MSLDPME
    // flambda += SHFL(flambda, i + 2, 8);
    flambda += __shfl_sync(shfl_mask, flambda, i + 2, 8);
#endif

    // f1 += SHFL(f1, i + 1, 8);
    // f2 += SHFL(f2, i + 1, 8);
    // f3 += SHFL(f3, i + 1, 8);
    f1 += __shfl_sync(shfl_mask, f1, i + 1, 8);
    f2 += __shfl_sync(shfl_mask, f2, i + 1, 8);
    f3 += __shfl_sync(shfl_mask, f3, i + 1, 8);
#ifdef DOMDEC_MSLDPME
    // flambda += SHFL(flambda, i + 1, 8);
    flambda += __shfl_sync(shfl_mask, flambda, i + 1, 8);
#endif

    if (i == 0) {
      shmem[base].f1 = f1;
      shmem[base].f2 = f2;
      shmem[base].f3 = f3;
#ifdef DOMDEC_MSLDPME
      sh_flambda[base] = flambda;
#endif
    }

#else
    const int i = threadIdx.x & 7;
    shred[i].x = f1;
    shred[i].y = f2;
    shred[i].z = f3;
#ifdef DOMDEC_MSLDPME
    shred[i].w = flambda;
#endif

    if (i < 4) {
      shred[i].x += shred[i + 4].x;
      shred[i].y += shred[i + 4].y;
      shred[i].z += shred[i + 4].z;
#ifdef DOMDEC_MSLDPME
      shred[i].w += shred[i + 4].w;
#endif
    }

    if (i < 2) {
      shred[i].x += shred[i + 2].x;
      shred[i].y += shred[i + 2].y;
      shred[i].z += shred[i + 2].z;
#ifdef DOMDEC_MSLDPME
      shred[i].w += shred[i + 2].w;
#endif
    }

    if (i == 0) {
      shmem[base].f1 = shred[0].x + shred[1].x;
      shmem[base].f2 = shred[0].y + shred[1].y;
      shmem[base].f3 = shred[0].z + shred[1].z;
#ifdef DOMDEC_MSLDPME
      sh_flambda[base] = shred[0].w + shred[1].w;
#endif
    }
#endif

    base += 8;
  }

  // Write forces
  __syncthreads();
  if (pos < pos_end && threadIdx.y == 0) {
    float f1 = shmem[threadIdx.x].f1;
    float f2 = shmem[threadIdx.x].f2;
    float f3 = shmem[threadIdx.x].f3;

    float q = shmem[threadIdx.x].charge * ccelec;

#ifdef DOMDEC_MSLDPME
    // Ryan rearranged and scaled fx fy fz by bixlam and biflam by q
    int iblock = blockIndexes[pos] & 0xffff;
    if (iblock > 0) {
      // Multiply by unscaled charge
      atomicAdd(&biflam[iblock], q * sh_flambda[threadIdx.x]);
      // Scale charge for spatial forces
      q *= bixlam[iblock];
    }
#endif

    float fx = q * recip1 * f1 * nfftx;
    float fy = q * recip2 * f2 * nffty;
    float fz = q * recip3 * f3 * nfftz;

    gather_force_store<FT>(fx, fy, fz, stride, pos, force);
  }
}
#undef HALF_ORDER
#undef SQUARE_HALF
#undef CUBE_HALF
