// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

// body of spread_charge_ortho_ORDER(_block)
//
// Spreads the charge on the grid. Calculates theta and dtheta on the fly
// blockDim.x               = Number of atoms each block loads
// blockDim.y*blockDim.x/64 = Number of atoms we spread at once
#define SQUARE_ORDER (ORDER * ORDER)

#if ORDER == 6
#define CUBE_ORDER 224
#else
#define CUBE_ORDER (ORDER * ORDER * ORDER)
#endif

(const float4 *xyzq, const int ncoord, const float recip11, const float recip22,
 const float recip33, const int nfftx, const int nffty, const int nfftz,
#ifdef DOMDEC_MSLDPME
 const float *bixlam, const int *blockIndexes,
#endif
 AT *data) {

  __shared__ int sh_ix[32];
  __shared__ int sh_iy[32];
  __shared__ int sh_iz[32];
  __shared__ float sh_q[32];

  __shared__ float sh_thetax[ORDER * 32];
  __shared__ float sh_thetay[ORDER * 32];
  __shared__ float sh_thetaz[ORDER * 32];

  // Process atoms pos to pos_end-1
  const unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x,
                     pos_end = min((blockIdx.x + 1) * blockDim.x, ncoord);

  if (pos < pos_end && threadIdx.y == 0) {
    float4 xyzqi = xyzq[pos];

    float x = xyzqi.x;
    float y = xyzqi.y;
    float z = xyzqi.z;
    float q = xyzqi.w;

#ifdef DOMDEC_MSLDPME
    int bii = blockIndexes[pos];
    bii &= 0xffff; // First 16 bits are site index
    if (bii > 0) {
      q *= bixlam[bii];
    }
#endif

    sh_q[threadIdx.x] = q;

    float w;

    w = x * recip11 + 2.0f;
    float frx = (float)(nfftx * (w - (floorf(w + 0.5f) - 0.5f)));
    w = y * recip22 + 2.0f;
    float fry = (float)(nffty * (w - (floorf(w + 0.5f) - 0.5f)));
    w = z * recip33 + 2.0f;
    float frz = (float)(nfftz * (w - (floorf(w + 0.5f) - 0.5f)));

    int frxi = (int)frx;
    int fryi = (int)fry;
    int frzi = (int)frz;

    sh_ix[threadIdx.x] = frxi;
    sh_iy[threadIdx.x] = fryi;
    sh_iz[threadIdx.x] = frzi;

    float wx = frx - (float)frxi;
    float wy = fry - (float)fryi;
    float wz = frz - (float)frzi;

    float theta[ORDER];

    calc_one_theta<float, ORDER>(wx, theta);
#pragma unroll
    for (int i = 0; i < ORDER; ++i)
      sh_thetax[threadIdx.x * ORDER + i] = theta[i];

    calc_one_theta<float, ORDER>(wy, theta);
#pragma unroll
    for (int i = 0; i < ORDER; ++i)
      sh_thetay[threadIdx.x * ORDER + i] = theta[i];

    calc_one_theta<float, ORDER>(wz, theta);
#pragma unroll
    for (int i = 0; i < ORDER; ++i)
      sh_thetaz[threadIdx.x * ORDER + i] = theta[i];
  }

  __syncthreads();

  // Grid point location, values of (ix0, iy0, iz0) are in range 0..5
  // NOTE: Only tid=0...215 do any computation for order 6
  const int tid = (threadIdx.x + threadIdx.y * blockDim.x) % CUBE_ORDER;
#if ORDER == 6
  if (tid >= 216)
    return;
#endif

  const int x0 = tid % ORDER;
  const int y0 = (tid / ORDER) % ORDER;
  const int z0 = tid / SQUARE_ORDER;

  // Loop over atoms pos..pos_end-1
  int iadd = blockDim.x * blockDim.y / CUBE_ORDER;
  int i = (threadIdx.x + threadIdx.y * blockDim.x) / CUBE_ORDER;
  int iend = pos_end - blockIdx.x * blockDim.x;
  for (; i < iend; i += iadd) {
    int x = sh_ix[i] + x0;
    int y = sh_iy[i] + y0;
    int z = sh_iz[i] + z0;

    float q = sh_q[i];

    if (x >= nfftx)
      x -= nfftx;
    if (y >= nffty)
      y -= nffty;
    if (z >= nfftz)
      z -= nfftz;

    // Get position on the grid
    int ind = x + nfftx * (y + nffty * z);

    // Here we unroll the CUBE_ORDER loop with CUBE_ORDER threads.
    // Calculate interpolated charge value and store it to global memory
    write_grid<AT>(q * sh_thetax[i * ORDER + x0] * sh_thetay[i * ORDER + y0] *
                       sh_thetaz[i * ORDER + z0],
                   ind, data);
  }
}
#undef SQUARE_ORDER
#undef CUBE_ORDER
