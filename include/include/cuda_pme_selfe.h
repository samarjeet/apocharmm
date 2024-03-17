// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

// calc_self_energy_kernel body
// Calculates self energy
// kappa_ccelec_sqrtpi = kappa*ccelec/sqrt(pi)
(const int ncoord, const float4 *xyzq, const double kappa_ccelec_sqrtpi,
#ifdef DOMDEC_MSLDPME
 const float *bixlam, double *biflam, const int *blockIndexes,
#endif
 double *__restrict__ energy_self) {

  // Shared memory
  // Required space: blockDim.x*sizeof(double)
  extern __shared__ double sh_q2[];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  float q = 0.0f;
  if (i < ncoord) {
    // printf("%d\t%f\t%f\t%f\t%f\n", i, xyzq[i].x, xyzq[i].y, xyzq[i].z,
    // xyzq[i].w);
    q = xyzq[i].w;
#ifdef DOMDEC_MSLDPME
    int bii = blockIndexes[i];
    bii &= 0xffff; // First 16 bits are site index
    if (bii > 0) {
      atomicAdd(&biflam[bii], -2.0 * bixlam[bii] * q * q * kappa_ccelec_sqrtpi);
      q *= bixlam[bii];
    }
#endif
  }
  sh_q2[threadIdx.x] = q * q;
  __syncthreads();
  for (int d = 1; d < blockDim.x; d *= 2) {
    int t = threadIdx.x + d;
    double q2_val = (t < blockDim.x) ? sh_q2[t] : 0.0;
    __syncthreads();
    sh_q2[threadIdx.x] += q2_val;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    // printf("Thread : %d >> Updating energy_self by %f\n",i, q);
    atomicAdd(energy_self, -sh_q2[0] * kappa_ccelec_sqrtpi);
    // printf("Thread : %d >> to  %.6f\n", i, *energy_self);
  }
}
