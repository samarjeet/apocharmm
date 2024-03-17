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

#include "gpu_utils.h"
#include <cuda.h>

//----------------------------------------------------------------------------------------

// Generic version
template <typename AT, typename CT>
__global__ void reduce_force(const int n, const int stride_in,
                             const AT *__restrict__ data_in,
                             const int stride_out, CT *__restrict__ data_out) {}

// Convert "long long int" -> "float"
template <>
__global__ void reduce_force<long long int, float>(
    const int n, const int stride_in, const long long int *__restrict__ data_in,
    const int stride_out, float *__restrict__ data_out) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < n) {
    long long int val1 = data_in[pos];
    long long int val2 = data_in[pos + stride_in];
    long long int val3 = data_in[pos + stride_in * 2];
    data_out[pos] = ((float)val1) * INV_FORCE_SCALE;
    data_out[pos + stride_out] = ((float)val2) * INV_FORCE_SCALE;
    data_out[pos + stride_out * 2] = ((float)val3) * INV_FORCE_SCALE;
    pos += blockDim.x * gridDim.x;
  }
}

// Convert "long long int" -> "double"
template <>
__global__ void reduce_force<long long int, double>(
    const int n, const int stride_in, const long long int *__restrict__ data_in,
    const int stride_out, double *__restrict__ data_out) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < n) {
    long long int val1 = data_in[pos];
    long long int val2 = data_in[pos + stride_in];
    long long int val3 = data_in[pos + stride_in * 2];
    data_out[pos] = ((double)val1) * INV_FORCE_SCALE;
    data_out[pos + stride_out] = ((double)val2) * INV_FORCE_SCALE;
    data_out[pos + stride_out * 2] = ((double)val3) * INV_FORCE_SCALE;
    pos += blockDim.x * gridDim.x;
  }
}

//----------------------------------------------------------------------------------------

// Generic version
template <typename AT, typename CT>
__global__ void reduce_force(const int nfft_tot, const AT *__restrict__ data_in,
                             CT *__restrict__ data_out) {}

// Convert "long long int" -> "float"
template <>
__global__ void
reduce_force<long long int, float>(const int nfft_tot,
                                   const long long int *__restrict__ data_in,
                                   float *__restrict__ data_out) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < nfft_tot) {
    long long int val = data_in[pos];
    data_out[pos] = ((float)val) * INV_FORCE_SCALE;
    pos += blockDim.x * gridDim.x;
  }
}

// Convert "int" -> "float"
template <>
__global__ void reduce_force<int, float>(const int nfft_tot,
                                         const int *__restrict__ data_in,
                                         float *__restrict__ data_out) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < nfft_tot) {
    int val = data_in[pos];
    data_out[pos] = ((float)val) * INV_FORCE_SCALE_I;
    pos += blockDim.x * gridDim.x;
  }
}

// Convert "long long int" -> "double"
template <>
__global__ void
reduce_force<long long int, double>(const int nfft_tot,
                                    const long long int *__restrict__ data_in,
                                    double *__restrict__ data_out) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < nfft_tot) {
    long long int val = data_in[pos];
    data_out[pos] = ((double)val) * INV_FORCE_SCALE;
    pos += blockDim.x * gridDim.x;
  }
}

// Convert "float" -> "double"
template <>
__global__ void reduce_force<float, double>(const int nfft_tot,
                                            const float *__restrict__ data_in,
                                            double *__restrict__ data_out) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < nfft_tot) {
    float val = data_in[pos];
    data_out[pos] = ((double)val);
    pos += blockDim.x * gridDim.x;
  }
}

//----------------------------------------------------------------------------------------

// Generic version
template <typename AT, typename CT>
__global__ void reduce_force(const int nfft_tot, AT *data_in) {}

// Convert "long long int" -> "double"
template <>
__global__ void reduce_force<long long int, double>(const int nfft_tot,
                                                    long long int *data_in) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double *data_out = (double *)data_in;

  while (pos < nfft_tot) {
    long long int val = data_in[pos];
    data_out[pos] = ((double)val) * INV_FORCE_SCALE;
    pos += blockDim.x * gridDim.x;
  }
}

//----------------------------------------------------------------------------------------

// Generic version
template <typename AT, typename CT1, typename CT2>
__global__ void reduce_add_force(const int nfft_tot,
                                 const CT2 *__restrict__ data_add,
                                 AT *__restrict__ data_inout) {}

// Convert "long long int" -> "double" and adds "float"
template <>
__global__ void reduce_add_force<long long int, double, float>(
    const int nfft_tot, const float *__restrict__ data_add,
    long long int *__restrict__ data_inout) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double *data_out = (double *)data_inout;

  while (pos < nfft_tot) {
    long long int val = data_inout[pos];
    double val_add = (double)data_add[pos];
    data_out[pos] = ((double)val) * INV_FORCE_SCALE + val_add;
    pos += blockDim.x * gridDim.x;
  }
}

//----------------------------------------------------------------------------------------

// Generic version
template <typename T1, typename T2>
__global__ void add_force(const int nfft_tot, const T2 *__restrict__ data_add,
                          T1 *__restrict__ data_inout);

// Adds: "double" -> "double" + "long long int"
template <>
__global__ void
add_force<double, long long int>(const int nfft_tot,
                                 const long long int *__restrict__ data_add,
                                 double *__restrict__ data_inout) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < nfft_tot) {
    long long int val_add = data_add[pos];
    data_inout[pos] += ((double)val_add) * INV_FORCE_SCALE;
    pos += blockDim.x * gridDim.x;
  }
}

//----------------------------------------------------------------------------------------

// Adds: "double" -> "double" + "float3"
__global__ void add_nonstrided_force(const int n,
                                     const float3 *__restrict__ data_add,
                                     const int stride,
                                     double *__restrict__ data_inout) {
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < n) {
    data_inout[pos] += (double)data_add[pos].x;
    data_inout[pos + stride] += (double)data_add[pos].y;
    data_inout[pos + stride * 2] += (double)data_add[pos].z;
    pos += blockDim.x * gridDim.x;
  }
}
#endif // NOCUDAC
