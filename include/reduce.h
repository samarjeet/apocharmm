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

#ifndef REDUCE_H
#define REDUCE_H

//----------------------------------------------------------------------------------------

template <typename AT, typename CT>
__global__ void reduce_force(const int n, const int stride_in,
                             const AT *__restrict__ data_in,
                             const int stride_out, CT *__restrict__ data_out);

//----------------------------------------------------------------------------------------

template <typename AT, typename CT>
__global__ void reduce_force(const int nfft_tot, const AT *__restrict__ data_in,
                             CT *__restrict__ data_out);

//----------------------------------------------------------------------------------------

template <typename AT, typename CT>
__global__ void reduce_force(const int nfft_tot, AT *data_in);

//----------------------------------------------------------------------------------------

template <typename AT, typename CT1, typename CT2>
__global__ void reduce_add_force(const int nfft_tot,
                                 const CT2 *__restrict__ data_add,
                                 AT *__restrict__ data_inout);

//----------------------------------------------------------------------------------------

template <typename T1, typename T2>
__global__ void add_force(const int nfft_tot, const T2 *__restrict__ data_add,
                          T1 *__restrict__ data_inout);

//----------------------------------------------------------------------------------------

__global__ void add_nonstrided_force(const int n,
                                     const float3 *__restrict__ data_add,
                                     const int stride,
                                     double *__restrict__ data_inout);
#endif // REDUCE_H
#endif // NOCUDAC
