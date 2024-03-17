// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Nathan Zimmerberg, Samarjeet Prasad
//
// ENDLICENSE

/** \file
 * \author Nathan Zimmerberg (nhz2@cornell.edu)
 * \date 05 Aug 2019
 * \brief Utilites to easily generate normal distrobutions using counter based
 * RNG on host or device.
 */
#pragma once
/*
#include <Random123/boxmuller.hpp>
#include <Random123/philox.h>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

// * Returns independent normalally distributed double2 based on the counters
and key.
// *  Uses the philox random CBRNG 4x32_10 and boxmuller.
// /
CUDA_CALLABLE_MEMBER inline double2 randnormal(uint64_t key, uint64_t
counter_msw, uint64_t counter_lsw) { uint64_t u0; uint64_t u1; philox4x32_key_t
k = {{uint32_t(key >> 32ull), uint32_t(key % (1ull << 32ull))}};
    philox4x32_ctr_t c = {{uint32_t(counter_msw >> 32ull), uint32_t(counter_msw
% (1ull << 32ull)), uint32_t(counter_lsw >> 32ull), uint32_t(counter_lsw % (1ull
<< 32ull))}}; philox4x32_ctr_t randval = philox4x32(c, k); u0 =
((uint64_t(randval.v[0])) << 32ull) | (uint64_t(randval.v[1])); u1 =
((uint64_t(randval.v[2])) << 32ull) | (uint64_t(randval.v[3]));
    // printf("key is, 0x%x%x\n",k.v[0],k.v[1]);
    // printf("counter is, 0x%x%x%x%x\n",c.v[0],c.v[1],c.v[2],c.v[3]);
    return r123::boxmuller(u0, u1);
}

*/
