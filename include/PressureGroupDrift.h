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
 * \date
 * \brief Classes and kernels to move and scale the centers of mass and net
 * momentums of each pressure group.
 */
#pragma once
#include <CudaIntegratorGraph.h>
#include <PressureGroupsUtil.h>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/** calculate new positions and momentums with scaling.
 *      new_momentum= old_momentum*momentum_scale
 *      new_xyz=
 * old_xyz*position_prescale+timestep*momentum*invmass*momentum_prescale*/
__global__ void scaleDriftSimpleKernel(
    // inputs
    const double3 *__restrict__ momentum_scale, /**< Device Pointer to total
                                                   scaling of momentum.*/
    const double3
        *__restrict__ momentum_prescale, /**< Device pointer to momentum scaling
                                            for position update.*/
    const double3
        *__restrict__ position_prescale, /**< Device pointer to position scaling
                                            before moving.*/
    const double4 *old_momentum_invmass, /**< Device pointer to array of
                                            momentums and inverse masses.*/
    const double4 *old_xyzq, /**< Device pointer to array of positions.*/
    // outputs
    double4 *new_momentum_invmass, /**< Device pointer to array of new
                                      momentums, and invers masses.*/
    double4 *new_xyzq, /**< Device pointer to array of new positions.*/
    // parameters
    double timestep, /**< time step.*/
    int num          /**< number of elements in the arrays.*/
);
