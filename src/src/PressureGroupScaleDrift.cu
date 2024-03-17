// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include <PressureGroupScaleDrift.h>

/** calculate new positions and momentums with scaling.
 *      new_momentum= old_momentum*momentum_scale
 *      new_xyz=
 * old_xyz*position_prescale+timestep*momentum*invmass*momentum_prescale*/
__global__ void scaleDriftSimpleKernel(
    // inputs
    const double3 *__restrict__ momentum_scale,    /**< Device Pointer to total
                                                      scaling of momentum.*/
    const double3 *__restrict__ momentum_prescale, /**< Device pointer to
                                                      momentum scaling for
                                                      position update.*/
    const double3 *__restrict__ position_prescale, /**< Device pointer to
                                                      position scaling before
                                                      moving.*/
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
) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid == 0) {
    for (int i = 0; i < num; i++) {
      double4 r = old_xyzq[i];
      double4 p = old_momentum_invmass[i];
      double inv_mass = p.w;
      r.x *= position_prescale->x;
      r.y *= position_prescale->y;
      r.z *= position_prescale->z;
      r.x += timestep * inv_mass * (momentum_prescale->x) * p.x;
      r.y += timestep * inv_mass * (momentum_prescale->y) * p.y;
      r.z += timestep * inv_mass * (momentum_prescale->z) * p.z;
      p.x *= momentum_scale->x;
      p.y *= momentum_scale->y;
      p.z *= momentum_scale->z;
      new_momentum_invmass[i].x = p.x;
      new_momentum_invmass[i].y = p.y;
      new_momentum_invmass[i].z = p.z;
      new_xyzq[i].x = r.x;
      new_xyzq[i].y = r.y;
      new_xyzq[i].z = r.z;
    }
  }
}

PressureGroupScaleDrift::PressureGroupScaleDrift(
    // inputs
    const double3 *__restrict__ momentum_scale,    /**< Device Pointer to total
                                                      scaling of momentum.*/
    const double3 *__restrict__ momentum_prescale, /**< Device pointer to
                                                      momentum scaling for
                                                      position update.*/
    const double3 *__restrict__ position_prescale, /**< Device pointer to
                                                      position scaling before
                                                      moving.*/
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
    )
    : momentum_scale(momentum_scale), momentum_prescale(momentum_prescale),
      position_prescale(position_prescale),
      old_momentum_invmass(old_momentum_invmass), old_xyzq(old_xyzq),
      new_momentum_invmass(new_momentum_invmass), new_xyzq(new_xyzq),
      timestep(timestep), num(num) {
  initializeGraph();
}

void PressureGroupScaleDrift::initializeGraph(void) {
  // create parameters
  myparams = {0};
  kernelArgs = {(void *)&momentum_scale,
                (void *)&momentum_prescale,
                (void *)&position_prescale,
                (void *)&old_momentum_invmass,
                (void *)&old_xyzq,
                (void *)&new_momentum_invmass,
                (void *)&new_xyzq,
                (void *)&timestep,
                (void *)&num};
  myparams.func = (void *)scaleDriftSimpleKernel;
  myparams.gridDim = dim3(1, 1, 1);
  myparams.blockDim = dim3(1, 1, 1);
  myparams.sharedMemBytes = 0;
  myparams.kernelParams = (void **)(kernelArgs.data());
  myparams.extra = NULL;
  // Add nodes
  cudaCheck(cudaGraphAddKernelNode(&mynode, graph, NULL, 0, &myparams));
}
