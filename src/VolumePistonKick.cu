// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include <VolumePistonKick.h>

/** Update the box time dirivative based on the virial.*/
__global__ void volumePistonKickKernel(
    // inputs
    const double3 *__restrict__ box, /**< device pointer to box dimensions.*/
    const double3 *old_box_dot, /**< device pointer to box time dirivative.*/
    const double3 *__restrict__ virial, /**< device pointer to Corrected
                                           virial.*/
    // outputs
    double3 *new_box_dot, /**< device pointer to new box time dirivatives.*/
    // parameters
    double timestep,   /**< Time step. */
    double piston_mass /**< Piston mass, dimensions of mass/length^4. */
) {
  double3 b = *box;
  double3 b_dot = *old_box_dot;
  double volume = b.x * b.y * b.z;
  double volume_dot =
      b_dot.x * b.y * b.z + b.x * b_dot.y * b.z + b.x * b.y * b_dot.z;
  volume_dot += -(virial->x + virial->y + virial->z) / 3.0 / volume /
                piston_mass * timestep;
  new_box_dot->x = volume_dot / 3.0 / b.y / b.z;
  new_box_dot->y = volume_dot / 3.0 / b.x / b.z;
  new_box_dot->z = volume_dot / 3.0 / b.x / b.y;
}

VolumePistonKick::VolumePistonKick(
    // inputs
    const double3 *__restrict__ box, /**< device pointer to box dimensions.*/
    const double3 *old_box_dot, /**< device pointer to box time dirivative.*/
    const double3 *__restrict__ virial, /**< device pointer to Corrected
                                           virial.*/
    // outputs
    double3 *new_box_dot, /**< device pointer to new box time dirivatives.*/
    // parameters
    double timestep,   /**< Time step. */
    double piston_mass /**< Piston mass, dimensions of mass/length^4. */
    )
    : box(box), old_box_dot(old_box_dot), virial(virial),
      new_box_dot(new_box_dot), timestep(timestep), piston_mass(piston_mass) {
  initializeGraph();
}

void VolumePistonKick::initializeGraph(void) {
  // create parameters
  myparams = {0};
  kernelArgs = {(void *)&box,      (void *)&old_box_dot,
                (void *)&virial,   (void *)&new_box_dot,
                (void *)&timestep, (void *)&piston_mass};
  myparams.func = (void *)volumePistonKickKernel;
  myparams.gridDim = dim3(1, 1, 1);
  myparams.blockDim = dim3(1, 1, 1);
  myparams.sharedMemBytes = 0;
  myparams.kernelParams = (void **)(kernelArgs.data());
  myparams.extra = NULL;
  // Add nodes
  cudaCheck(cudaGraphAddKernelNode(&mynode, graph, NULL, 0, &myparams));
}
