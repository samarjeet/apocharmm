// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include <VolumePistonLeapfrogDrift.h>

/**Calculate new box lengths and scaling for center of mass coordinate,
 * momentums and the new kinetic energy.*/
__global__ void volumePistonLeapfrogDriftKernel(
    // inputs
    const double3 *old_com_ke, /**< Device pointer to kinetic energy of pressure
                                  group momentums*/
    const double3 *old_box,    /**< Device pointer to box lengths*/
    const double3
        *old_box_dot, /**< Device pointer to box length time dirivatives*/
    // outputs
    double3 *new_com_ke,  /**< Device pointer to save post drift pressure group
                             momentum kinetic energy*/
    double3 *new_box,     /**< Device pointer to save post drift box lengths*/
    double3 *new_box_dot, /**< Device pointer to save post drift box length time
                             dirivatives*/
    double3 *__restrict__ com_momentum_scale,    /**< Device pointer to final
            pressure group momentum scale.
            This is the total scale from predrift p group momentum to post drift p
            group momentum.*/
    double3 *__restrict__ com_momentum_prescale, /**< Device pointer to initial
         pressure group momentum scale. This is the scaleing of momentum that
         happens before the pressure group centers of mass are moved.*/
    double3 *__restrict__ com_position_prescale, /**< Device pointer to initial
         pressure group COM scale. This is the scaling p group centers of mass
         before the are moved by their prescaled momenentums.*/
    // parameters
    double ref_pressure, /**< Reference pressure, units are not atms, it is SI
                            compatible with other units.*/
    double timestep,     /**< Time step.*/
    double piston_mass   /**< Piston mass, dimensions of mass/length^4. */
) {
  // reading inputs
  double piston_invmass = 1.0 / piston_mass;
  double3 box = *old_box;
  double3 box_dot = *old_box_dot;
  double3 com_ke = *old_com_ke;
  double ke = com_ke.x + com_ke.y + com_ke.z;
  double volume0 = box.x * box.y * box.z;
  double volume_dot = box_dot.x * box.y * box.z + box.x * box_dot.y * box.z +
                      box.x * box.y * box_dot.z;
  // actual integration of piston
  volume_dot += piston_invmass * (2.0 / 3.0 * ke / volume0 - ref_pressure) *
                timestep * 0.5;
  double volume1 = volume0 + volume_dot * timestep * 0.5;
  double volume2 = volume0 + volume_dot * timestep;
  double momentum_scale = cbrt(volume0 / volume2);
  ke *= momentum_scale * momentum_scale;
  volume_dot += piston_invmass * (2.0 / 3.0 * ke / volume2 - ref_pressure) *
                timestep * 0.5;
  // write outputs
  double ke_scale = momentum_scale * momentum_scale;
  double position_scale = cbrt(volume2 / volume0);
  new_com_ke->x = com_ke.x * ke_scale;
  new_com_ke->y = com_ke.y * ke_scale;
  new_com_ke->z = com_ke.z * ke_scale;
  box.x *= position_scale;
  box.y *= position_scale;
  box.z *= position_scale;
  *new_box = box;
  new_box_dot->x = volume_dot / 3.0 / box.y / box.z;
  new_box_dot->y = volume_dot / 3.0 / box.x / box.z;
  new_box_dot->z = volume_dot / 3.0 / box.x / box.y;
  com_momentum_scale->x = momentum_scale;
  com_momentum_scale->y = momentum_scale;
  com_momentum_scale->z = momentum_scale;
  double momentum_prescale = cbrt(volume2 * volume0 / volume1 / volume1);
  com_momentum_prescale->x = momentum_prescale;
  com_momentum_prescale->y = momentum_prescale;
  com_momentum_prescale->z = momentum_prescale;
  com_position_prescale->x = position_scale;
  com_position_prescale->y = position_scale;
  com_position_prescale->z = position_scale;
}

VolumePistonLeapfrogDrift::VolumePistonLeapfrogDrift(
    // inputs
    const double3 *old_com_ke, /**< Device pointer to kinetic energy of pressure
                                  group momentums*/
    const double3 *old_box,    /**< Device pointer to box lengths*/
    const double3
        *old_box_dot, /**< Device pointer to box length time dirivatives*/
    // outputs
    double3 *new_com_ke,  /**< Device pointer to save post drift pressure group
                             momentum kinetic energy*/
    double3 *new_box,     /**< Device pointer to save post drift box lengths*/
    double3 *new_box_dot, /**< Device pointer to save post drift box length time
                             dirivatives*/
    double3 *__restrict__ com_momentum_scale,    /**< Device pointer to final
            pressure group momentum scale.
            This is the total scale from predrift p group momentum to post drift p
            group momentum.*/
    double3 *__restrict__ com_momentum_prescale, /**< Device pointer to initial
         pressure group momentum scale. This is the scaleing of momentum that
         happens before the pressure group centers of mass are moved.*/
    double3 *__restrict__ com_position_prescale, /**< Device pointer to initial
         pressure group COM scale. This is the scaling p group centers of mass
         before the are moved by their prescaled momenentums.*/
    // parameters
    double ref_pressure, /**< Reference pressure, units are not atms, it is SI
                            compatible with other units.*/
    double timestep,     /**< Time step.*/
    double piston_mass   /**< Piston mass, dimensions of mass/length^4. */
    )
    : old_com_ke(old_com_ke), old_box(old_box), old_box_dot(old_box_dot),
      new_com_ke(new_com_ke), new_box(new_box), new_box_dot(new_box_dot),
      com_momentum_scale(com_momentum_scale),
      com_momentum_prescale(com_momentum_prescale),
      com_position_prescale(com_position_prescale), ref_pressure(ref_pressure),
      timestep(timestep), piston_mass(piston_mass) {
  initializeGraph();
}

void VolumePistonLeapfrogDrift::initializeGraph(void) {
  // create parameters
  myparams = {0};
  kernelArgs = {(void *)&old_com_ke,
                (void *)&old_box,
                (void *)&old_box_dot,
                (void *)&new_com_ke,
                (void *)&new_box,
                (void *)&new_box_dot,
                (void *)&com_momentum_scale,
                (void *)&com_momentum_prescale,
                (void *)&com_position_prescale,
                (void *)&ref_pressure,
                (void *)&timestep,
                (void *)&piston_mass};
  myparams.func = (void *)volumePistonLeapfrogDriftKernel;
  myparams.gridDim = dim3(1, 1, 1);
  myparams.blockDim = dim3(1, 1, 1);
  myparams.sharedMemBytes = 0;
  myparams.kernelParams = (void **)(kernelArgs.data());
  myparams.extra = NULL;
  // Add nodes
  cudaCheck(cudaGraphAddKernelNode(&mynode, graph, NULL, 0, &myparams));
}
