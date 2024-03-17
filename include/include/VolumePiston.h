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
 * \brief Class to hold constant pressure volume piston parameters and helper
 * functions
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/**Struct to hold the volume piston parameters and some functions to help with
 * dynamics.*/
struct VolumePiston {
  double pressure;    /**< Reference pressure, units are not atms, it is SI
                         compatible with other units.*/
  double piston_mass; /**< Piston Mass, dimensions of mass/length^4.*/
  inline CUDA_CALLABLE_MEMBER double calcpe(double3 box) const {
    return box.x * box.y * box.z * pressure;
  }
  inline CUDA_CALLABLE_MEMBER double calcke(double3 box,
                                            double3 box_dot) const {
    double volume_dot = box_dot.x * box.y * box.z + box.x * box_dot.y * box.z +
                        box.x * box.y * box_dot.z;
    return 0.5 * piston_mass * volume_dot * volume_dot;
  }
  inline CUDA_CALLABLE_MEMBER void leapfrogdrift(
      // inputs
      const double3 *old_com_ke,  /**< Pointer to kinetic energy of pressure
                                     group momentums*/
      const double3 *old_box,     /**< Pointer to box lengths*/
      const double3 *old_box_dot, /**< Pointer to box length time dirivatives*/
      // outputs
      double3 *new_com_ke,  /**< Pointer to save post drift pressure group
                               momentum kinetic energy*/
      double3 *new_box,     /**< Pointer to save post drift box lengths*/
      double3 *new_box_dot, /**< Pointer to save post drift box length time
                               dirivatives*/
      double3 *__restrict__ com_momentum_scale, /**< Pointer to final pressure
           group momentum scale. This is the total scale from predrift p group
           momentum to post drift p group momentum.*/
      double3
          *__restrict__ com_momentum_prescale, /**< Pointer to initial pressure
       group momentum scale. This is the scaleing of momentum that happens
       before the pressure group centers of mass are moved.*/
      double3
          *__restrict__ com_position_prescale, /**< Pointer to initial pressure
       group COM scale. This is the scaling p group centers of mass before the
       are moved by their prescaled momenentums.*/
      // parameters
      double timestep /**< Time step.*/
      ) const {
    double3 box = *old_box;
    double3 box_dot = *old_box_dot;
    double3 com_ke = *old_com_ke;
    double ke = com_ke.x + com_ke.y + com_ke.z;
    double volume0 = box.x * box.y * box.z;
    double volume_dot = box_dot.x * box.y * box.z + box.x * box_dot.y * box.z +
                        box.x * box.y * box_dot.z;
    // actual integration of piston
    double volume1 = volume0 + volume_dot * timestep * 0.5;
    double volume2 = volume0 + volume_dot * timestep;
    double momentum_scale = cbrt(volume0 / volume2);
    ke *= momentum_scale * momentum_scale;
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
  inline CUDA_CALLABLE_MEMBER void pressureKick(
      // inputs
      const double3 *__restrict__ box, /**< Pointer to box lengths*/
      const double3 *old_box_dot, /**< Pointer to box length time dirivatives*/
      // outputs
      double3 *new_box_dot, /**< Pointer to save post drift box length time
                               dirivatives*/
      // parameters
      double timestep /**< Time step.*/
      ) const {
    double3 b = *box;
    double3 box_dot = *old_box_dot;
    double volume_dot =
        box_dot.x * b.y * b.z + b.x * box_dot.y * b.z + b.x * b.y * box_dot.z;
    // actual integration of piston
    volume_dot += (-pressure) / piston_mass * timestep;
    // write outputs
    new_box_dot->x = volume_dot / 3.0 / b.y / b.z;
    new_box_dot->y = volume_dot / 3.0 / b.x / b.z;
    new_box_dot->z = volume_dot / 3.0 / b.x / b.y;
  }
  inline CUDA_CALLABLE_MEMBER void virialKick(
      // inputs
      const double3 *__restrict__ virial, /**< Pointer to corrected virial*/
      const double3 *__restrict__ box,    /**< Pointer to box lengths*/
      const double3 *old_box_dot, /**< Pointer to box length time dirivatives*/
      // outputs
      double3 *new_box_dot, /**< Pointer to save post drift box length time
                               dirivatives*/
      // parameters
      double timestep /**< Time step.*/
      ) const {
    double3 b = *box;
    double3 box_dot = *old_box_dot;
    double volume = b.x * b.y * b.z;
    double volume_dot =
        box_dot.x * b.y * b.z + b.x * box_dot.y * b.z + b.x * b.y * box_dot.z;
    // actual integration of piston
    volume_dot += (-virial->x - virial->y - virial->z) / volume / 3.0 /
                  piston_mass * timestep;
    // write outputs
    new_box_dot->x = volume_dot / 3.0 / b.y / b.z;
    new_box_dot->y = volume_dot / 3.0 / b.x / b.z;
    new_box_dot->z = volume_dot / 3.0 / b.x / b.y;
  }
  inline CUDA_CALLABLE_MEMBER void kineticEnergyKick(
      // inputs
      const double3 *__restrict__ com_ke, /**< Pointer to kinetic energy of
                                             pressure group momentums*/
      const double3 *__restrict__ box,    /**< Pointer to box lengths*/
      const double3 *old_box_dot, /**< Pointer to box length time dirivatives*/
      // outputs
      double3 *new_box_dot, /**< Pointer to save post drift box length time
                               dirivatives*/
      // parameters
      double timestep /**< Time step.*/
      ) const {
    double3 b = *box;
    double3 box_dot = *old_box_dot;
    double ke = com_ke->x + com_ke->y + com_ke->z;
    double volume = b.x * b.y * b.z;
    double volume_dot =
        box_dot.x * b.y * b.z + b.x * box_dot.y * b.z + b.x * b.y * box_dot.z;
    // actual integration of piston
    volume_dot += ke / volume * 2.0 / 3.0 / piston_mass * timestep;
    // write outputs
    new_box_dot->x = volume_dot / 3.0 / b.y / b.z;
    new_box_dot->y = volume_dot / 3.0 / b.x / b.z;
    new_box_dot->z = volume_dot / 3.0 / b.x / b.y;
  }
};
