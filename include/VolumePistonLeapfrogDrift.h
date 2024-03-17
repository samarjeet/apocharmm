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
 * \brief this is where the volume updates are calculated for a leapfrog
 * integrator during the drift.
 */
#pragma once
#include <CudaIntegratorGraph.h>
#include <PressureGroupsUtil.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/**Calculate new box lengths and scaling for center of mass coordinate,
 * momentums and the new kinetic energy. Only use one thread for this kernel.
 * Uses The Anderson Barostat, with homogenous expansions.*/
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
    double3
        *__restrict__ com_momentum_scale, /**< Device pointer to final pressure
     group momentum scale. This is the total scale from predrift p group
     momentum to post drift p group momentum.*/
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
    double piston_mass   /**< Piston Mass, dimensions of mass/length^4. */
);
/** Create a graph that calculates the effect of constant pressure during the
 * drift*/
class VolumePistonLeapfrogDrift : public CudaIntegratorGraph {
public:
  /** 
   * @brief Create a graph that calculates the effect of constant pressure
   */
  VolumePistonLeapfrogDrift(
      // inputs
      const double3 *old_com_ke, /**< Device pointer to kinetic energy of
                                    pressure group momentums*/
      const double3 *old_box,    /**< Device pointer to box lengths*/
      const double3
          *old_box_dot, /**< Device pointer to box length time dirivatives*/
      // outputs
      double3 *new_com_ke, /**< Device pointer to save post drift pressure group
                              momentum kinetic energy*/
      double3 *new_box,    /**< Device pointer to save post drift box lengths*/
      double3 *new_box_dot, /**< Device pointer to save post drift box length
                               time dirivatives*/
      double3
          *__restrict__ com_momentum_scale, /**< Device pointer to final
       pressure group momentum scale. This is the total scale from predrift p
       group momentum to post drift p group momentum.*/
      double3
          *__restrict__ com_momentum_prescale, /**< Device pointer to initial
       pressure group momentum scale. This is the scaleing of momentum that
       happens before the pressure group centers of mass are moved.*/
      double3
          *__restrict__ com_position_prescale, /**< Device pointer to initial
       pressure group COM scale. This is the scaling p group centers of mass
       before the are moved by their prescaled momenentums.*/
      // parameters
      double ref_pressure, /**< Reference pressure, units are not atms, it is SI
                              compatible with other units.*/
      double timestep,     /**< Time step.*/
      double piston_mass /**< Piston Mass, dimensions of mass/length^4. */
         );
private:
  void initializeGraph(void);
  cudaKernelNodeParams myparams;
  std::array<void *, 12> kernelArgs;
  cudaGraphNode_t mynode;
  // inputs
  const double3 *old_com_ke; /**< Device pointer to kinetic energy of pressure
                                group momentums*/
  const double3 *old_box;    /**< Device pointer to box lengths*/
  const double3
      *old_box_dot; /**< Device pointer to box length time dirivatives*/
  // outputs
  double3 *new_com_ke;  /**< Device pointer to save post drift pressure group
                           momentum kinetic energy*/
  double3 *new_box;     /**< Device pointer to save post drift box lengths*/
  double3 *new_box_dot; /**< Device pointer to save post drift box length time
                           dirivatives*/
  double3 *__restrict__ com_momentum_scale;    /**< Device pointer to final
          pressure group momentum scale.    This is the total scale from predrift p
          group momentum to post drift p group momentum.*/
  double3 *__restrict__ com_momentum_prescale; /**< Device pointer to initial
       pressure group momentum scale. This is the scaleing of momentum that
       happens before the pressure group centers of mass are moved.*/
  double3 *__restrict__ com_position_prescale; /**< Device pointer to initial
       pressure group COM scale. This is the scaling p group centers of mass
       before the are moved by their prescaled momenentums.*/
  // parameters
  double ref_pressure; /**< Reference pressure, units are not atms, it is SI
                          compatible with other units.*/
  double timestep;     /**< Time step.*/
  double piston_mass;  /**< Piston Mass, dimensions of mass/length^4. */
};
