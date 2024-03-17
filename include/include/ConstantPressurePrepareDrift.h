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
 * \brief Class and kernels for Cuda graphs to prepare for the drift stage.
 * \details The graph will calculate the net momentums, center of masses(COM),
 * relative coordinates to the COMs, internal momentums, and kinetic energies
 * from the COM motion, and kinetic energies from the internal motions.
 *
 */
#pragma once
#include <CudaIntegratorGraph.h>
#include <PressureGroupsUtil.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/** calculate the net momentums, center of masses(COM), relative coordinates to
 * the COMs, internal momentums, and kinetic energies from the COM motion, and
 * kinetic energies from the internal motions.*/
__global__ void constantPressurePrepareDriftSimpleKernel(
    // inputs
    const double4
        *__restrict__ absolute_vel_mass, /**< The device pointer to the array of
                                            absolute velocity and mass of
                                            every atom.*/
    const double4
        *__restrict__ absolute_xyzq, /**< The device pointer to the array of
                                        absolute atom positions.*/
    // outputs
    double4
        *__restrict__ relative_xyzq, /**< The device pointer to the array of
                                        atom positions relative to the center
                                        of mass of their pressure group.*/
    double4 *__restrict__ relative_vel_mass, /**< The device pointer to the
                                                array of relative velocity and
                                                mass of every atom.*/
    ComID_t *__restrict__ com_ids, /**< The device pointer to the array of the
                                      position of the center of mass of each
                                      pressure group, and the atoms that are in
                                      the group.*/
    double4
        *__restrict__ com_momentum_invmass, /**< The device pointer to the array
                                               of the group net momentums and
                                               inverse masses.*/
    double3 *__restrict__ com_kinetic_energy,  /**< The xx,yy, and zz components
                                                  of the groups center of mass
                                                  kinetic  energy.*/
    double *__restrict__ total_kinetic_energy, /**< The total kinetic energy.*/
    // parameters
    const int
        *sorted_atomids, /**< Array of all atom ids sorted by pressure group.*/
    int numGroups /**< number of groups.*/);

/** Create a graph that calculates the net momentums, center of masses(COM),
 * relative coordinates to the COMs, internal momentums, and kinetic energies
 * from the COM motion, and kinetic energies from the internal motions.*/
class ConstantPressurePrepareDrift : public CudaIntegratorGraph {
public:
  /** Create a graph that calculate the net momentums, center of masses(COM),
   * relative coordinates to the COMs, internal momentums, and kinetic energies
   * from the COM motion, and kinetic energies from the internal motions. All
   * units are SI compatible.
   */
  ConstantPressurePrepareDrift(
      // inputs
      const double4
          *__restrict__ absolute_vel_mass, /**< The device pointer to the array
                                              of absolute velocity and mass of
                                              every atom.*/
      const double4
          *__restrict__ absolute_xyzq, /**< The device pointer to the array of
                                          absolute atom positions.*/
      // outputs
      double4 *__restrict__ relative_xyzq, /**< The device pointer to the array
                                              of atom positions relative to the
                                              center of mass of their pressure
                                              group.*/
      double4 *__restrict__ relative_vel_mass, /**< The device pointer to the
                                                  array of relative velocity and
                                                  mass of every atom.*/
      ComID_t *__restrict__ com_ids, /**< The device pointer to the array of the
                                        position of the center of mass of each
                                        pressure group, and the atoms that are
                                        in the group.*/
      double4
          *__restrict__ com_momentum_invmass, /**< The device pointer to the
                                                 array of the group net
                                                 momentums and inverse masses.*/
      double3
          *__restrict__ com_kinetic_energy, /**< The xx,yy, and zz components of
                                               the groups center of mass kinetic
                                               energy.*/
      double *__restrict__ total_kinetic_energy, /**< The total kinetic energy,
                                                    used for the thermostat
                                                    piston.*/
      // parameters
      const int *
          sorted_atomids, /**< Array of all atom ids sorted by pressure group.*/
      int numAtoms,       /**< number of atoms.*/
      int numGroups /**< number of groups.*/);
  /** Destroy the graph, its nodes, and free internal memory.*/
  ~ConstantPressurePrepareDrift();

private:
  void initializeGraph(void);
  cudaKernelNodeParams myparams;
  std::array<void *, 10> kernelArgs;
  cudaGraphNode_t mynode;
  // inputs
  const double4 *__restrict__ absolute_vel_mass; /**< The device pointer to the
                                                    array of absolute velocity
                                                    and mass of every atom.*/
  const double4
      *__restrict__ absolute_xyzq; /**< The device pointer to the array of
                                      absolute atom positions.*/
  // outputs
  double4 *__restrict__ relative_xyzq; /**< The device pointer to the array of
                                          atom positions relative to the center
                                          of mass of their pressure group.*/
  double4 *__restrict__ relative_vel_mass; /**< The device pointer to the array
                                              of relative velocity and mass of
                                              every atom.*/
  ComID_t *__restrict__ com_ids; /**< The device pointer to the array of the
                                    position of the center of mass of each
                                    pressure group, and the atoms that are in
                                    the group.*/
  double4 *__restrict__ com_momentum_invmass; /**< The device pointer to the
                                                 array of the group net
                                                 momentums and inverse masses.*/
  double3 *__restrict__ com_kinetic_energy; /**< The xx,yy, and zz components of
                                               the groups center of mass kinetic
                                               energy.*/
  double *__restrict__ total_kinetic_energy; /**< The total kinetic energy.*/
  // parameters
  const int
      *sorted_atomids; /**< Array of all atom ids sorted by pressure group.*/
  int numAtoms;        /**< number of atoms.*/
  int numGroups;       /**< number of groups.*/
};
