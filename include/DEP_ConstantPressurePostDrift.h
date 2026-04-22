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
 * \brief Recombine com coordinates and relative coordinates for next force
 * calculation.
 */
#pragma once
#include <CudaIntegratorGraph.h>
#include <PressureGroupsUtil.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/** Recombine center of mass coordinates and relative coordinates to get
 * absolute coordinates.*/
__global__ void constantPressurePostDriftSimpleKernel(
    // inputs
    const double4
        *__restrict__ relative_xyzq, /**< The device pointer to the array of
                                        atom positions relative to the
                                        center of mass of their pressure
                                        group.*/
    const double4
        *__restrict__ relative_vel_mass, /**< The device pointer to the array of
                                            relative velocity and mass of
                                            every atom.*/
    const ComID_t *__restrict__ com_ids, /**< The device pointer to the array of
                                            the position of the center of mass
                                            of each pressure group, and the
                                            atoms that are in the group.*/
    const double4
        *__restrict__ com_momentum_invmass, /**< The device pointer to the array
                                               of the group net momentums and
                                               inverse masses.*/
    // outputs
    double4 *__restrict__ absolute_xyzq, /**< The device pointer to the array of
                                            absolute atom positions.*/
    double4 *__restrict__ absolute_vel_mass, /**< The device pointer to the
                                                array of absolute velocity and
                                                mass of every atom.*/
    // parameters
    const int
        *sorted_atomids, /**< Array of all atom ids sorted by pressure group.*/
    int numGroups /**< number of groups.*/);

/** Create a graph that gets absolute coords back from relative coords.*/
class ConstantPressurePostDrift : public CudaIntegratorGraph {
public:
  /** Create a graph that gets absolute coords back from relative coords.*/
  ConstantPressurePostDrift(
      // inputs
      const double4
          *__restrict__ relative_xyzq, /**< The device pointer to the array of
                                          atom positions relative to
                                          the center of mass of their pressure
                                          group.*/
      const double4
          *__restrict__ relative_vel_mass, /**< The device pointer to the array
                                              of relative velocity and mass of
                                              every atom.*/
      const ComID_t *__restrict__ com_ids, /**< The device pointer to the array
                                              of the position of the center of
                                              mass of each pressure group, and
                                              the atoms that are in the group.*/
      const double4
          *__restrict__ com_momentum_invmass, /**< The device pointer to the
                                                 array of the group net
                                                 momentums and inverse masses.*/
      // outputs
      double4 *__restrict__ absolute_xyzq, /**< The device pointer to the array
                                              of absolute atom positions.*/
      double4 *__restrict__ absolute_vel_mass, /**< The device pointer to the
                                                  array of absolute velocity and
                                                  mass of every atom.*/
      // parameters
      const int *
          sorted_atomids, /**< Array of all atom ids sorted by pressure group.*/
      int numGroups /**< number of groups.*/);

private:
  void initializeGraph(void);
  cudaKernelNodeParams myparams;
  std::array<void *, 8> kernelArgs;
  cudaGraphNode_t mynode;
  // inputs
  const double4
      *__restrict__ relative_xyzq; /**< The device pointer to the array of atom
                                      positions relative to the center of mass
                                      of their pressure group.*/
  const double4 *__restrict__ relative_vel_mass; /**< The device pointer to the
                                                    array of relative velocity
                                                    and mass of every atom.*/
  const ComID_t *__restrict__ com_ids; /**< The device pointer to the array of
                                          the position of the center of mass of
                                          each pressure group, and the atoms
                                          that are in the group.*/
  const double4
      *__restrict__ com_momentum_invmass; /**< The device pointer to the array
                                             of the group net momentums and
                                             inverse masses.*/
  // outputs
  double4 *__restrict__ absolute_xyzq; /**< The device pointer to the array of
                                          absolute atom positions.*/
  double4 *__restrict__ absolute_vel_mass; /**< The device pointer to the array
                                              of absolute velocity and mass of
                                              every atom.*/
  // parameters
  const int
      *sorted_atomids; /**< Array of all atom ids sorted by pressure group.*/
  int numGroups;       /**< number of groups.*/
};
