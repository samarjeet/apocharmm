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
 * \brief Graphs and kernels to update the atom velocities by a force
 */
#pragma once
#include <CudaIntegratorGraph.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdint.h>

/** Update atom velocities based on their invese mass, their force, and the
 * timestep.*/
__global__ void atomVelocityKickSimpleKernel(
    // inputs
    const double4
        *old_vel_mass, /**< Device pointer to atom velocity and mass.*/
    const double4 *__restrict__ force_invmass, /**< Device pointer to force and
                                                  inverse mass of atoms.*/
    // outputs
    double *new_vel_mass, /**< Device pointer to updated atom velocities.*/
    // parameters
    int numAtoms,   /**< Number of atoms.*/
    double timestep /**< Time step.*/
);

/** Create a graph that updates the atom velocities based on the forces.*/
class AtomVelocityKick(
    public
    : AtomVelocityKick(
        // inputs
        const double4
            *old_vel_mass, /**< Device pointer to atom velocity and mass.*/
        const double4
            *__restrict__ force_invmass, /**< Device pointer to force and
                                            inverse mass of atoms.*/
        // outputs
        double *new_vel_mass, /**< Device pointer to updated atom velocities.*/
        // parameters
        int numAtoms,   /**< Number of atoms.*/
        double timestep /**< Time step.*/
    );
    private
    : void initializeGraph(void);
    cudaKernelNodeParams myparams; std::array<void *, 8> kernelArgs;
    cudaGraphNode_t mynode;
    // inputs
    const double4
        *old_vel_mass; /**< Device pointer to atom velocity and mass.*/
    const double4 *__restrict__ force_invmass; /**< Device pointer to force and
                                                  inverse mass of atoms.*/
    // outputs
    double *new_vel_mass; /**< Device pointer to updated atom velocities.*/
    // parameters
    int numAtoms;   /**< Number of atoms.*/
    double timestep /**< Time step.*/
);
