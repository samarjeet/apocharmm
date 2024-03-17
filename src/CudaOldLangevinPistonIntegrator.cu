// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaLangevinPistonIntegrator.h"
#include <PrintEnergiesGraph.h>
#include <SimpleLeapfrogGraph.h>
#include <VolumePiston.h>
#include <cuda_utils.h>
#include <iostream>

CudaLangevinPistonIntegrator::CudaLangevinPistonIntegrator(double timeStep)
    : CudaIntegrator(timeStep) {}

extern __global__ void printKernel(int numAtoms, float4 *array);
extern __global__ void printKernel(int numAtoms, double4 *array);

void CudaLangevinPistonIntegrator::initialize() {}
__global__ void imageCenteringKernel(double4 *__restrict__ coords,
                                     const double3 *__restrict__ box,
                                     int numAtoms) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numAtoms) {
    while (coords[tid].x > box->x / 2) {
      coords[tid].x -= box->x;
    }
    while (coords[tid].y > box->y / 2) {
      coords[tid].y -= box->y;
    }
    while (coords[tid].z > box->z / 2) {
      coords[tid].z -= box->z;
    }

    while (coords[tid].x < -box->x / 2) {
      coords[tid].x += box->x;
    }
    while (coords[tid].y < -box->y / 2) {
      coords[tid].y += box->y;
    }
    while (coords[tid].z < -box->z / 2) {
      coords[tid].z += box->z;
    }
  }
}

__global__ void piston_vel_update(VolumePiston piston,
                                  const double3 *__restrict__ virial,
                                  const double3 *__restrict__ com_ke,
                                  const double3 *__restrict__ box,
                                  double3 *__restrict__ box_dot,
                                  double timestep) {
  piston.virialKick(virial, box, box_dot, box_dot, timestep);
  piston.kineticEnergyKick(com_ke, box, box_dot, box_dot, timestep);
  piston.pressureKick(box, box_dot, box_dot, timestep);
}

__global__ void drift(VolumePiston piston, double4 *__restrict__ xyz,
                      double4 *__restrict__ velmass, double3 *__restrict__ box,
                      double3 *__restrict__ box_dot, double timestep,
                      int numAtoms) {
  double3 dummyke;
  double3 rprescale;
  double3 vprescale;
  double3 vscale;
  piston.leapfrogdrift(&dummyke, box, box_dot, &dummyke, box, box_dot, &vscale,
                       &vprescale, &rprescale, timestep);
  for (int i = 0; i < numAtoms; i++) {
    xyz[i].x *= rprescale.x;
    xyz[i].x += velmass[i].x * timestep * vprescale.x;
    velmass[i].x *= vscale.x;
    xyz[i].y *= rprescale.y;
    xyz[i].y += velmass[i].y * timestep * vprescale.y;
    velmass[i].y *= vscale.y;
    xyz[i].z *= rprescale.z;
    xyz[i].z += velmass[i].z * timestep * vprescale.z;
    velmass[i].z *= vscale.z;
    printf("atom %d position,%f,%f,%f\n", i, xyz[i].x, xyz[i].y, xyz[i].z);
  }
  printf("vprescale is,%.16f,%.16f,%.16f\n", vprescale.x, vprescale.y,
         vprescale.z);
}

__global__ void vel_update(double4 *__restrict__ force_invmass,
                           double4 *__restrict__ velmass, double timestep,
                           int numAtoms) {
  for (int i = 0; i < numAtoms; i++) {
    double invmass = force_invmass[i].w;
    velmass[i].x += timestep * invmass * force_invmass[i].x;
    velmass[i].y += timestep * invmass * force_invmass[i].y;
    velmass[i].z += timestep * invmass * force_invmass[i].z;
  }
}

__global__ void calc_ke(const double4 *__restrict__ velmass,
                        double3 *__restrict__ ke, int numAtoms) {
  double3 ke3;
  ke3.x = 0.0;
  ke3.y = 0.0;
  ke3.z = 0.0;
  for (int i = 0; i < numAtoms; i++) {
    double mass = velmass[i].w;
    ke3.x += 0.5 * velmass[i].x * velmass[i].x * mass;
    ke3.y += 0.5 * velmass[i].y * velmass[i].y * mass;
    ke3.z += 0.5 * velmass[i].z * velmass[i].z * mass;
  }
  ke->x = ke3.x;
  ke->y = ke3.y;
  ke->z = ke3.z;
}

void CudaLangevinPistonIntegrator::propagateOneStep(){

}
/*
void CudaLangevinPistonIntegrator::propagate(int numSteps) {
  int numAtoms = simulationContext->getNumAtoms();
  int nThreads = 128;
  // int nThreads = 512;
  int nBlocks = (numAtoms - 1) / nThreads + 1;
  auto piston = simulationContext->piston;
  double4* xyzq_d = simulationContext->xyz.d;
  double4* vel_mass_d = simulationContext->vel_mass.d;
  double4* force_invmass_d = simulationContext->force_invmass.d;
  double3* virial_d = simulationContext->virial.d;
  double3* box_d = simulationContext->box.d;
  double3* box_dot_d = simulationContext->box_dot.d;
  double* pe_d = simulationContext->potential_energy.d;
  SimpleLeapfrogGraphInputs inputs = {xyzq_d,  vel_mass_d, force_invmass_d,
                                      xyzq_d,  vel_mass_d, numAtoms,
                                      timeStep};
  SimpleLeapfrogGraph leapfrogclass(inputs);
  PrintEnergiesGraphInputs in;
  in.velmass = vel_mass_d;
  in.potential_energy = pe_d;
  in.box = box_d;
  in.box_dot = box_dot_d;
  in.piston = simulationContext->piston;
  in.numAtoms = numAtoms;
  H_DVector<double3> com_ke(1);
  PrintEnergiesGraph printclass(in);
  cudaGraph_t leapfrog_graph;
  cudaGraphNode_t leapfrog_node;
  cudaGraphNode_t print_node;
  cudaGraph_t mygraph;
  cudaGraph_t print_graph;
  print_graph = printclass.getGraph();
  leapfrog_graph = leapfrogclass.getGraph();
  // add child graphs to mygraph
  cudaCheck(cudaGraphCreate(&mygraph, 0));
  // cudaCheck(cudaGraphAddChildGraphNode ( &leapfrog_node, mygraph, NULL , 0,
  // leapfrog_graph ));
  cudaCheck(
      cudaGraphAddChildGraphNode(&print_node, mygraph, NULL, 0, print_graph));
  // create executable graph
  cudaGraphExec_t exec_graph;
  cudaCheck(cudaGraphInstantiate(&exec_graph, mygraph, NULL, NULL, 0));

  // numSteps = 10000;
  for (int i = 0; i < numSteps; ++i) {
      if (i % 20 == 0) {
          if (i % 100 == 0) {
              imageCenteringKernel<<<nBlocks, nThreads>>>(xyzq_d, box_d,
                                                          numAtoms);
          }
          cudaCheck(cudaDeviceSynchronize());
          simulationContext->calculatePotentialEnergy(true);
      } else {
          simulationContext->calculatePotentialEnergy(false);
      }
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGraphLaunch(exec_graph, 0));
      cudaCheck(cudaDeviceSynchronize());
      calc_ke<<<1, 1>>>(vel_mass_d, com_ke.d, numAtoms);
      piston_vel_update<<<1, 1>>>(piston, virial_d, com_ke.d, box_d,
                                  box_dot_d, timeStep * 0.5);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGraphLaunch(exec_graph, 0));
      cudaCheck(cudaDeviceSynchronize());
      vel_update<<<1, 1>>>(force_invmass_d, vel_mass_d, timeStep * 0.5,
                           numAtoms);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGraphLaunch(exec_graph, 0));
      cudaCheck(cudaDeviceSynchronize());
      vel_update<<<1, 1>>>(force_invmass_d, vel_mass_d, timeStep * 0.5,
                           numAtoms);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGraphLaunch(exec_graph, 0));
      cudaCheck(cudaDeviceSynchronize());
      calc_ke<<<1, 1>>>(vel_mass_d, com_ke.d, numAtoms);
      piston_vel_update<<<1, 1>>>(piston, virial_d, com_ke.d, box_d,
                                  box_dot_d, timeStep * 0.5);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGraphLaunch(exec_graph, 0));
      cudaCheck(cudaDeviceSynchronize());
      drift<<<1, 1>>>(piston, xyzq_d, vel_mass_d, box_d, box_dot_d, timeStep,
                      numAtoms);
      cudaCheck(cudaDeviceSynchronize());
      std::cout << "\nstep," << i << "\n";
  }
}
*/

