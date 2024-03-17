// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Nathan Zimmerberg, Samarjeet Prasad
//
// ENDLICENSE

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <PressureGroupVirialGraph.h>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <stdlib.h>
TEST_CASE("Pressure group simple kernel") {
  int numAtoms = 1 << 20;
  double *forcex_h;
  double *forcey_h;
  double *forcez_h;
  double *forcex_d;
  double *forcey_d;
  double *forcez_d;
  double *coordsx_h;
  double *coordsy_h;
  double *coordsz_h;
  double *coordsx_d;
  double *coordsy_d;
  double *coordsz_d;
  double3 *virial_d;
  double3 virial;
  double3 *virial_h = &virial;
  forcex_h = (double *)malloc(numAtoms * sizeof(double));
  forcey_h = (double *)malloc(numAtoms * sizeof(double));
  forcez_h = (double *)malloc(numAtoms * sizeof(double));
  cudaCheck(cudaMalloc((void **)&forcex_d, numAtoms * sizeof(double)));
  cudaCheck(cudaMalloc((void **)&forcey_d, numAtoms * sizeof(double)));
  cudaCheck(cudaMalloc((void **)&forcez_d, numAtoms * sizeof(double)));
  cudaCheck(cudaMalloc((void **)&virial_d, sizeof(double3)));
  coordsx_h = (double *)malloc(numAtoms * sizeof(double));
  coordsy_h = (double *)malloc(numAtoms * sizeof(double));
  coordsz_h = (double *)malloc(numAtoms * sizeof(double));
  cudaCheck(cudaMalloc((void **)&coordsx_d, numAtoms * sizeof(double)));
  cudaCheck(cudaMalloc((void **)&coordsy_d, numAtoms * sizeof(double)));
  cudaCheck(cudaMalloc((void **)&coordsz_d, numAtoms * sizeof(double)));
  // create atom coords and forces randomly
  srand48(2);
  for (int i = 0; i < numAtoms; i++) {
    forcex_h[i] = drand48() - 0.5;
    forcey_h[i] = drand48() - 0.5;
    forcez_h[i] = drand48() - 0.5;
    coordsx_h[i] = drand48() - 0.5;
    coordsy_h[i] = drand48() - 0.5;
    coordsz_h[i] = drand48() - 0.5;
  }
  cudaCheck(cudaMemcpy(forcex_d, forcex_h, numAtoms * sizeof(double),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(forcey_d, forcey_h, numAtoms * sizeof(double),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(forcez_d, forcez_h, numAtoms * sizeof(double),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(coordsx_d, coordsx_h, numAtoms * sizeof(double),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(coordsy_d, coordsy_h, numAtoms * sizeof(double),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(coordsz_d, coordsz_h, numAtoms * sizeof(double),
                       cudaMemcpyHostToDevice));
  SECTION("virial of lots of random atoms -0.5 to 0.5") {
    PressureGroupVirialGraph mygraphclass(forcex_d, forcey_d, forcez_d,
                                          coordsx_d, coordsy_d, coordsz_d,
                                          numAtoms, virial_d);
    cudaGraph_t mygraph;
    mygraph = mygraphclass.getGraph();
    cudaGraphExec_t exec_graph;
    cudaCheck(cudaGraphInstantiate(&exec_graph, mygraph, NULL, NULL, 0));
    cudaCheck(cudaGraphLaunch(exec_graph, 0));
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(virial_h, virial_d, sizeof(double3),
                         cudaMemcpyDeviceToHost));

    cudaCheck(cudaGraphExecDestroy(exec_graph));
    double virialxx = 0.0;
    double virialyy = 0.0;
    double virialzz = 0.0;
    for (int i = 0; i < numAtoms; i++) {
      virialxx += forcex_h[i] * coordsx_h[i];
      virialyy += forcey_h[i] * coordsy_h[i];
      virialzz += forcez_h[i] * coordsz_h[i];
    }
    REQUIRE(virialxx == Approx(virial_h->x));
    REQUIRE(virialyy == Approx(virial_h->y));
    REQUIRE(virialzz == Approx(virial_h->z));
  }

  SECTION(
      "virial of lots of random atoms manuel graph of simple kernel -0.5 to "
      "0.5") {
    // cudaGraph_t graph;
    // cudaCheck(cudaGraphCreate(&graph,0));
    // //create parameters
    // cudaKernelNodeParams myparams={0};
    // void *kernelArgs[8]=
    // {(void*)&forcex_d,(void*)&forcey_d,(void*)&forcez_d,(void*)&coordsx_d,(void*)&coordsy_d,(void*)&coordsz_d,(void*)&numAtoms,(void*)&virial_d};//an
    // array of void pointers
    // myparams.func=(void*)pressureGroupVirialSimpleKernel;
    // myparams.gridDim=dim3(1,1,1);
    // myparams.blockDim=dim3(1,1,1);
    // myparams.sharedMemBytes=0;
    // myparams.kernelParams= (void **)kernelArgs;
    // myparams.extra=NULL;
    // //Add nodes
    // cudaGraphNode_t mynode;
    // cudaCheck(cudaGraphAddKernelNode(&mynode,graph,NULL,0,&myparams));
    // cudaGraphExec_t exec_graph;
    // cudaCheck(cudaGraphInstantiate(&exec_graph,graph,NULL,NULL,0));
    // cudaCheck(cudaGraphLaunch(exec_graph,0));
    // cudaCheck(cudaDeviceSynchronize());
    // cudaCheck(cudaGraphExecDestroy(exec_graph));
    // cudaCheck(cudaGraphDestroy(graph));
    // REQUIRE(1==1);
  }
}
