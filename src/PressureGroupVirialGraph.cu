// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Nathan Zimmerberg, Samarjeet Prasad
//
// ENDLICENSE

#include <PressureGroupVirialGraph.h>
__global__ void
pressureGroupVirialSimpleKernel(const double *__restrict__ forcex,
                                const double *__restrict__ forcey,
                                const double *__restrict__ forcez,
                                const double *__restrict__ relative_coordsx,
                                const double *__restrict__ relative_coordsy,
                                const double *__restrict__ relative_coordsz,
                                int numAtoms, double3 *__restrict__ virial) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid == 0) {
    double virialxx = 0.0;
    double virialyy = 0.0;
    double virialzz = 0.0;
    for (int i = 0; i < numAtoms; i++) {
      virialxx += forcex[i] * relative_coordsx[i];
      virialyy += forcey[i] * relative_coordsy[i];
      virialzz += forcez[i] * relative_coordsz[i];
    }
    virial->x = virialxx;
    virial->y = virialyy;
    virial->z = virialzz;
  }
}

PressureGroupVirialGraph::PressureGroupVirialGraph(
    const double *__restrict__ fx, const double *__restrict__ fy,
    const double *__restrict__ fz, const double *__restrict__ x,
    const double *__restrict__ y, const double *__restrict__ z, int numAtoms,
    double3 *__restrict__ virial)
    : fx(fx), fy(fy), fz(fz), x(x), y(y), z(z), numAtoms(numAtoms),
      virial(virial) {
  initializeGraph();
  // cudaStream_t stream1;
  // cudaCheck(cudaStreamCreate(&stream1));
  // cudaCheck(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));
  // pressureGroupVirialSimpleKernel<<<1, 1, 0, stream1>>>(
  //          forcex,      forcey,      forcez,      relative_coordsx,
  //          relative_coordsy,      relative_coordsz, numAtoms,      virial);
  // cudaCheck(cudaStreamEndCapture(stream1, &graph));
  // cudaCheck(cudaStreamDestroy(stream1));

  // cudaCheck(cudaGraphCreate(&graph,0));
  ////create parameters
  // cudaKernelNodeParams myparams={0};
  // void *kernelArgs[8]=
  // {(void*)&forcex,(void*)&forcey,(void*)&forcez,(void*)&relative_coordsx,(void*)&relative_coordsy,(void*)&relative_coordsz,(void*)&numAtoms,(void*)&virial};//an
  // array of void pointers
  // myparams.func=(void*)pressureGroupVirialSimpleKernel;
  // myparams.gridDim=dim3(1,1,1);
  // myparams.blockDim=dim3(1,1,1);
  // myparams.sharedMemBytes=0;
  // myparams.kernelParams= (void **)kernelArgs;
  // myparams.extra=NULL;
  ////Add nodes
  // cudaGraphNode_t mynode;
  // cudaCheck(cudaGraphAddKernelNode(&mynode,graph,NULL,0,&myparams));
}

void PressureGroupVirialGraph::initializeGraph(void) {
  // create parameters
  myparams = {0};
  kernelArgs = {(void *)&fx,       (void *)&fy,    (void *)&fz,
                (void *)&x,        (void *)&y,     (void *)&z,
                (void *)&numAtoms, (void *)&virial}; // an array of void
                                                     // pointers
  myparams.func = (void *)pressureGroupVirialSimpleKernel;
  myparams.gridDim = dim3(1, 1, 1);
  myparams.blockDim = dim3(1, 1, 1);
  myparams.sharedMemBytes = 0;
  myparams.kernelParams = (void **)(kernelArgs.data());
  myparams.extra = NULL;
  // Add nodes
  cudaCheck(cudaGraphAddKernelNode(&mynode, graph, NULL, 0, &myparams));
}

PressureGroupVirialGraph::~PressureGroupVirialGraph() {}
