// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include <CudaIntegratorGraph.h>
#include <cuda_runtime.h>
#include <cuda_utils.h>
/** Simple test that executes the graph*/
int main(void) {
  CudaIntegratorGraph myintegratorgraph;
  cudaGraph_t mygraph;
  mygraph = myintegratorgraph.getGraph();
  cudaGraphExec_t exec_graph;
  cudaCheck(cudaGraphInstantiate(&exec_graph, mygraph, NULL, NULL, 0));
  cudaCheck(cudaGraphLaunch(exec_graph, 0));
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGraphExecDestroy(exec_graph));
  return 0;
}
