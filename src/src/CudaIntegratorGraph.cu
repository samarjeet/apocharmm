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
CudaIntegratorGraph::CudaIntegratorGraph() {
  cudaCheck(cudaGraphCreate(&graph, 0));
  cudaGraphNode_t emptynode;
  cudaCheck(cudaGraphAddEmptyNode(&emptynode, graph, NULL, 0));
}
cudaGraph_t CudaIntegratorGraph::getGraph() { return graph; }
CudaIntegratorGraph::~CudaIntegratorGraph() {
  cudaCheck(cudaGraphDestroy(graph));
}
