// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, Nathan Zimmerberg
//
// ENDLICENSE

/** \file
 * \author Nathan Zimmerberg (nhz2@cornell.edu)
 * \date 07/16/2019
 * \brief Base class for cuda integrator graphs and some utilities for making
 * nodes from kernals easily.
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda_utils.h>
/**Base class for other cuda integrator classes. */
class CudaIntegratorGraph {
public:
  /** Create a graph with one empty node.
   */
  CudaIntegratorGraph();
  /** Return the graph.
   */
  cudaGraph_t getGraph();
  /** Destroy the graph and its nodes.
   */
  ~CudaIntegratorGraph();

protected:
  cudaGraph_t graph;
};
