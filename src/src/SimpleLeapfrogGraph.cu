// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include <SimpleLeapfrogGraph.h>
__global__ void SimpleLeapfrogGraphKernel(SimpleLeapfrogGraphInputs in) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int gridstride = blockDim.x * gridDim.x;
  int i = tid;
  while (i < in.numAtoms) {
    double invmass = in.force_invmass[i].w;
    in.new_velmass[i].x =
        in.old_velmass[i].x + in.timestep * in.force_invmass[i].x * invmass;
    in.new_velmass[i].y =
        in.old_velmass[i].y + in.timestep * in.force_invmass[i].y * invmass;
    in.new_velmass[i].z =
        in.old_velmass[i].z + in.timestep * in.force_invmass[i].z * invmass;
    in.new_xyzq[i].x = in.old_xyzq[i].x + in.timestep * in.new_velmass[i].x;
    in.new_xyzq[i].y = in.old_xyzq[i].y + in.timestep * in.new_velmass[i].y;
    in.new_xyzq[i].z = in.old_xyzq[i].z + in.timestep * in.new_velmass[i].z;
    i += gridstride;
  }
}
