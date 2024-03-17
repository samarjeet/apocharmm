// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include <PrintEnergiesGraph.h>

__global__ void PrintEnergiesGraphKernel(PrintEnergiesGraphInputs in) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid == 0) {
    double ke = 0.0;
    for (int i = 0; i < in.numAtoms; i++) {
      double4 vm = in.velmass[i];
      ke += 0.5 * vm.w * (vm.x * vm.x + vm.y * vm.y + vm.z * vm.z);
    }
    printf("Potential Energy,%.17g,", *(in.potential_energy));
    printf("Kinetic Energy,%.17g,", ke);
    double boxpe = in.piston.calcpe(*in.box);
    double boxke = in.piston.calcke(*in.box, *in.box_dot);
    printf("Box Piston Potential Energy,%.17g,", boxpe);
    printf("Box Piston Kinetic Energy,%.17g\n", boxke);
    printf("Total Energy,%.17g\n", boxke + boxpe + ke + *(in.potential_energy));
    printf("Box_len,%.17g\n", (*in.box).x);
  }
}
