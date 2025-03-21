// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaExternalForce.h"
#include <iostream>

template <typename AT, typename CT>
CudaExternalForce<AT, CT>::CudaExternalForce() {}

template <typename AT, typename CT>
void CudaExternalForce<AT, CT>::calc_force() {
  std::cout << "Calculating external force\n";
}

template class CudaExternalForce<long long int, float>;
