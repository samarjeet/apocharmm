// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "CudaRestraintForce.h"

template <typename AT, typename CT>
CudaRestraintForce<AT, CT>::CudaRestraintForce() {}

template <typename AT, typename CT>
void CudaRestraintForce<AT, CT>::calc_force() {
  std::cout << "Calculating restraint force\n";
}

template class CudaRestraintForce<long long int, float>;
