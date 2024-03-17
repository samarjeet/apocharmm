// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include <iostream>

//
// Calculates restraint forces
//

template <typename AT, typename CT> class CudaRestraintForce {

public:
  CudaRestraintForce();
  void calc_force();

private:
  int i;
};
