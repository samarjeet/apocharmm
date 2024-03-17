// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#pragma once

//
// Calculates external forces
//

template <typename AT, typename CT> class CudaExternalForce {

public:
  CudaExternalForce();
  void calc_force();

private:
  int i;
};
