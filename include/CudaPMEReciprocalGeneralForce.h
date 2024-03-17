// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include <vector>

class CudaPMEReciprocalGeneralForce {
private:
  int nfftx, nffty, nfftz;
  int order;
  double kappa;
  int numAtoms;

  std::vector<double> boxDimensions;

public:
  CudaPMEReciprocalGeneralForce();
  void setFFTGrid(int nfftx, int nffty, int nfftz);
};
