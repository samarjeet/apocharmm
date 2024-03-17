// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "CudaPMEReciprocalGeneralForce.h"

CudaPMEReciprocalGeneralForce::CudaPMEReciprocalGeneralForce() {}

void CudaPMEReciprocalGeneralForce::setFFTGrid(int _nfftx, int _nffty,
                                               int _nfftz) {

  nfftx = _nfftx;
  nffty = _nffty;
  nfftz = _nfftz;
}
