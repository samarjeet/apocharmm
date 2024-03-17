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
#include "ForceManager.h"

class NonEquilibriumForceManager {
public:
   NonEquilibriumForceManager();
};

//class NonEquilibriumForceManager : 
//   public ForceManagerComposite {
//public:
//  NonEquilibriumForceManager();
//  void setLambdaIncrements(float lambdaIncrement);
//  virtual float calc_force(const float4 *xyzq, bool reset = false,
//                           bool calcEnergy = false,
//                           bool calcVirial = false) override;
//
//private:
//  float lambdaIncrement;
//};
