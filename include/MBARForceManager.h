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

/* Question : how do I run standard FEP(via MBAR) computation ? One way would be
 * to save n_lambda traj and post-process. Another is to run n_lambda dynamics,
 * and for each compute n_lambda energies at each MBAR report step. That seems
 * like the most "FMComposite" way. */

/** @brief ForceManagerComposite subclass to compute FEP free energies. Each FM
 * child represents one alchemical window. Forces are computed for one specific
 * child (a given lambda_elec/vdw value), and energies are computed for all
 * children at each MBAR report step.
 */
class MBARForceManager : public ForceManagerComposite {
public:
  MBARForceManager();
  MBARForceManager(std::vector<std::shared_ptr<ForceManager>> fmList);
  void initialize() override;
  void setSelectorVec(std::vector<float> lambdaIn) override;
  float calc_force(const float4 *xyzq, bool reset = false,
                   bool calcEnergy = false, bool calcVirial = false) override;
  std::shared_ptr<Force<double>> getForces() override;
  CudaContainer<double> getVirial() override;
  // void storePotentialEnergy();
protected:
  int storePECounter, storePotentialEnergyFrequency;
  /** @brief Index of the child driving the simulation. Initially, set to -1
   * to raise error if no selector vector has been set. */
  int nonZeroLambdaIndex;
  /** @brief Number of alchemical windows */
  int nAlchemicalWindows;
  /** @brief lambda_elec values */
};
