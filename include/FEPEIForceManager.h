// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, Felix Aviat
//
// ENDLICENSE

#pragma once
#include "ForceManager.h"

/** @brief ForceManagerComposite subclass to compute FEP free energies using
 * Energy Interpolation.
 *
 * Dynamics are controlled by $(1-\lambda) E_0 + \lambda E_1$.
 * Will somehow provide a way for a subscriber to ouput all the energies
 */
class FEPEIForceManager : public ForceManagerComposite {
public:
  FEPEIForceManager();
  void initialize() override;
  void setSelectorVec(std::vector<float> lambdaIn) override;
  float calc_force(const float4 *xyzq, bool reset = false,
                   bool calcEnergy = false, bool calcVirial = false) override;
  std::shared_ptr<Force<double>> getForces() override;
  // void storePotentialEnergy();

  /** @brief Set Alchemical \f$\lambda\f$ values for FEP */
  void setLambdas(std::vector<float> lambdasIn);

  /** @brief The lambda for the current alchemical window. **/
  void setLambda(float _lambda) { lambda = _lambda; }

  /**
   * @brief Get the Lambda Potential Energies object
   *
   * @return CudaContainer<double>
   */
  CudaContainer<double> getLambdaPotentialEnergies();
  CudaContainer<double> getVirial() override;

protected:
  /**
   * @brief all the alchemical lambdas
   */
  CudaContainer<float> lambdas;

  /**
   * @brief the current lambda being used for energy interpolation
   *
   */
  float lambda;

  void weighForces();
  int storePECounter, storePotentialEnergyFrequency;
  int nonZeroLambdaIndex;
  /** @brief Number of alchemical windows */
  int nAlchemicalWindows;
  /** @brief lambda_elec values */

  CudaContainer<double> lambdaPotentialEnergies;
};
