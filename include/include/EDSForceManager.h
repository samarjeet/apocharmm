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

/**
 * @brief Composite force manager designed to handle Enveloping Distribution
 * Sampling simulations
 *
 * Computes forces using energy interpolation of two Hamiltonians, represented
 * by two different ForceManager objects.
 *
 * Uses a smoothness parameter \f$s\f$ and, for N different states, N energy
 * offsets \f$E_i\f$,  such that 
 * \f[     H = - \frac{1}{\beta s} \exp^{-\beta s \left( H -
 * E_i)\right) }  \f] See <a
 * href="https://aip.scitation.org/doi/10.1063/1.2730508">Enveloping
 * Distribution Sampling</a> (doi:10.1063/1.2730508)
 */
class EDSForceManager : public ForceManagerComposite {
public:
  /**
   * @brief Constructor using two ForceManager
   * @todo Should not be used.
   * @attention delete that
   */
  EDSForceManager(std::shared_ptr<ForceManager> fm1,
                  std::shared_ptr<ForceManager> fm2);

  /**
   * @brief Base constructor
   */
  EDSForceManager();
  /**
   * @brief Initializes each ForceManager child. Should be called *after* all
   * children ForceManager have been added, and *before* setting the energy
   * offsets.
   */
  void initialize() override;
  // void setLambdaVec(std::vector<float> lambdaIn) override;

  /**
   * @brief Compute forces following the Energy Interpolation (used in EDS)
   *
   * Computes energies and forces for all children.
   */
  float calc_force(const float4 *xyzq, bool reset = false,
                   bool calcEnergy = false, bool calcVirial = false) override;

  /**
   * @brief Returns getForces on the FIRST child
   */
  std::shared_ptr<Force<double>> getForces() override;

  /**
   * @brief ???
   *
   * @todo
   */
  void storePotentialEnergy();

  /**
   * @brief Set smoothing parameter value
   * @todo Should have a way to implement a \f$s(\lambda)\f$ rather than a
   * single s value
   *
   * See description of EDSForceManager
   */
  void setSValue(float sValue);

  /** @brief EDS S-value getter */
  float getSValue();

  /**
   * @brief Set potential energy offsets (one per Hamiltonian, aka per child).
   * Should be called *after* initialization, as its size is allocated based on
   * the number of FMchildren.
   */
  void setEnergyOffsets(std::vector<double> _energyOffsets);
  /** 
   * @brief Get potential energy offsets used for EDS
  */
  CudaContainer<double> getEnergyOffsets();

  /** @brief Set \f$\lambda\f$ value for \f$\lambda\f$-EDS */
  void setLambdaValue(float lambdaIn);

protected:
  int storePECounter, storePotentialEnergyFrequency;
  int nonZeroLambdaIndex;
  /** @brief \f$\lambda\f$ value for \f$\lambda\f$-EDS method */
  float lambdaValue = -1;

  /** @brief Smoothness parameter of EDS 
   *
   * Default value (set by constructor) 0.05
   */
  float sValue;

  // std::vector<double> energyOffsets;
  /** @brief Energy offsets values for EDS */
  CudaContainer<double> energyOffsets;

  /** @brief Force weighting factors */
  CudaContainer<double> weights;

  /**
   * @brief udpates the weights based on the potential energies of the children.
   * Should be called only after the potential energies of the children
   * have been calculated.
   */
  void updateWeights();

  /**
   * @brief use the updated weights to calculate the weighted combination of the
   * forces from the children
   */
  void weighForces();
};
