// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, Félix Aviat
//
// ENDLICENSE

#pragma once
#include "ForceManager.h"

/**
 * @brief Idea: tool to generate modified ForceManager objects
 */
class ForceManagerGenerator {
public:
  /**
   * @brief Base constructor
   */
  ForceManagerGenerator();
  /**
   * @brief Constructor setting a basic ForceManager, from which generated
   * ones can be created
   */
  ForceManagerGenerator(std::shared_ptr<ForceManager> baseForceManager);
  /**
   * @brief Generate a SINGLE ForceManager following arguments
   */
  virtual std::shared_ptr<ForceManager> generateForceManager();

protected:
  /**
   * @brief Starting ForceManager object to be copied and modified
   */
  std::shared_ptr<ForceManager> baseForceManager;
};

/**
 * @brief Generates ForceManager objects that are alchemically modified
 *
 * For now :
 * - only lambda_electrostatic !
 * - only one alchemical region
 */
class AlchemicalForceManagerGenerator : public ForceManagerGenerator {
public:
  /**
   * @brief Base constructor.
   */
  AlchemicalForceManagerGenerator();
  /**
   * @brief Generator constructor with a base ForceManager
   *
   * @param baseForceManager ForceManager used as a base to generate new
   * ForceManager objects
   */
  AlchemicalForceManagerGenerator(
      std::shared_ptr<ForceManager> baseForceManager);
  /**
   * @brief Defines the atoms concerned by alchemical transformation
   * @param[in] alchRegionIn Vector of integers, each corresponding to the
   * index of an atom
   *
   * Index starts at 0.
   */
  void setAlchemicalRegion(std::vector<int> alchRegionIn);

  /** @brief Returns the alchemical region (vector of indices) */
  std::vector<int> getAlchemicalRegion();

  /**
   * @brief Generate a SINGLE ForceManager following arguments. Specific to
   * alchemical.
   *
   * returns an **uninitialized** ForceManager object.
   *
   * Staging:
   * 1. Copy baseFM to newFM
   * 2. Modify parameters of newFM
   */
  std::shared_ptr<ForceManager> generateForceManager(double lambdaElecIn,
                                                     double lambdaVdWIn);

  /**
   * @brief Set base ForceManager
   */
  void setBaseForceManager(std::shared_ptr<ForceManager> forceManagerIn);

  /**
   * @brief Modify ForceManager object to scale electrostatics
   *
   * @param[in] lambdaIn: factor to scale electrostatic interactions
   * @param[in] fmIn ForceManager to be modified
   */
  void modifyElectrostatics(std::shared_ptr<ForceManager> fmIn,
                            double lambdaIn);

  /**
   * @brief Modify van der Waals interactions
   *
   * @todo Not implemented yet
   */
  void modifyvdW(std::shared_ptr<ForceManager> fmIn, float lambdaIn);

private:
  /**
   * @brief Alchemical schedule (list of lambda values)
   */
  std::vector<int> lambdaElecSchedule;

  /**
   * @brief Selection of atoms concerned by the alchemical transformation
   *
   * A vector of int, each element being an atom index belonging to the
   * alchemical region
   */
  std::vector<int> alchemicalRegion;
};
