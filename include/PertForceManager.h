// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

/*
 * This header file contains the interface for Force management in CHARMM.
 * This class ensures that the Forces are destroyed when the ForceManager
 * is destroyed.
 *
 */

#pragma once
#include "ForceManager.h"

/**
 * @brief Composite force manager for Free Energy Perturbation computation
 *
 * Composite Force Manager designed to compute FEP free energies. 
 * Contains two lists of lambda values ranging from 1.0 to 0.0, corresponding
 * to the van der Waals and electrostatics deactivation.
 * ForceManager objects corresponding to the successive values should be added
 * as members via addForceManager.
 *
 * Child ForceManager can either be added along with their lambda corresponding
 * values, or independently, in which case the lambda schedule has to be set
 * elsewhere.
 *
 * TODO: add van der Waals (non bonded) part
 *
 */
class PertForceManager : public ForceManagerComposite {
public:
  PertForceManager();

//  /**
//   * @brief Set electrostatics schedule.
//   *
//   * Initial value should be 1.0, last value should be 0.0. 
//   */
//  void setLambdaElecSchedule(std::vector<float> lambdasElecIn);
//
//  std::vector<float> getLambdaElectrostatics();
//
//  /**
//   * @brief Add a ForceManager child with corresponding lambda value
//   *
//   * NOTE/TODO : will require vdW lambda in the future !
//   *
//   * @param fmIn ForceManager child
//   *
//   * @param lambdaElecIn Value of the scaling factor for the electrostatic
//   * interactions
//   */
//  void addForceManager(std::shared_ptr<ForceManager> fmIn, float lambdaElecIn);
//
//
//  /**
//   * @brief Generates all required ForceManager objects.
//   *
//   * Assuming that the non-alchemically modified ForceManager is already in
//   * "children", for each step (except the first) of the alchemical schedule,
//   * creates a ForceManager matching the lambda values, and adds it to the children .
//   */
//  void prepareAllForceManagers();
//
//   /* TODO
//    * - Check that there is as many lambda windows as there are children !
//    * - For now, only Electrostatic deactivation available. More
//    *   implementation/corrections expected when adding vdW
//    */

//protected:
//  /**
//   * @brief Schedule for lambda_electrostatics. 
//   *
//   * Float vector, should start with 1 and finish with 0.
//   */
//  std::vector<float> lambdaElecSchedule;
//
//  /**
//   * @brief Number of lambdas in the electrostatic schedule
//   */
//  int nLambdasElec;
};


