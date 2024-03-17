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

#include "PertForceManager.h"
#include <iostream>

PertForceManager::PertForceManager() {

  std::cout << "Pert Force Manager created\n";
}

//PertForceManager::setLambdaElecSchedule(std::vector<float> lambdasElecIn) { 
//   /* Check that:
//      - initial value is 1.0, 
//      - last value is 0.0, 
//      - values ordered in decreasing order.
//    */
//   nLambdasElec = lambdasElecIn.size()
//   if (lambdasElecIn[0] != 1.0) {
//      throw std::invalid_argument("Initial alchemical window lambda should be 1.0");
//   }
//   if (lambdasElecin[n-1] != 0.0) {
//      throw std::invalid_argument("Final alchemical window lambda should be 0.0");
//   }
//
//   throw std::invalid_argument("Not implemented yet !");
//}
//
//PertForceManager::addForceManager(std::shared_ptr<ForceManager> fmIn, float lambdaElecIn) {
//   this.addForceManager(std::shared_ptr<ForceManager> fmIn);
//   lambdaElectrostatics.push_back(lambdaElecIn);
//}
//
//
///*
//   Loop over all steps of the alchemical schedule. For each step, prepare a
//   ForceManager object. Add it as a child.
// */
//PertForceManager::prepareAllForceManagers(){
//   std::shared_ptr<ForceManager> tmpfm;
//   for (int i = 1 ; i < nLambdasElec ; i++) {
//      tmpfm = generateAlchemicallyModifiedForceManager();
//      this->addForceManager(tmpfm);
//   }
//}

