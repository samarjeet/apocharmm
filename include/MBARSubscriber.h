// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include "ForceManager.h"
#include "Subscriber.h"

/** @brief Subscriber linked to one or more ForceManager. Computes at each
 * update step the energies according to each ForceManager, and outputs them in
 * a single file in different columns (as a typical MBAR input).
 *
 * The various ForceManager for which only energies should be computed can be
 * linked via a ForceManagerComposite object, or a list of ForceManager, or
 * sequentially adding single ForceManager one by one.
 *
 * @attention Currently, it is not possible to unsub, create a new MBARSub in
 * the same variable, then resub that new MBARSub.
 *
 * @note The ForceManager in charge of the dynamics
 * (this.charmmContext.getForceManager()) is included by default in the
 * forceManagerList (all FM for which energies should be computed) upon
 * subscription.
 */
class MBARSubscriber : public Subscriber {
public:
  /** @brief Basic constructor. Specify output file name and report frequency.
   * Will initialize its forceManagerList with
   * this.charmmContext.getForceManager.
   * @todo Do we need a header ? What info could we add in there ?
   */
  MBARSubscriber(const std::string &fileName, int reportFreqIn);
  /** @brief Basic constructor. Specify output file name. Default output
   * interval of 1000 steps will be used.
   * Will initialize its forceManagerList with
   * this.charmmContext.getForceManager.
   */
  MBARSubscriber(const std::string &fileName);

  /** @brief Report to the output */
  void update() override;

  /** @brief Add a ForceManager to the list. When updating, energy will be
   * computed using this ForceManager as well as all other previously added.
   */
  void addForceManager(std::shared_ptr<ForceManager> fmIn);

  /** @brief Add a list of ForceManager obejcts */
  void addForceManager(std::vector<std::shared_ptr<ForceManager>> fmlist);

  /** @brief Add all child ForceManager from a CompositeForceManager */
  void addForceManager(std::shared_ptr<ForceManagerComposite> fmcomposite);

  /** @brief Returns current list of ForceManager attached to MBARSubscriber */
  std::vector<std::shared_ptr<ForceManager>> getForceManagerList();

  /** @brief Sets CharmmContext (usually called by Integrator.subscribe), adds
   * its ForceManager to the forceManagerList.
   *
   * Add functionality to the basic Subscriber::setCharmmContext function in
   * order to put, in first position of the forceManagerList, the ForceManager
   * used to "drive" the simulation
   */
  void setCharmmContext(std::shared_ptr<CharmmContext> ctx);

private:
  /** @brief Vector of all ForceManager objects to be used to compute energy at
   * each update
   */
  std::vector<std::shared_ptr<ForceManager>> forceManagerList{};
  int numFramesWritten = 0;
};