// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "MBARSubscriber.h"

#include "CharmmContext.h"
#include "MBARForceManager.h"
#include "XYZQ.h"
#include <iomanip>
#include <iostream>

MBARSubscriber::MBARSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  m_NumFramesWritten = 0;
}

MBARSubscriber::MBARSubscriber(const std::string &fileName, int reportFrequency)
    : Subscriber(fileName, reportFrequency) {
  m_NumFramesWritten = 0;
}

void MBARSubscriber::update(void) {
  auto fm = m_CharmmContext->getForceManager();
  // auto tempfm = std::dynamic_pointer_cast<ForceManagerComposite>(
  auto tempfm = std::dynamic_pointer_cast<MBARForceManager>(
      fm); // dynamic cast to FMComposite in order to use its calc_force

  XYZQ *xyzq = m_CharmmContext->getXYZQ();
  xyzq->transferFromDevice();
  float4 *xyzqPointer = xyzq->xyzq;
  tempfm->ForceManagerComposite::calcForce(xyzqPointer, false, true, false);
  // fm->ForceManagerComposite::calc_force(xyzqPointer, false, true, false);
  auto peCC = tempfm->getPotentialEnergy();
  peCC.transferFromDevice();

  for (std::size_t i = 0; i < peCC.getHostArray().size(); i++)
    m_FileStream << std::setw(12) << peCC[i] << "\t";
  m_FileStream << std::endl;

  return;
}

void MBARSubscriber::addForceManager(
    std::shared_ptr<ForceManager> forceManager) {
  m_ForceManagers.push_back(forceManager);
  return;
}

void MBARSubscriber::addForceManager(
    std::vector<std::shared_ptr<ForceManager>> forceManagers) {
  for (std::size_t i = 0; i < forceManagers.size(); i++)
    this->addForceManager(forceManagers[i]);
  return;
}

void MBARSubscriber::addForceManager(
    std::shared_ptr<ForceManagerComposite> forceManagerComposite) {
  this->addForceManager(forceManagerComposite->getChildren());
  return;
}

std::vector<std::shared_ptr<ForceManager>>
MBARSubscriber::getForceManagerList(void) {
  return m_ForceManagers;
}

// This function is called upon subscription to an Integrator.
// Its parent version just adds a charmmcontext.
// However, here, we also want to add the charmmContext.getForceManager() to our
// FMlist, in first position.
void MBARSubscriber::setCharmmContext(std::shared_ptr<CharmmContext> ctx) {
  Subscriber::setCharmmContext(ctx);
  std::cout << "The modified version of setCharmmContext runs\n";
  // add ctx's FM to forceManagerList
  // forceManagerList.push_front(ctx->getForceManager());
  m_ForceManagers.insert(m_ForceManagers.begin(), ctx->getForceManager());
  return;
}
