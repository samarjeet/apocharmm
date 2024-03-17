// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "ForceManager.h"
#include "MBARForceManager.h"
#include "MBARSubscriber.h"
#include "XYZQ.h"
#include <iomanip>
#include <iostream>

MBARSubscriber::MBARSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  numFramesWritten = 0;
}

MBARSubscriber::MBARSubscriber(const std::string &fileName, int reportFreq)
    : Subscriber(fileName, reportFreq) {
  numFramesWritten = 0;
}

void MBARSubscriber::update() {
  auto fm = charmmContext->getForceManager();
  // auto tempfm = std::dynamic_pointer_cast<ForceManagerComposite>(
  auto tempfm = std::dynamic_pointer_cast<MBARForceManager>(
      fm); // dynamic cast to FMComposite in order to use its calc_force

  XYZQ *xyzq = charmmContext->getXYZQ();
  xyzq->transferFromDevice();
  float4 *xyzqPointer = xyzq->xyzq;
  tempfm->ForceManagerComposite::calc_force(xyzqPointer, false, true, false);
  // fm->ForceManagerComposite::calc_force(xyzqPointer, false, true, false);
  auto peCC = tempfm->getPotentialEnergy();
  peCC.transferFromDevice();

  for (int i = 0; i < peCC.getHostArray().size(); i++) {
    fout << std::setw(12) << peCC[i] << "\t";
  }
  fout << std::endl;
}

void MBARSubscriber::addForceManager(std::shared_ptr<ForceManager> fmIn) {
  forceManagerList.push_back(fmIn);
}

void MBARSubscriber::addForceManager(
    std::vector<std::shared_ptr<ForceManager>> fmlist) {
  for (int i = 0; i < fmlist.size(); i++) {
    forceManagerList.push_back(fmlist[i]);
  }
}

void MBARSubscriber::addForceManager(
    std::shared_ptr<ForceManagerComposite> fmcomposite) {
  auto children = fmcomposite->getChildren();
  for (int i = 0; i < children.size(); i++) {
    forceManagerList.push_back(children[i]);
  }
}

std::vector<std::shared_ptr<ForceManager>>
MBARSubscriber::getForceManagerList() {
  return forceManagerList;
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
  forceManagerList.insert(forceManagerList.begin(), ctx->getForceManager());
}
