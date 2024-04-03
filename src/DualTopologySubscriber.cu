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
#include "DualTopologySubscriber.h"
#include <fstream>
#include <iostream>

DualTopologySubscriber::DualTopologySubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  numFramesWritten = 0;
}
DualTopologySubscriber::DualTopologySubscriber(
    const std::string &fileName,
    int reportFreq) //, std::shared_ptr<CharmmContext> ctx)
    : Subscriber(fileName, reportFreq) {
  numFramesWritten = 0;
}

DualTopologySubscriber::~DualTopologySubscriber() {
  std::cout << "Trying to close the dual topology subscriber\n";
  fout.close();
}

void DualTopologySubscriber::update() {

  // std::cout << pe << "\t" << ke << "\n";
  float pe = this->charmmContext->calculateForces(false, true, true);

  // the previous command has calculated the energies as well
  // here we are just calling them. No force or energy is being calculated here
  auto pes = this->charmmContext->getPotentialEnergies();

  // fout << pes[0] << "\t" << pes[1] << "\t" << pe << std::endl;
  for (auto pesi : pes) {
    fout << pesi << "\t";
  }
  fout << pe << std::endl;
  // std::cout << pes[0] << "\t" << pes[1] << "\t" << pes[0] + pes[1] <<
  // std::endl;

  ++numFramesWritten;
}
