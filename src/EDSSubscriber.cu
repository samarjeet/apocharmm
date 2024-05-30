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
#include "EDSForceManager.h"
#include "EDSSubscriber.h"
#include <fstream>
#include <iostream>

EDSSubscriber::EDSSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  initialize();
  numFramesWritten = 0;
}
EDSSubscriber::EDSSubscriber(const std::string &fileName, int reportFreq)
    : Subscriber(fileName, reportFreq) {
  initialize();
  numFramesWritten = 0;
}

EDSSubscriber::~EDSSubscriber() { fout.close(); }

void EDSSubscriber::initialize() {}

void EDSSubscriber::update() {
  // We need to access the potential energy for all members of the CompositeFM
  auto fm = charmmContext->getForceManager();

  if (fm->isComposite()) {
    auto children = fm->getChildren();
    for (int i = 0; i < children.size(); i++) {
      auto childPotentialEnergy = children[i]->getPotentialEnergy();
      childPotentialEnergy.transferFromDevice();
      fout << childPotentialEnergy[0] << "\t";
    }
    auto pe = fm->getPotentialEnergy();
    pe.transferFromDevice();
    fout << pe[0] << "\t";
    fout << std::endl;

    ++numFramesWritten;
  } else {
    std::cout << "WARNING -- You should not be using a EDSSubscriber with a "
                 "non-composite ForceManager.\n";
  }
}

// void MBARSubscriber::printHeader() {
//   // Print header info to the file
//   // * lambda schedule / number of lambda points
//   // * energy offsets used
//   // * s value used for EDS
//   // *  ?
//   // This assumes we're using an EDSForceManager
//   std::shared_ptr<EDSForceManager> fm = charmmContext->getForceManager();
//   fout << "# S value: " << fm.getSValue() << " .\n" ;
//   fout << "# Energy offsets : " ;
//   auto eo = fm.getEnergyOffsets();
//   for (int i=0; i < eo.size(); i++) {
//     fout << eo[i] << " ";
//   }
//   fout << std::endl;
//
//
// }
