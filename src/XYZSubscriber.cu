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
#include "XYZSubscriber.h"
#include <iostream>

XYZSubscriber::XYZSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  numFramesWritten = 0;
}
XYZSubscriber::XYZSubscriber(const std::string &fileName, int reportFreq)
    : Subscriber(fileName, reportFreq) {
  numFramesWritten = 0;
}

XYZSubscriber::~XYZSubscriber() { fout.close(); }

void XYZSubscriber::initialize() {
  if (!hasCharmmContext) {
    throw std::invalid_argument(
        "XYZSubscriber: Can't initialize without a CharmmContext.\n");
  }
}

void XYZSubscriber::update() {
  if (!isInitialized) {
    initialize();
  }
  int numAtoms = charmmContext->getNumAtoms();

  // vector in the shared ptr
  auto xyzq = *(this->charmmContext->getXYZQ()->getHostXYZQ());

  // std::cout << xyzq[0].x << "\n";
  for (int i = 0; i < numAtoms; ++i) {
    // if (xyzq[i].x < -10.0)
    // std::cout << i << "\t" << xyzq[i].x << "\t" << xyzq[i].y << "\t"
    //           << xyzq[i].z << "\n";
    fout << i << "\t" << xyzq[i].x << "\t" << xyzq[i].y << "\t" << xyzq[i].z
         << std::endl;
  }

  // int status = nc_put_vara_float(ncid, coordVariableId, start, count, xyzNC);

  ++numFramesWritten;
}
