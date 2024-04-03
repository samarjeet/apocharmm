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
#include "DcdSubscriber.h"
#include <cstdio>
#include <iostream>

// DcdSubscriber::DcdSubscriber(const std::string &fileName)
//     : Subscriber(fileName) {
//   fout.open(fileName, std::ios::out | std::ios::binary);
//   numFramesWritten = 0;
//   numAtoms = charmmContext->getNumAtoms();
//   initialize();
// }

DcdSubscriber::DcdSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  // fout.open(fileName, std::ios::out | std::ios::binary);
  numFramesWritten = 0;
  isInitialized = false;
}

DcdSubscriber::DcdSubscriber(const std::string &fileName, int reportFreq)
    : Subscriber(fileName, reportFreq) {
  // fout.open(fileName, std::ios::out | std::ios::binary);
  numFramesWritten = 0;
  isInitialized = false;
}

DcdSubscriber::DcdSubscriber(const std::string &fileName, int reportFreq,
                             std::shared_ptr<CharmmContext> ctx)
    : Subscriber(fileName, reportFreq) {

  // fout.open(fileName, std::ofstream::out | std::ofstream::binary);
  // fout.open(fileName, std::ios::out | std::ios::binary);
  numFramesWritten = 0;
  numAtoms = ctx->getNumAtoms();
  initialize();
}

struct DcdHeader {
  int size1;
  char cord[4];
  int ints1[9];
  float timeStep;
  int ints2[13];
  char str1[80];
  char str2[80];
  int last[4];
};

void DcdSubscriber::initialize() {
  if (!hasCharmmContext) {
    throw std::invalid_argument("DcdSubscriber does not have a CharmmContext, "
                                "can't initialize() properly.\n");
  }
  numAtoms = charmmContext->getNumAtoms();
  // Writing the headers
  int firstStep = 0; // TODO : get the firstStep from integrator
  // int interval = reportFreq;
  float timeStep = integrator->getTimeStep();
  int ndegf = charmmContext->getDegreesOfFreedom();
  // int boxFlag = 0;
  int boxFlag = 1; // adding unit cell dimensions

  DcdHeader header;
  header.size1 = 84;
  header.cord[0] = 'C';
  header.cord[1] = 'O';
  header.cord[2] = 'R';
  header.cord[3] = 'D';

  header.ints1[0] = 0; // number of frames written
  header.ints1[1] = firstStep;
  header.ints1[2] = reportFreq;
  header.ints1[3] = 0;     // reportFreq * number of frame written
  header.ints1[4] = 0;     // velocity saving frequency
  header.ints1[5] = 0;     // unused
  header.ints1[6] = 0;     // unused
  header.ints1[7] = ndegf; // ndegf
  header.ints1[8] = 0;     // Number of fixed atoms

  header.timeStep = timeStep;

  header.ints2[0] = boxFlag;
  header.ints2[1] = 0;  // 4d data
  header.ints2[2] = 0;  // cheq charge data
  header.ints2[3] = 0;  // non-contiguous data
  header.ints2[4] = 0;  // unused
  header.ints2[5] = 0;  // unused
  header.ints2[6] = 0;  // unused
  header.ints2[7] = 0;  // unused
  header.ints2[8] = 0;  // unused
  header.ints2[9] = 35; // CHARMM version, should be >= 22
  header.ints2[10] = 84;

  header.ints2[11] = 164;
  header.ints2[12] = 2;
  // header.str1 = "Created by CHARMM";
  // header.str2 = "Created at time ";
  header.last[0] = 164;

  header.last[1] = 4;
  header.last[2] = numAtoms;
  header.last[3] = 4;

  fout.write((char *)&header, sizeof(header));
  isInitialized = true;
}

DcdSubscriber::~DcdSubscriber() { fout.close(); }

void DcdSubscriber::update() {
  // Initializing if not done yet -- to remove need for initialization with
  // CharmmContext
  if (!isInitialized) {
    initialize();
  }
  // std::cout << "In DCD update\n";

  auto boxDimensions = charmmContext->getBoxDimensions();

  // write 6 double
  float zero;
  int boxSize = 6 * sizeof(double);

  fout.write((char *)&boxSize, sizeof(int));

  fout.write((char *)&boxDimensions[0], sizeof(double));
  fout.write((char *)&zero, sizeof(double));
  fout.write((char *)&boxDimensions[1], sizeof(double));
  fout.write((char *)&zero, sizeof(double));
  fout.write((char *)&zero, sizeof(double));
  fout.write((char *)&boxDimensions[2], sizeof(double));

  fout.write((char *)&boxSize, sizeof(int));

  auto xyzq = *(this->charmmContext->getXYZQ()->getHostXYZQ());

  int length = numAtoms * 4;

  fout.write((char *)&length, sizeof(int));
  for (int i = 0; i < numAtoms; ++i) {
    fout.write((char *)&xyzq[i].x, sizeof(float));
  }
  fout.write((char *)&length, sizeof(int));

  fout.write((char *)&length, sizeof(int));
  for (int i = 0; i < numAtoms; ++i) {
    fout.write((char *)&xyzq[i].y, sizeof(float));
  }
  fout.write((char *)&length, sizeof(int));

  fout.write((char *)&length, sizeof(int));
  for (int i = 0; i < numAtoms; ++i) {
    fout.write((char *)&xyzq[i].z, sizeof(float));
  }
  fout.write((char *)&length, sizeof(int));

  ++numFramesWritten;

  fout.seekp(8);
  fout.write((char *)&numFramesWritten, sizeof(int));

  int p = numFramesWritten * reportFreq;
  fout.seekp(20);
  fout.write((char *)&p, sizeof(int));

  fout.seekp(0, std::ofstream::end);
}
