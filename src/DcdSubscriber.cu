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
  int firstStep = 0;
  int interval = 1000; // TODO : get it from integrator
  float timeStep = 0.001;
  int boxFlag = 0;

  DcdHeader header;
  header.size1 = 84;
  header.cord[0] = 'C';
  header.cord[1] = 'O';
  header.cord[2] = 'R';
  header.cord[3] = 'D';

  header.ints1[0] = 0;
  header.ints1[1] = firstStep;
  header.ints1[2] = interval;
  header.ints1[3] = 0;
  header.ints1[4] = 0;
  header.ints1[5] = 0;
  header.ints1[6] = 0;
  header.ints1[7] = 0;
  header.ints1[8] = 0;

  header.timeStep = timeStep;

  header.ints2[0] = boxFlag;
  header.ints2[1] = 0;
  header.ints2[2] = 0;
  header.ints2[3] = 0;
  header.ints2[4] = 0;
  header.ints2[5] = 0;
  header.ints2[6] = 0;
  header.ints2[7] = 0;
  header.ints2[8] = 0;
  header.ints2[9] = 24;
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

  int p = numFramesWritten * 1000;
  fout.seekp(20);
  fout.write((char *)&p, sizeof(int));

  fout.seekp(0, std::ofstream::end);
}
