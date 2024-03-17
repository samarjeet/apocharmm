// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include "TestSystem.h"

TestSystem::~TestSystem() {}

std::unique_ptr<CharmmContext> TestSystem::getCharmmContext() {
  return nullptr;
}

std::unique_ptr<CharmmContext> HarmonicOscillator::getCharmmContext() {
  // auto ptr = std::make_unique<CharmmContext>(1);
  // Create force over here as well
  // and add it to the context
  // return ptr;
  return NULL;
}
