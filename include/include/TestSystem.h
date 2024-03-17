// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include "CharmmContext.h"
#include <memory>
#include <string>

class TestSystem {
private:
  std::string name;

public:
  TestSystem(std::string name_) : name(name_) {}
  ~TestSystem();
  virtual std::unique_ptr<CharmmContext> getCharmmContext();
};

class HarmonicOscillator : public TestSystem {
private:
public:
  // HarmonicOscillator(std::string name_="Harmonic Oscialltor") :
  // TestSystem(name_) {}
  HarmonicOscillator(std::string name_) : TestSystem(name_) {}
  std::unique_ptr<CharmmContext> getCharmmContext() override;
};
