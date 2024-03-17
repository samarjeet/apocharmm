// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#include "TestForce.h"
#include <iostream>

void TestForce::calc_force() {

  std::cout << "Calculating test force \n";
  std::cout << "Value stored in object is : " << object->getNum() << "\n";
  std::cout << "Value stored is : " << num << "\n";
}
