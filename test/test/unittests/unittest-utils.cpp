// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, Félix Aviat
//
// ENDLICENSE

// Test util functions
#include "cuda_runtime.h"

#include "catch.hpp"
#include "compare.h"
#include "cpp_utils.h"
#include <iostream>

TEST_CASE("stringUtils") {
  std::string toBeTrimmed = " hello ", referenceString = "hello",
              toBeTrimmedRight = "hello ", toBeTrimmedLeft = " hello";

  CHECK(ltrim(toBeTrimmedLeft) == referenceString);
  CHECK(rtrim(toBeTrimmedRight) == referenceString);
  CHECK(trim(toBeTrimmed) == referenceString);
}

TEST_CASE("vectorFlattener") {
  std::vector<std::vector<int>> inputVector = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  std::vector<int3> expectedVector = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  std::vector<int3> outputVector = flattenVector<int, int3>(inputVector);
  CHECK(CompareVectors(outputVector, expectedVector));
}