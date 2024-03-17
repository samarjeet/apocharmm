// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "cpp_utils.h"
#include <iostream>
#include <source_location>
#include <string>

void location(std::string_view message, const std::source_location location =
                                            std::source_location::current()) {
  std::clog << "File: " << location.file_name() << " Line: " << location.line()
            << " Function: " << location.function_name()
            << " Message: " << message << "\n";
}
