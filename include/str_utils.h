// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include <string>
#include <vector>

namespace apo {

void ltrimIP(std::string &str);
void rtrimIP(std::string &str);
void trimIP(std::string &str);

std::string ltrim(const std::string &str);
std::string rtrim(const std::string &str);
std::string trim(const std::string &str);

void toLowerIP(std::string &str);
void toUpperIP(std::string &str);

std::string toLower(const std::string &str);
std::string toUpper(const std::string &str);

std::vector<std::string> split(const std::string &str,
                               const std::string &delimiter = " ");

} // namespace apo
