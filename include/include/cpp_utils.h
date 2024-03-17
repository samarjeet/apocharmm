// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include <source_location>
#include <string>
#include <vector>

// const std::string WHITESPACE = " \n\r\t\f\v";

std::string ltrim(const std::string &s);

std::string rtrim(const std::string &s);

std::string trim(const std::string &s);

std::string removeComments(std::string line);

std::vector<std::string> split(const std::string &str);

void location(const std::string_view str);

std::string toLower(const std::string &inpstr);