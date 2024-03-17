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
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

const std::string WHITESPACE = " \n\r\t\f\v";

std::string ltrim(const std::string &s) {
  size_t start = s.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string &s) {
  size_t end = s.find_last_not_of(WHITESPACE);
  return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string &s) { return rtrim(ltrim(s)); }

std::string removeComments(std::string line) {
  line = trim(line);
  auto npos = line.find_first_of('!');
  return trim(line.substr(0, npos));
}

std::vector<std::string> split(const std::string &str) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  tokenStream >> token;
  while (token.size() && tokenStream) {
    tokens.push_back(token);
    tokenStream >> token;
  }
  return tokens;
}

std::string toLower(const std::string &str) {
  std::string lower = str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  return lower;
}