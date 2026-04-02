// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "str_utils.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

void apo::ltrimIP(std::string &str) {
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char c) { return !std::isspace(c); }));
  return;
}

void apo::rtrimIP(std::string &str) {
  str.erase(std::find_if(str.rbegin(), str.rend(),
                         [](unsigned char c) { return !std::isspace(c); })
                .base(),
            str.end());
  return;
}

void apo::trimIP(std::string &str) {
  apo::rtrimIP(str);
  apo::ltrimIP(str);
  return;
}

std::string apo::ltrim(const std::string &str) {
  std::string s = str;
  apo::ltrimIP(s);
  return s;
}

std::string apo::rtrim(const std::string &str) {
  std::string s = str;
  apo::rtrimIP(s);
  return s;
}

std::string apo::trim(const std::string &str) {
  std::string s = str;
  apo::trimIP(s);
  return s;
}

void apo::toLowerIP(std::string &str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return;
}

void apo::toUpperIP(std::string &str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return;
}

std::string apo::toLower(const std::string &str) {
  std::string s = str;
  apo::toLowerIP(s);
  return s;
}

std::string apo::toUpper(const std::string &str) {
  std::string s = str;
  apo::toUpperIP(s);
  return s;
}

std::vector<std::string> apo::split(const std::string &str,
                                    const std::string &delimiter) {
  std::string s = apo::trim(str);
  std::vector<std::string> tokens;
  std::size_t pos = 0;

  while ((pos = s.find(delimiter)) != std::string::npos) {
    tokens.push_back(s.substr(0, pos));
    s.erase(0, pos + delimiter.length());
    apo::ltrimIP(s);
  }
  tokens.push_back(s);

  return tokens;
}

std::string apo::cDoubleToFortSciStr(const double val, const int prec) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(prec) << val;
  std::string str = oss.str();
  std::replace(str.begin(), str.end(), 'e', 'D');
  return str;
}

double apo::fortSciStrToCDouble(const std::string &str) {
  std::string s = str;
  std::replace(s.begin(), s.end(), 'D', 'e');
  return std::stod(s);
}
