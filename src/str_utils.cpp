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
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

void apo::ltrim_ip(std::string &str) {
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char c) { return !std::isspace(c); }));
  return;
}

void apo::rtrim_ip(std::string &str) {
  str.erase(std::find_if(str.rbegin(), str.rend(),
                         [](unsigned char c) { return !std::isspace(c); })
                .base(),
            str.end());
  return;
}

void apo::trim_ip(std::string &str) {
  apo::rtrim_ip(str);
  apo::ltrim_ip(str);
  return;
}

std::string apo::ltrim(const std::string &str) {
  std::string s = str;
  apo::ltrim_ip(s);
  return s;
}

std::string apo::rtrim(const std::string &str) {
  std::string s = str;
  apo::rtrim_ip(s);
  return s;
}

std::string apo::trim(const std::string &str) {
  std::string s = str;
  apo::trim_ip(s);
  return s;
}

void apo::to_lower_ip(std::string &str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return;
}

void apo::to_upper_ip(std::string &str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return;
}

std::string apo::to_lower(const std::string &str) {
  std::string s = str;
  apo::to_lower_ip(s);
  return s;
}

std::string apo::to_upper(const std::string &str) {
  std::string s = str;
  apo::to_upper_ip(s);
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
    apo::ltrim_ip(s);
  }
  tokens.push_back(s);

  return tokens;
}

void apo::get_line(std::string &line, std::size_t &pos,
                   const std::string &file_data) {
  const std::size_t pos1 = file_data.find_first_of('\n', pos);
  line = file_data.substr(pos, pos1 - pos);
  pos = pos1 + 1;
  return;
}

void apo::read_file_into_string(std::string &file_data,
                                const std::string &file_name) {
  std::ifstream ifs(file_name, std::ios::in | std::ios::binary | std::ios::ate);
  if (ifs.is_open() == false)
    throw std::runtime_error("Failed to open file \"" + file_name + "\"");

  // Store the size of the file
  const std::size_t fsize = ifs.tellg();

  // Initialize a std::string with length fileSize, filled with null characters
  file_data = std::string(fsize, '\0');

  // Go back to the beginning of the file and store the contents in the string
  ifs.seekg(0, std::ios::beg);
  ifs.read(&file_data[0], fsize);
  ifs.close();

  return;
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
