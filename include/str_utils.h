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

#include <cstddef>
#include <string>
#include <vector>

namespace apo {

void ltrim_ip(std::string &str);
void rtrim_ip(std::string &str);
void trim_ip(std::string &str);

std::string ltrim(const std::string &str);
std::string rtrim(const std::string &str);
std::string trim(const std::string &str);

void to_lower_ip(std::string &str);
void to_upper_ip(std::string &str);

std::string to_lower(const std::string &str);
std::string to_upper(const std::string &str);

std::vector<std::string> split(const std::string &str,
                               const std::string &delimiter = " ");

void get_line(std::string &line, std::size_t &pos,
              const std::string &file_data);

void read_file_into_string(std::string &file_data,
                           const std::string &file_name);

std::string cDoubleToFortSciStr(const double val, const int prec);

double fortSciStrToCDouble(const std::string &str);

} // namespace apo
