// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CharmmCrd.h"

#include "str_utils.h"

CharmmCrd::CharmmCrd(const std::string &fileName) : Coordinates() {
  this->readCharmmCrdFile(fileName);
}

void CharmmCrd::readCharmmCrdFile(const std::string &fileName) {
  std::string fileData = "";
  apo::read_file_into_string(fileData, fileName);

  std::size_t pos = 0;
  std::string line = "";
  std::vector<std::string> tokens;

  // Parse TITLE
  do {
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
  } while (line[0] == '*');

  // Read coordinate data
  const unsigned long long int natom = std::stoull(tokens[0]);
  const bool isExt = ((tokens.size() >= 2) && (tokens[1] == "EXT"));
  this->setNumAtoms(static_cast<int>(natom));
  for (unsigned long long int i = 0; i < natom; i++) {
    line.clear();
    apo::get_line(line, pos, fileData);
    double x = -9999.9999, y = -9999.9999, z = -9999.9999;
    if (isExt) {
      x = std::stod(line.substr(40, 20));
      y = std::stod(line.substr(60, 20));
      z = std::stod(line.substr(80, 20));
    } else {
      x = std::stod(line.substr(20, 10));
      y = std::stod(line.substr(30, 10));
      z = std::stod(line.substr(40, 10));
    }
    m_CoordinatesD[i] = make_double4(x, y, z, 0.0);
    m_CoordinatesF[i] =
        make_float4(static_cast<float>(x), static_cast<float>(y),
                    static_cast<float>(z), 0.0f);
  }

  return;
}
