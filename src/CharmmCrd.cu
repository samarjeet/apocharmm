// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "CharmmCrd.h"

CharmmCrd::CharmmCrd(const std::string &fileName) {
  readCharmmCrdFile(fileName);
}

CharmmCrd::CharmmCrd(const std::vector<float3> _inpCoords) {
  numAtoms = _inpCoords.size();

  int count = 0;
  while (count < numAtoms) {
    coords.push_back(make_float4(_inpCoords[count].x, _inpCoords[count].y,
                                 _inpCoords[count].z, 0.0));
    count++;
  }
}

CharmmCrd::CharmmCrd(const std::vector<std::vector<double>> _inpCoords) {
  // convert to float3 then use other constructor
  numAtoms = _inpCoords.size();
  for (int i = 0; i < numAtoms; i++) {
    coords.push_back(
        make_float4(_inpCoords[i][0], _inpCoords[i][1], _inpCoords[i][2], 0.0));
  }
}

// Split a single crd line into a vector containing
// atomId, resId, resName, atom, x, y ,z
//static std::vector<std::string> split(std::string line) {
//  std::stringstream ss(line);
//  std::string atomId, resId, resName, atom, x, y, z;
//  ss >> atomId >> resId >> resName >> atom >> x >> y >> z;
//  std::vector<std::string> content = {atomId, resId, resName, atom, x, y, z};
//
//  return content;
//}

void CharmmCrd::readCharmmCrdFile(std::string fileName) {

  // Assume that the first few continuous lines are comments
  // followed by a line with N (number of atoms)
  // next N lines are coordinates
  // if any of the lines is blank, exit with error

  std::string line;
  std::ifstream crdFile(fileName);

  if (!crdFile.is_open()) {
    throw std::invalid_argument("ERROR: Cannot open the file " + fileName +
                                "\nExiting\n");
    exit(0);
  }

  //  comment lines
  while (1) {
    std::getline(crdFile, line);
    if (line[0] != '*')
      break;
  }
  numAtoms = std::stoul(line);

  int count = 1;
  std::getline(crdFile, line);

  while (count <= numAtoms) {
    if (line.size() == 0) {
      throw std::invalid_argument("ERROR: Blank line read in " + fileName +
                                  "\n. Exiting\n");
      exit(0);
    }
    // auto content = split(line);
    int atomId, resId, resIdInSeg;
    std::string resName, atomName, segName;
    float x, y, z, bFactor;

    std::stringstream ss(line);
    ss >> atomId >> resId >> resName >> atomName >> x >> y >> z >> segName >>
        resIdInSeg >> bFactor;
    coords.push_back(make_float4(x, y, z, 0.0));
    // coords.push_back(make_float4(std::stod(content[4]),
    // std::stod(content[5]),std::stod(content[6])));

    // Next line
    std::getline(crdFile, line);
    count++;
  }
}
