// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmResidueTopology.h"
#include "cpp_utils.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>

CharmmResidueTopology::CharmmResidueTopology() {}

CharmmResidueTopology::CharmmResidueTopology(std::string rtfFileName) {}

std::string getCleanLine(std::ifstream &rtfFile) {
  std::string line;
  std::getline(rtfFile, line);
  line = removeComments(line);
  line = trim(line);
  // line = std::toupper(line);
  std::transform(line.begin(), line.end(), line.begin(),
                 [](unsigned char c) { return std::toupper(c); });

  return line;
}
void CharmmResidueTopology::readRTF(std::string fileName) {

  enum FileType { RTF, TOPPAR };
  FileType fileType;
  int pos = fileName.find_last_of('/');
  if (pos != std::string::npos) {
    int topparPos = fileName.find("toppar", pos);
    if (topparPos != std::string::npos)
      fileType = TOPPAR;
    else
      fileType = RTF;
  }

  std::ifstream rtfFile(fileName);
  std::string line;

  if (!rtfFile.is_open()) {
    throw std::invalid_argument("ERROR: Cannot open the file " + fileName +
                                "\nExiting.\n");
    // exit(0);
    return;
  }

  if (fileType == RTF) {
    // std::cout << "Reading a  RTF file\n";
  }

  line = getCleanLine(rtfFile);

  int residuesInFile = 0;
  Residue residue;
  while (!rtfFile.eof()) {
    if (line == "") {
      line = getCleanLine(rtfFile);
      continue;
    }
    // std::cout << line << "\n";
    if (line.find("MASS") == 0) {
      // std::cout << line << "\n";
      auto parts = split(line);
      atomicMasses[parts[2]] = std::stof(parts[3]);
      // continue;
    }

    if (line.find("RESI") == 0 || line.find("PRES") == 0) {

      if (residuesInFile != 0) {
        residues[residue.residueName] = std::move(residue);
        // residues.push_back(std::move(residue));
      }
      ++residuesInFile;
      auto parts = split(line);
      // std::cout << line << "\n";
      if (parts[0] == "PRES")
        residue.isPatch = true;
      residue.residueName = parts[1];
      residue.charge = std::stof(parts[2]);
      AtomRTF atom;

      line = getCleanLine(rtfFile);

      while (line.find("RESI") != 0 && line.find("PRES") != 0) {
        if (line.find("END") == 0)
          break;
        if (line.find("ATOM") == 0) {
          // std::cout << "--" << line << "\n";
          parts = split(line);
          atom.atomName = parts[1];
          atom.atomType = parts[2];
          atom.charge = std::stof(parts[3]);

          residue.atoms.push_back(std::move(atom));
        }

        if (line.find("BOND") == 0 || line.find("DOUB") == 0) {
          parts = split(line);
          for (int i = 1; i < parts.size(); i += 2) {
            residue.bonds.push_back({parts[i], parts[i + 1]});
          }
        }
        if (line.find("DELE") == 0) {
          parts = split(line);
          if (parts[1] == "ATOM") {
            // Assuming only atoms are being deleteed
            // TODO : make it more general
            for (int i = 2; i < parts.size(); ++i) {
              residue.atomsToDelete.push_back(parts[i]);
            }
          }
        }
        line = getCleanLine(rtfFile);
      }

      // std::cout << line << "\n";
    } else {
      line = getCleanLine(rtfFile);
    }
  }
}

void CharmmResidueTopology::readRTF(std::vector<std::string> rtfFileNames) {
  for (auto rtfFileName : rtfFileNames) {
    readRTF(rtfFileName);
  }
}

void CharmmResidueTopology::print() {

  for (auto elem : residues) {
    auto residue = elem.second;
    if (!residue.isPatch) {
      std::cout << "Residue : " << residue.residueName << "\t" << residue.charge
                << "\n";
      std::cout << "Atoms : ";
      for (auto atom : residue.atoms) {
        std::cout << "\t" << atom.atomName << " " << atom.atomType << " "
                  << atom.charge << "\n";
      }
      std::cout << "Bonds : ";
      for (auto bond : residue.bonds) {
        std::cout << "(" << std::get<0>(bond) << "," << std::get<1>(bond)
                  << ") ";
      }
      std::cout << "\n";
    }
  }

  for (auto elem : atomicMasses) {
    std::cout << elem.first << " " << elem.second << "\n";
  }
}

Residue CharmmResidueTopology::applyPatch(Residue residue,
                                          const Residue patch) const {
  // const Residue patch = residues[patchName];
  residue.charge = patch.charge;

  for (auto atom : patch.atomsToDelete) {
  }
  return residue;
}