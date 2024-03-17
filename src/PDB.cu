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

#include "PDB.h"

PDB::PDB(const std::string &fileName) { readPDBFile(fileName); }

/*

COLUMNS        DATA TYPE       CONTENTS                            
--------------------------------------------------------------------------------
   1 -  6        Record name     "ATOM  "                                            
   7 - 11        Integer         Atom serial number.                   
  13 - 16        Atom            Atom name.                            
  17             Character       Alternate location indicator.         
  18 - 20        Residue name    Residue name.                         
  22             Character       Chain identifier.                     
  23 - 26        Integer         Residue sequence number.              
  27             AChar           Code for insertion of residues.       
  31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.         
  39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.         
  47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.         
  55 - 60        Real(6.2)       Occupancy.                            
  61 - 66        Real(6.2)       Temperature factor (Default = 0.0).                
  73 - 76        LString(4)      Segment identifier, left-justified.   
  77 - 78        LString(2)      Element symbol, right-justified.      
  79 - 80        LString(2)      Charge on the atom.
*/

static std::vector<std::string> split(std::string line) {
  //std::stringstream ss(line);
  std::string atomId, resId, resName, atom, x, y, z;
  //ss >> atomId >> resId >> resName >> atom >> x >> y >> z;

  atomId = line.substr(6, 5);
  atom = line.substr(12, 4);
  resName = line.substr(17, 3);
  resId = line.substr(22, 4);
  x = line.substr(30, 8);
  y = line.substr(38, 8);
  z = line.substr(46, 8);
  std::vector<std::string> content = {atomId, resId, resName, atom, x, y, z};

  return content;
}

void PDB::readPDBFile(std::string fileName) {
  // nelecting lines that do not contain ATOM as the first word
  std::string line;
  std::ifstream pdbFile(fileName);

  if (!pdbFile.is_open()) {
    //std::exception ; //<< "ERROR: Cannot open the file " << fileName << "\n. Exiting\n";
    //std::bad_exception ; //<< "ERROR: Cannot open the file " << fileName << "\n. Exiting\n";
    std::cout << "ERROR: Cannot open the file " << fileName << "\nExiting\n";
    exit(0);
  }

  std::getline(pdbFile, line);
  while (line.size() != 0) {
    if (line.find("ATOM") == 0  || line.find("HETATM") == 0 ){
      //std::cout << line << "\n";
      auto content = split(line);
      float x = std::stof(content[4]);
      float y = std::stof(content[5]);
      float z = std::stof(content[6]);
      coords.push_back(make_float4(x, y, z, 0.0));
    }
    std::getline(pdbFile, line);
  }
}
