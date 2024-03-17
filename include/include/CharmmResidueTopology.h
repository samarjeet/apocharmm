// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE
//

#pragma once

#include "CudaContainer.h"
#include <map>
#include <string>
/**
 * @brief An atom entry in a residue in RTF file
 *
 */
struct AtomRTF {
  std::string atomName;
  std::string atomType;
  float charge;
};

/**
 * @brief A residue entry from RTF file
 *
 */
struct Residue {
  std::string residueName;
  float charge;
  std::vector<AtomRTF> atoms;

  std::vector<std::tuple<std::string, std::string>> bonds;

  // TODO : move this to PatchResidue sub-struct
  std::vector<std::string> atomsToDelete;
  bool isPatch = false;
};

struct PatchResidue : public Residue {};

/**
 * @brief This class handles CHARMM RTF files
 * It can read one or more RTF files
 */
class CharmmResidueTopology {
public:
  CharmmResidueTopology();
  CharmmResidueTopology(std::string rtfFileName);

  void readRTF(std::string fileName);
  void readRTF(std::vector<std::string> rtfFileNames);

  void print();

  const std::map<std::string, Residue> getResidues() const { return residues; }

  const std::map<std::string, float> getAtomicMasses() const {
    return atomicMasses;
  }
  Residue applyPatch(Residue residue, const Residue patch) const;

private:
  // std::vector<Residue> residues;
  std::map<std::string, Residue> residues;
  std::map<std::string, float> atomicMasses;
};