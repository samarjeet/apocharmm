// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmPSF.h"
#include "CharmmResidueTopology.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

CharmmPSF::CharmmPSF() {}

CharmmPSF::CharmmPSF(const std::string &fileName) {
  readCharmmPSFFile(fileName);
  buildTopologicalExclusions();
}

CharmmPSF::CharmmPSF(const CharmmPSF &psfIn)
    : numAtoms(psfIn.numAtoms), numBonds(psfIn.numBonds),
      numAngles(psfIn.numAngles), numDihedrals(psfIn.numDihedrals),
      numImpropers(psfIn.numImpropers), masses(psfIn.masses),
      charges(psfIn.charges), atomNames(psfIn.atomNames),
      atomTypes(psfIn.atomTypes), bonds(psfIn.bonds), angles(psfIn.angles),
      dihedrals(psfIn.dihedrals), impropers(psfIn.impropers),
      waterMolecules(psfIn.waterMolecules), residues(psfIn.residues),
      groups(psfIn.groups), connected12(psfIn.connected12),
      connected13(psfIn.connected13), connected14(psfIn.connected14),
      iblo14(psfIn.iblo14), inb14(psfIn.inb14) {}

void CharmmPSF::readCharmmPSFFile(std::string fileName) {
  // TODO
  // Convert to 0-base from the 1-base in the PSF file
  // Add Drude support

  // std::vector<std::string> atomTypePSF;
  std::string line;
  charges.clear();
  masses.clear();
  atomNames.clear();
  bonds.clear();
  angles.clear();
  dihedrals.clear();
  impropers.clear();

  std::ifstream psfFile(fileName);
  if (!psfFile.is_open()) {
    // std::cerr << "ERROR: Cannot open the file " << fileName <<
    // ".\nExiting\n";
    throw std::invalid_argument("ERROR: Cannot open the file " + fileName +
                                "\nExiting\n");
    exit(0);
  }

  originalPSFFileName = fileName;

  std::getline(psfFile, line);
  if (line.find_first_of("PSF") != 0) {
    // std::cerr << "Wrong format 'PSF' declaration line missing!\n";
    throw std::invalid_argument(
        "OH MAN Wrong format 'PSF' declaration line missing!\n");
    exit(0);
  }

  // read blank line(s) and title
  std::getline(psfFile, line);
  while (line.size() == 0)
    std::getline(psfFile, line);

  auto pos = line.find_first_of("!NTITLE");
  if (pos == std::string::npos) {
    // std::cerr << line << "\n";
    // std::cerr << "Wrong format no title!\n";
    throw std::invalid_argument(line + "\n\nWrong format no title!\n");
    exit(0);
  }
  int nTitleLines = std::stoul(line.substr(0, pos));
  for (int i = 0; i < nTitleLines; ++i) {
    std::getline(psfFile, line);
  }

  // read blank line(s) and NATOM line
  std::getline(psfFile, line);
  while (line.size() == 0)
    std::getline(psfFile, line);

  pos = line.find_first_of("!NATOM");
  if (pos == std::string::npos) {
    // std::cerr << line << "\n";
    // std::cerr << "Wrong format :  NATOM missing!\n";
    throw std::invalid_argument(line + "\n\nWrong format: NATOM missing!\n");
    exit(0);
  }
  numAtoms = std::stoul(line.substr(0, pos));

  int prevResId = 0;
  int startResIdAtom, endResIdAtom;
  int2 startEnd;
  std::vector<int2> groupsPrep;

  for (int i = 0; i < numAtoms; ++i) {
    // read atom information
    std::getline(psfFile, line);
    std::string segName, resName, resIdString, atom, atomType;
    int num, resId;
    double charge, mass;
    std::stringstream ss(line);
    ss >> num >> segName >> resIdString >> resName >> atom >> atomType >>
        charge >> mass;

    // resIds are not integers and can have characters eg: 27A
    resId = std::stoul(resIdString);

    charges.push_back(charge);
    masses.push_back(mass);
    atomNames.push_back(atom);
    atomTypes.push_back(atomType);

    if (prevResId != resId) {
      if (prevResId != 0) {
        endResIdAtom = num - 1;
        startEnd = {startResIdAtom - 1, endResIdAtom - 1};
        groupsPrep.push_back(startEnd);
        startResIdAtom = num;
      } else {
        startResIdAtom = 1;
      }
    }
    prevResId = resId;
  }
  startEnd = {startResIdAtom - 1, numAtoms - 1};
  groupsPrep.push_back(startEnd);
  residues.allocate(groupsPrep.size());
  residues.set(groupsPrep);
  // setMasses(massesPSF);

  // read blank line(s) and NBOND line
  std::getline(psfFile, line);
  while (line.size() == 0)
    std::getline(psfFile, line);

  pos = line.find_first_of("!NBOND");
  if (pos == std::string::npos) {
    // std::cerr << line << "\n";
    // std::cerr << "Wrong format :  NBOND missing!\n";
    throw std::invalid_argument(line + "\n\nWrong format : NBOND missing!\n");
    exit(0);
  }

  numBonds = std::stoul(line.substr(0, pos));

  int bondId = 0;
  for (int i = 0; i < (numBonds - 1) / 4 + 1; ++i) {
    // read bond information
    std::getline(psfFile, line);
    std::stringstream ss(line);
    int atom1, atom2;
    for (int j = 0; j < 4 && bondId < numBonds; ++j, bondId++) {
      ss >> atom1 >> atom2;
      // bonds.emplace_back(atom1-1, atom2-1);
      bonds.emplace_back(Bond{atom1 - 1, atom2 - 1});
    }
  }

  // read blank line(s) and numAngles line
  std::getline(psfFile, line);
  while (line.size() == 0)
    std::getline(psfFile, line);

  pos = line.find_first_of("!numAngles");
  if (pos == std::string::npos) {
    // std::cerr << line << "\n";
    // std::cerr << "Wrong format :  numAngles missing!\n";
    throw std::invalid_argument(line +
                                "\n\nWrong format : numAngles missing!\n");
    exit(0);
  }

  numAngles = std::stoul(line.substr(0, pos));

  for (int i = 0; i < (numAngles - 1) / 3 + 1; ++i) {
    // read angle information
    std::getline(psfFile, line);
    std::stringstream ss(line);
    int atom1, atom2, atom3;
    for (int j = 0; j < 3 && angles.size() < numAngles; ++j) {
      ss >> atom1 >> atom2 >> atom3;
      angles.emplace_back(Angle{atom1 - 1, atom2 - 1, atom3 - 1});
    }
  }
  assert(numAngles == angles.size());

  // read blank line(s) and NPHI line
  std::getline(psfFile, line);
  while (line.size() == 0)
    std::getline(psfFile, line);

  pos = line.find_first_of("!NPHI");
  if (pos == std::string::npos) {
    // std::cerr << line << "\n";
    // std::cerr << "Wrong format :  NPHI missing!\n";
    throw std::invalid_argument(line + "\n\nWrong format : NPHI missing!\n");
    exit(0);
  }

  int nPhi = std::stoul(line.substr(0, pos));
  numDihedrals = nPhi;

  if (nPhi > 0)
    for (int i = 0; i < (nPhi - 1) / 2 + 1; ++i) {
      // read dihedral information
      std::getline(psfFile, line);
      std::stringstream ss(line);
      int atom1, atom2, atom3, atom4;
      for (int j = 0; j < 2 && dihedrals.size() < nPhi; ++j) {
        ss >> atom1 >> atom2 >> atom3 >> atom4;
        dihedrals.emplace_back(
            Dihedral{atom1 - 1, atom2 - 1, atom3 - 1, atom4 - 1});
      }
    }
  assert((nPhi == dihedrals.size()));

  // read blank line(s) and NIMPHI line
  std::getline(psfFile, line);
  while (line.size() == 0)
    std::getline(psfFile, line);

  pos = line.find_first_of("!NIMPHI");
  if (pos == std::string::npos) {
    // std::cerr << line << "\n";
    // std::cerr << "Wrong format :  NIMPHI missing!\n";
    throw std::invalid_argument(line + "\n\nWrong format : NIMPHI missing!\n");
    exit(0);
  }

  int nImPhi = std::stoul(line.substr(0, pos));
  numImpropers = nImPhi;
  // std::cout << "Number of Improper dihedrals is " << nImPhi << std::endl;

  if (nImPhi > 0)
    for (int i = 0; i < (nImPhi - 1) / 2 + 1; ++i) {
      // read improper dihedral information
      std::getline(psfFile, line);
      std::stringstream ss(line);
      int atom1, atom2, atom3, atom4;
      for (int j = 0; j < 2 && impropers.size() < nImPhi; ++j) {
        ss >> atom1 >> atom2 >> atom3 >> atom4;
        impropers.emplace_back(
            Dihedral{atom1 - 1, atom2 - 1, atom3 - 1, atom4 - 1});
      }
    }
  assert((nImPhi == impropers.size()));

  // read blank line
  std::getline(psfFile, line);

  while (psfFile) {
    std::getline(psfFile, line);
    // std::cout << line << "\n";
  }

  createConnectedComponents();
}

void CharmmPSF::buildTopologicalExclusions() {
  connected12.resize(numAtoms);
  connected13.resize(numAtoms);
  connected14.resize(numAtoms);

  for (const auto &bond : bonds) {
    connected12[bond.atom1].insert(bond.atom2);
    connected12[bond.atom2].insert(bond.atom1);
  }

  for (const auto &bond : bonds) {
    for (const auto &atom1bond : connected12[bond.atom2]) {
      if (atom1bond != bond.atom1)
        connected13[bond.atom1].insert(atom1bond);
    }
    for (const auto &atom2bond : connected12[bond.atom1]) {
      if (atom2bond != bond.atom2)
        connected13[bond.atom2].insert(atom2bond);
    }
  }

  for (int atom1 = 0; atom1 < numAtoms; ++atom1) {
    for (const auto &atom2 : connected13[atom1]) {
      for (const auto &atom3 : connected12[atom2]) {
        if (connected12[atom1].count(atom3) == 0)
          connected14[atom1].insert(atom3);
      }
    }
  }
  inb14.clear();
  iblo14.clear();

  for (int atom = 0; atom < numAtoms; ++atom) {
    std::set<int> connectedAtoms;
    for (auto &atom2 : connected12[atom])
      connectedAtoms.insert(atom2);
    for (auto &atom3 : connected13[atom])
      connectedAtoms.insert(atom3);
    for (auto &atom4 : connected14[atom])
      connectedAtoms.insert(atom4);
    for (auto &connectedAtom : connectedAtoms) {
      if (connectedAtom > atom) {
        inb14.push_back(connectedAtom + 1);
      }
    }
    iblo14.push_back(inb14.size());
  }
}

void CharmmPSF::setHydrogenMass(double _newHyrogenMass) {
  for (int i = 0; i < numAtoms; ++i) {
    if (atomTypes[i][0] == 'H') {
      masses[i] = _newHyrogenMass;
    }
  }
}

CudaContainer<int4> CharmmPSF::getWaterMolecules() {

  if (waterMolecules.size() == 0) {
    int pos = 0;
    std::vector<int4> waterMols;
    while (pos < numAtoms - 2) {
      if (atomTypes[pos] == "OT" && atomTypes[pos + 1] == "HT" &&
          atomTypes[pos + 2] == "HT") {

        int4 water = {pos, pos + 1, pos + 2, 0};
        waterMols.push_back(water);
        pos += 3;
      } else {
        ++pos;
      }
    }
    waterMolecules.allocate(waterMols.size());
    waterMolecules.set(waterMols);
  }
  return waterMolecules;
}

CudaContainer<int2> CharmmPSF::getResidues() { return residues; }

int find(int node, const std::vector<int> &link) {
  while (node != link[node])
    node = link[node];
  return node;
}

bool same(int node1, int node2, const std::vector<int> &link) {
  return find(node1, link) == find(node2, link);
}

void CharmmPSF::createConnectedComponents() {
  std::vector<int> link(numAtoms, -1);
  // std::vector<int> size(numAtoms, 1);
  for (int i = 0; i < numAtoms; ++i) {
    link[i] = i;
  }
  for (auto bond : bonds) {
    int rep1 = find(bond.atom1, link);
    int rep2 = find(bond.atom2, link);
    if (rep1 != rep2) {
      if (rep1 > rep2)
        link[rep2] = rep1;
      else
        link[rep1] = rep2;
    }
  }
  // std::cout << find(0, link) << std::endl;
  int startAtom = 0;
  std::vector<int2> groupsPrep;
  int2 startEnd;
  while (startAtom < numAtoms) {
    int endAtom = find(startAtom, link);
    startEnd = {startAtom, endAtom};
    startAtom = endAtom + 1;
    groupsPrep.push_back(startEnd);
  }
  groups.allocate(groupsPrep.size());
  groups.set(groupsPrep);
}

int CharmmPSF::getDegreesOfFreedom() {
  int ndegf = 0;
  // TODO This part of the code seems... useless
  //  VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
  for (auto group : groups.getHostArray()) {
    int size = group.y - group.x + 1;
    ndegf += (3 * size - 6);
  }
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  ndegf = 3 * numAtoms - 6;
  return ndegf;
}

CudaContainer<int2> CharmmPSF::getGroups() { return groups; }

InclusionExclusion CharmmPSF::getInclusionExclusionLists() {
  std::vector<int> inclusion, exclusion;

  // Fill in from connected12,13,14
  for (int atomI = 0; atomI < numAtoms; ++atomI) {
    for (auto atomJ : connected14[atomI]) {
      if (atomJ > atomI) {
        // ex : removing cycle (proline)
        if (connected12[atomI].find(atomJ) == connected12[atomI].end() &&
            connected13[atomI].find(atomJ) == connected13[atomI].end()) {
          inclusion.push_back(atomI);
          inclusion.push_back(atomJ);
        }
      }
    }

    std::set<int> exclusionAtoms;
    exclusionAtoms.clear();

    for (auto atomJ : connected12[atomI]) {
      if (atomJ > atomI) {
        exclusionAtoms.insert(atomJ);
      }
    }

    for (auto atomJ : connected13[atomI]) {
      if (atomJ > atomI) {
        exclusionAtoms.insert(atomJ);
      }
    }

    for (auto atomJ : exclusionAtoms) {
      exclusion.push_back(atomI);
      exclusion.push_back(atomJ);
    }
  }

  std::vector<int> sizes{(int)inclusion.size() / 2, (int)exclusion.size() / 2};
  std::vector<int> in14_ex14;
  in14_ex14.insert(in14_ex14.end(), inclusion.begin(), inclusion.end());
  in14_ex14.insert(in14_ex14.end(), exclusion.begin(), exclusion.end());

  // std::ofstream out;
  // out.open("in14_ex14_list.txt");
  // for (int i=0; i < inclusion.size()/2; ++i){
  //  out << inclusion[2*i] << "\t" << inclusion[2*i + 1] << "\n";
  //}
  // out.close();

  return InclusionExclusion(sizes, in14_ex14);
}

void CharmmPSF::setAtomCharges(std::vector<double> chargesIn) {
  charges = chargesIn;
}

void CharmmPSF::generate(const CharmmResidueTopology &rtf,
                         const std::vector<std::string> &sequence,
                         std::string segment) {
  auto atomicMasses = rtf.getAtomicMasses();

  auto rtfResidues = rtf.getResidues();
  int atomNumber = 0;
  int residueNumber = 0;

  for (auto resName : sequence) {
    ++residueNumber;
    auto residue = rtfResidues[resName];
    if (residueNumber == 1) {
      auto patch = rtfResidues["NTER"];
      residue = rtf.applyPatch(residue, patch);
    }
    for (auto atom : residue.atoms) {
      ++atomNumber;
      std::cout << atomNumber << '\t' << segment << '\t' << residueNumber
                << '\t' << resName << '\t' << atom.atomName << '\t'
                << atom.atomType << '\t' << atom.charge << '\t'
                << atomicMasses[atom.atomType] << '\n';
    }

    // std::cout << "\n";
  }
  std::cout << "\n";
}

void CharmmPSF::append(const CharmmResidueTopology &rtf,
                       const std::vector<std::string> &sequence,
                       std::string segment) {}