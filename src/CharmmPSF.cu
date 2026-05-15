// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CharmmPSF.h"

#include "str_utils.h"

#include <iostream> // TMP?

CharmmPSF::CharmmPSF(void)
    : m_NumAtoms(-1), m_Masses(), m_Charges(), m_AtomNames(), m_AtomTypes(),
      m_NumBonds(-1), m_Bonds(), m_NumAngles(-1), m_Angles(),
      m_NumDihedrals(-1), m_Dihedrals(), m_NumImpropers(-1), m_Impropers(),
      m_NumCrossTerms(-1), m_CrossTerms(), m_Connected12(), m_Connected13(),
      m_Connected14(), m_Iblo14(), m_Inb14(), m_WaterMolecules(), m_Residues(),
      m_Groups(), m_FileName("") {}

CharmmPSF::CharmmPSF(const std::string &fileName) : CharmmPSF() {
  this->readCharmmPSF(fileName);
  this->initializeWaterMolecules();
  this->createConnectedComponents();
  this->buildTopologicalExclusions();
}

CharmmPSF::CharmmPSF(const CharmmPSF &other)
    : m_NumAtoms(other.m_NumAtoms), m_Masses(other.m_Masses),
      m_Charges(other.m_Charges), m_AtomNames(other.m_AtomNames),
      m_AtomTypes(other.m_AtomTypes), m_NumBonds(other.m_NumBonds),
      m_Bonds(other.m_Bonds), m_NumAngles(other.m_NumAngles),
      m_Angles(other.m_Angles), m_NumDihedrals(other.m_NumDihedrals),
      m_Dihedrals(other.m_Dihedrals), m_NumImpropers(other.m_NumImpropers),
      m_Impropers(other.m_Impropers), m_NumCrossTerms(other.m_NumCrossTerms),
      m_CrossTerms(other.m_CrossTerms), m_Connected12(other.m_Connected12),
      m_Connected13(other.m_Connected13), m_Connected14(other.m_Connected14),
      m_Iblo14(other.m_Iblo14), m_Inb14(other.m_Inb14),
      m_WaterMolecules(other.m_WaterMolecules), m_Residues(other.m_Residues),
      m_Groups(other.m_Groups), m_FileName(other.m_FileName) {}

CharmmPSF::CharmmPSF(const CharmmPSF &&other)
    : m_NumAtoms(other.m_NumAtoms), m_Masses(other.m_Masses),
      m_Charges(other.m_Charges), m_AtomNames(other.m_AtomNames),
      m_AtomTypes(other.m_AtomTypes), m_NumBonds(other.m_NumBonds),
      m_Bonds(other.m_Bonds), m_NumAngles(other.m_NumAngles),
      m_Angles(other.m_Angles), m_NumDihedrals(other.m_NumDihedrals),
      m_Dihedrals(other.m_Dihedrals), m_NumImpropers(other.m_NumImpropers),
      m_Impropers(other.m_Impropers), m_NumCrossTerms(other.m_NumCrossTerms),
      m_CrossTerms(other.m_CrossTerms), m_Connected12(other.m_Connected12),
      m_Connected13(other.m_Connected13), m_Connected14(other.m_Connected14),
      m_Iblo14(other.m_Iblo14), m_Inb14(other.m_Inb14),
      m_WaterMolecules(other.m_WaterMolecules), m_Residues(other.m_Residues),
      m_Groups(other.m_Groups), m_FileName(other.m_FileName) {}

void CharmmPSF::setNumAtoms(const int numAtoms) {
  m_NumAtoms = numAtoms;
  m_Masses.resize(numAtoms);
  m_Charges.resize(numAtoms);
  m_AtomNames.resize(numAtoms);
  m_AtomTypes.resize(numAtoms);
  return;
}

void CharmmPSF::setAtomCharges(const std::vector<double> &charges) {
  m_Charges = charges;
  return;
}

int CharmmPSF::getNumAtoms(void) const { return m_NumAtoms; }

int CharmmPSF::getNumBonds(void) const { return m_NumBonds; }

int CharmmPSF::getNumAngles(void) const { return m_NumAngles; }

int CharmmPSF::getNumDihedrals(void) const { return m_NumDihedrals; }

int CharmmPSF::getNumImpropers(void) const { return m_NumImpropers; }

int CharmmPSF::getNumCrossTerms(void) const { return m_NumCrossTerms; }

const std::vector<double> &CharmmPSF::getMasses(void) const { return m_Masses; }

const std::vector<double> &CharmmPSF::getCharges(void) const {
  return m_Charges;
}

const std::vector<std::string> &CharmmPSF::getAtomNames(void) const {
  return m_AtomNames;
}

const std::vector<std::string> &CharmmPSF::getAtomTypes(void) const {
  return m_AtomTypes;
}

const std::vector<Bond> &CharmmPSF::getBonds(void) const { return m_Bonds; }

const std::vector<Angle> &CharmmPSF::getAngles(void) const { return m_Angles; }

const std::vector<Dihedral> &CharmmPSF::getDihedrals(void) const {
  return m_Dihedrals;
}

const std::vector<Dihedral> &CharmmPSF::getImpropers(void) const {
  return m_Impropers;
}

const std::vector<CrossTerm> &CharmmPSF::getCrossTerms(void) const {
  return m_CrossTerms;
}

const std::vector<std::set<int>> &CharmmPSF::getConnected12(void) const {
  return m_Connected12;
}

const std::vector<std::set<int>> &CharmmPSF::getConnected13(void) const {
  return m_Connected13;
}

const std::vector<std::set<int>> &CharmmPSF::getConnected14(void) const {
  return m_Connected14;
}

const std::vector<int> &CharmmPSF::getIblo14(void) const { return m_Iblo14; }

const std::vector<int> &CharmmPSF::getInb14(void) const { return m_Inb14; }

const CudaContainer<int4> &CharmmPSF::getWaterMolecules(void) const {
  return m_WaterMolecules;
}

const CudaContainer<int2> &CharmmPSF::getResidues(void) const {
  return m_Residues;
}

const CudaContainer<int2> &CharmmPSF::getGroups(void) const { return m_Groups; }

const std::string &CharmmPSF::getFileName(void) const { return m_FileName; }

std::vector<double> &CharmmPSF::getMasses(void) { return m_Masses; }

std::vector<double> &CharmmPSF::getCharges(void) { return m_Charges; }

std::vector<std::string> &CharmmPSF::getAtomNames(void) { return m_AtomNames; }

std::vector<std::string> &CharmmPSF::getAtomTypes(void) { return m_AtomTypes; }

std::vector<Bond> &CharmmPSF::getBonds(void) { return m_Bonds; }

std::vector<Angle> &CharmmPSF::getAngles(void) { return m_Angles; }

std::vector<Dihedral> &CharmmPSF::getDihedrals(void) { return m_Dihedrals; }

std::vector<Dihedral> &CharmmPSF::getImpropers(void) { return m_Impropers; }

std::vector<CrossTerm> &CharmmPSF::getCrossTerms(void) { return m_CrossTerms; }

std::vector<std::set<int>> &CharmmPSF::getConnected12(void) {
  return m_Connected12;
}

std::vector<std::set<int>> &CharmmPSF::getConnected13(void) {
  return m_Connected13;
}

std::vector<std::set<int>> &CharmmPSF::getConnected14(void) {
  return m_Connected14;
}

std::vector<int> &CharmmPSF::getIblo14(void) { return m_Iblo14; }

std::vector<int> &CharmmPSF::getInb14(void) { return m_Inb14; }

CudaContainer<int4> &CharmmPSF::getWaterMolecules(void) {
  return m_WaterMolecules;
}

CudaContainer<int2> &CharmmPSF::getResidues(void) { return m_Residues; }

CudaContainer<int2> &CharmmPSF::getGroups(void) { return m_Groups; }

std::string &CharmmPSF::getFileName(void) { return m_FileName; }

double CharmmPSF::getTotalMass(void) const {
  double totalMass = 0.0;
  for (int i = 0; i < m_NumAtoms; i++)
    totalMass += m_Masses[i];
  return totalMass;
}

InclusionExclusion CharmmPSF::getInclusionExclusionLists(void) const {
  // Fill in from 1-2, 1-3, 1-4 connections
  std::vector<int> inclusion, exclusion;
  for (int iatom = 0; iatom < m_NumAtoms; iatom++) {
    // Inclusions
    for (const int jatom : m_Connected14[iatom]) {
      if (jatom > iatom) {
        if ((m_Connected12[iatom].find(jatom) == m_Connected12[iatom].end()) &&
            (m_Connected13[iatom].find(jatom) == m_Connected13[iatom].end())) {
          inclusion.push_back(iatom);
          inclusion.push_back(jatom);
        }
      }
    }

    // Exclusions
    std::set<int> ex;
    for (const int jatom : m_Connected12[iatom]) {
      if (jatom > iatom)
        ex.insert(jatom);
    }
    for (const int jatom : m_Connected13[iatom]) {
      if (jatom > iatom)
        ex.insert(jatom);
    }
    for (const int jatom : ex) {
      exclusion.push_back(iatom);
      exclusion.push_back(jatom);
    }
  }

  std::vector<int> sizes = {static_cast<int>(inclusion.size() / 2),
                            static_cast<int>(exclusion.size() / 2)};
  std::vector<int> in14_ex14;
  in14_ex14.insert(in14_ex14.end(), inclusion.begin(), inclusion.end());
  in14_ex14.insert(in14_ex14.end(), exclusion.begin(), exclusion.end());
  // for (std::size_t i = 0; i < inclusion.size(); i++)
  //   std::cout << "inclusion[" << i << "] = " << inclusion[i] << std::endl;
  // for (std::size_t i = 0; i < exclusion.size(); i++)
  //   std::cout << "exclusion[" << i << "] = " << exclusion[i] << std::endl;
  // for (std::size_t i = 0; i < in14_ex14.size(); i++)
  //   std::cout << "in14_ex14[" << i << "] = " << in14_ex14[i] << std::endl;

  return InclusionExclusion(sizes, in14_ex14);
}

void CharmmPSF::initializeWaterMolecules(void) {
  m_WaterMolecules.clear();

  for (int i = 0; i < m_NumAtoms - 2; i++) {
    if ((m_AtomTypes[i + 0] == "OT") && (m_AtomTypes[i + 1] == "HT") &&
        (m_AtomTypes[i + 2] == "HT")) {
      m_WaterMolecules.push_back(make_int4(i + 0, i + 1, i + 2, 0));
      i += 2;
    }
  }
  m_WaterMolecules.shrink_to_fit();

  return;
}

int find(int node, const std::vector<int> &link) {
  while (node != link[node])
    node = link[node];
  return node;
}

void CharmmPSF::createConnectedComponents(void) {
  std::vector<int> link(m_NumAtoms);
  for (int i = 0; i < m_NumAtoms; i++)
    link[i] = i;

  for (int i = 0; i < m_NumBonds; i++) {
    const int iatom = m_Bonds[i].iatom;
    const int jatom = m_Bonds[i].jatom;
    const int rep1 = find(iatom, link);
    const int rep2 = find(jatom, link);
    if (rep1 != rep2) {
      if (rep1 > rep2)
        link[rep2] = rep1;
      else
        link[rep1] = rep2;
    }
  }

  int startAtom = 0;
  while (startAtom < m_NumAtoms) {
    int endAtom = find(startAtom, link);
    m_Groups.push_back(make_int2(startAtom, endAtom));
    startAtom = endAtom + 1;
  }
  m_Groups.shrink_to_fit();

  return;
}

void CharmmPSF::buildTopologicalExclusions(void) {
  m_Connected12.resize(m_NumAtoms);
  m_Connected13.resize(m_NumAtoms);
  m_Connected14.resize(m_NumAtoms);

  // 1-2 exclusions
  for (const Bond bond : m_Bonds) {
    m_Connected12[bond.iatom].insert(bond.jatom);
    m_Connected12[bond.jatom].insert(bond.iatom);
  }

  // 1-3 exclusions
  for (const Bond bond : m_Bonds) {
    for (const int iatom : m_Connected12[bond.jatom]) {
      if (iatom != bond.iatom)
        m_Connected13[bond.iatom].insert(iatom);
    }
    for (const int jatom : m_Connected12[bond.iatom]) {
      if (jatom != bond.jatom)
        m_Connected13[bond.jatom].insert(jatom);
    }
  }

  // 1-4 exclusions
  for (int iatom = 0; iatom < m_NumAtoms; iatom++) {
    for (const int jatom : m_Connected13[iatom]) {
      for (const int katom : m_Connected12[jatom]) {
        if (m_Connected12[iatom].count(katom) == 0)
          m_Connected14[iatom].insert(katom);
      }
    }
  }

  m_Iblo14.clear();
  m_Inb14.clear();

  for (int iatom = 0; iatom < m_NumAtoms; iatom++) {
    std::set<int> connectedAtoms;
    for (const int jatom : m_Connected12[iatom])
      connectedAtoms.insert(jatom);
    for (const int katom : m_Connected13[iatom])
      connectedAtoms.insert(katom);
    for (const int latom : m_Connected14[iatom])
      connectedAtoms.insert(latom);
    for (const int connectedAtom : connectedAtoms) {
      if (connectedAtom > iatom)
        m_Inb14.push_back(connectedAtom + 1);
    }
    m_Iblo14.push_back(static_cast<int>(m_Inb14.size()));
  }
  m_Iblo14.shrink_to_fit();
  m_Inb14.shrink_to_fit();

  return;
}

void CharmmPSF::readCharmmPSF(const std::string &fileName) {
  std::string fileData = "";
  apo::read_file_into_string(fileData, fileName);

  std::size_t pos = 0;
  std::string line = "";
  bool foundSection = false;
  std::vector<std::string> tokens;

  // Parse TITLE section
  foundSection = false;
  do {
    if (pos >= fileData.length()) {
      throw std::runtime_error("Could not find TITLE section in PSF \"" +
                               fileName + "\"");
    }
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    if ((tokens.size() == 2) && (tokens[1] == "!NTITLE"))
      foundSection = true;
  } while (foundSection == false);
  const unsigned long long int ntitle = std::stoull(tokens[0]);
  const unsigned long long int nlineTitle = ntitle;
  for (unsigned long long int i = 0; i < nlineTitle; i++) {
    line.clear();
    apo::get_line(line, pos, fileData);
  }

  // Parse ATOM section
  foundSection = false;
  do {
    if (pos >= fileData.length()) {
      throw std::runtime_error("Could not find ATOM section in PSF \"" +
                               fileName + "\"");
    }
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    if ((tokens.size() == 2) && (tokens[1] == "!NATOM"))
      foundSection = true;
  } while (foundSection == false);
  const unsigned long long int natom = std::stoull(tokens[0]);
  const unsigned long long int nlineAtom = natom;
  this->setNumAtoms(static_cast<int>(natom));
  int resiOld = 0;
  int resiStartIdx = 0, resiEndIdx = -1;
  for (unsigned long long int i = 0; i < nlineAtom; i++) {
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    m_Masses[i] = std::stod(tokens[7]);
    m_Charges[i] = std::stod(tokens[6]);
    m_AtomNames[i] = tokens[4];
    m_AtomTypes[i] = tokens[5];
    const int resi = std::stoi(tokens[2]);
    if (resiOld == 0)
      resiOld = resi;
    if (resiOld != resi) {
      resiEndIdx = i - 1;
      m_Residues.push_back(make_int2(resiStartIdx, resiEndIdx));
      resiStartIdx = i;
    }
    resiOld = resi;
  }
  m_Residues.push_back(make_int2(resiStartIdx, m_NumAtoms - 1));
  m_Residues.shrink_to_fit();

  // Parse BOND section
  foundSection = false;
  do {
    if (pos >= fileData.length()) {
      throw std::runtime_error("Could not find BOND section in PSF \"" +
                               fileName + "\"");
    }
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    if ((tokens.size() >= 2) && (tokens[1] == "!NBOND:"))
      foundSection = true;
  } while (foundSection == false);
  const unsigned long long int nbond = std::stoull(tokens[0]);
  const unsigned long long int nlineBond =
      nbond / 4 + ((nbond % 4 == 0) ? 0 : 1);
  m_NumBonds = static_cast<int>(nbond);
  m_Bonds.resize(nbond);
  unsigned long long int ibond = 0;
  for (unsigned long long int i = 0; i < nlineBond; i++) {
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    for (std::size_t j = 0; j < tokens.size(); j += 2) {
      if (ibond < nbond) {
        m_Bonds[ibond].iatom = std::stoi(tokens[j + 0]) - 1;
        m_Bonds[ibond].jatom = std::stoi(tokens[j + 1]) - 1;
        ibond++;
      }
    }
  }

  // Parse ANGLe section
  foundSection = false;
  do {
    if (pos >= fileData.length()) {
      throw std::runtime_error("Could not find ANGLE section in PSF \"" +
                               fileName + "\"");
    }
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    if ((tokens.size() >= 2) && (tokens[1] == "!NTHETA:"))
      foundSection = true;
  } while (foundSection == false);
  const unsigned long long int ntheta = std::stoull(tokens[0]);
  const unsigned long long int nlineTheta =
      ntheta / 3 + ((ntheta % 3 == 0) ? 0 : 1);
  m_NumAngles = static_cast<int>(ntheta);
  m_Angles.resize(ntheta);
  unsigned long long int itheta = 0;
  for (unsigned long long int i = 0; i < nlineTheta; i++) {
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    for (std::size_t j = 0; j < tokens.size(); j += 3) {
      if (itheta < ntheta) {
        m_Angles[itheta].iatom = std::stoi(tokens[j + 0]) - 1;
        m_Angles[itheta].jatom = std::stoi(tokens[j + 1]) - 1;
        m_Angles[itheta].katom = std::stoi(tokens[j + 2]) - 1;
        itheta++;
      }
    }
  }

  // Parse DIHEdral section
  foundSection = false;
  do {
    if (pos >= fileData.length()) {
      throw std::runtime_error("Could not find DIHEDRAL section in PSF \"" +
                               fileName + "\"");
    }
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    if ((tokens.size() >= 2) && (tokens[1] == "!NPHI:"))
      foundSection = true;
  } while (foundSection == false);
  const unsigned long long int nphi = std::stoull(tokens[0]);
  const unsigned long long int nlinePhi = nphi / 2 + ((nphi % 2 == 0) ? 0 : 1);
  m_NumDihedrals = static_cast<int>(nphi);
  m_Dihedrals.resize(nphi);
  unsigned long long int iphi = 0;
  for (unsigned long long int i = 0; i < nlinePhi; i++) {
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    for (std::size_t j = 0; j < tokens.size(); j += 4) {
      if (iphi < nphi) {
        m_Dihedrals[iphi].iatom = std::stoi(tokens[j + 0]) - 1;
        m_Dihedrals[iphi].jatom = std::stoi(tokens[j + 1]) - 1;
        m_Dihedrals[iphi].katom = std::stoi(tokens[j + 2]) - 1;
        m_Dihedrals[iphi].latom = std::stoi(tokens[j + 3]) - 1;
        iphi++;
      }
    }
  }

  // Parse IMPRoper dihedral section
  foundSection = false;
  do {
    if (pos >= fileData.length()) {
      throw std::runtime_error("Could not find IMPROPER section in PSF \"" +
                               fileName + "\"");
    }
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    if ((tokens.size() >= 2) && (tokens[1] == "!NIMPHI:"))
      foundSection = true;
  } while (foundSection == false);
  const unsigned long long int nimphi = std::stoull(tokens[0]);
  const unsigned long long int nlineImphi =
      nimphi / 2 + ((nimphi % 2 == 0) ? 0 : 1);
  m_NumImpropers = static_cast<int>(nimphi);
  m_Impropers.resize(nimphi);
  unsigned long long int iimphi = 0;
  for (unsigned long long int i = 0; i < nlineImphi; i++) {
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    for (std::size_t j = 0; j < tokens.size(); j += 4) {
      if (iimphi < nimphi) {
        m_Impropers[iimphi].iatom = std::stoi(tokens[j + 0]) - 1;
        m_Impropers[iimphi].jatom = std::stoi(tokens[j + 1]) - 1;
        m_Impropers[iimphi].katom = std::stoi(tokens[j + 2]) - 1;
        m_Impropers[iimphi].latom = std::stoi(tokens[j + 3]) - 1;
        iimphi++;
      }
    }
  }

  // Parse DONOr section
  foundSection = false;
  do {
    if (pos >= fileData.length()) {
      throw std::runtime_error("Could not find DONOR section in PSF \"" +
                               fileName + "\"");
    }
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    if ((tokens.size() >= 2) && (tokens[1] == "!NDON:"))
      foundSection = true;
  } while (foundSection == false);
  const unsigned long long int ndon = std::stoull(tokens[0]);
  const unsigned long long int nlineDon = ndon / 4 + ((ndon % 4 == 0) ? 0 : 1);
  // unsigned long long int idon = 0;
  for (unsigned long long int i = 0; i < nlineDon; i++) {
    line.clear();
    apo::get_line(line, pos, fileData);
    // tokens.clear();
    // tokens = apo::split(line);
  }

  // Parse ACCEptor section
  foundSection = false;
  do {
    if (pos >= fileData.length()) {
      throw std::runtime_error("Could not find ACCEPTOR section in PSF \"" +
                               fileName + "\"");
    }
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    if ((tokens.size() >= 2) && (tokens[1] == "!NACC:"))
      foundSection = true;
  } while (foundSection == false);
  const unsigned long long int nacc = std::stoull(tokens[0]);
  const unsigned long long int nlineAcc = nacc / 4 + ((nacc % 4 == 0) ? 0 : 1);
  // unsigned long long int iacc = 0;
  for (unsigned long long int i = 0; i < nlineAcc; i++) {
    line.clear();
    apo::get_line(line, pos, fileData);
    // tokens.clear();
    // tokens = apo::split(line);
  }

  // Other sections are optional

  // Parse CRoss TERM section
  foundSection = false;
  do {
    if (pos >= fileData.length()) {
      throw std::runtime_error("Could not find CROSS-TERM section in PSF \"" +
                               fileName + "\"");
    }
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    if ((tokens.size() >= 2) && (tokens[1] == "!NCRTERM:"))
      foundSection = true;
  } while (foundSection == false);
  const unsigned long long int ncrterm = std::stoull(tokens[0]);
  const unsigned long long int nlineCrterm = ncrterm;
  m_NumCrossTerms = static_cast<int>(ncrterm);
  m_CrossTerms.resize(ncrterm);
  for (unsigned long long int i = 0; i < nlineCrterm; i++) {
    line.clear();
    apo::get_line(line, pos, fileData);
    tokens.clear();
    tokens = apo::split(line);
    m_CrossTerms[i].iatom1 = std::stoi(tokens[0]);
    m_CrossTerms[i].jatom1 = std::stoi(tokens[1]);
    m_CrossTerms[i].katom1 = std::stoi(tokens[2]);
    m_CrossTerms[i].latom1 = std::stoi(tokens[3]);
    m_CrossTerms[i].iatom2 = std::stoi(tokens[4]);
    m_CrossTerms[i].jatom2 = std::stoi(tokens[5]);
    m_CrossTerms[i].katom2 = std::stoi(tokens[6]);
    m_CrossTerms[i].latom2 = std::stoi(tokens[7]);
  }

  return;
}
