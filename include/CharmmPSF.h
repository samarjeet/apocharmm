// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "CudaContainer.h"
#include <set>
#include <string>
#include <vector>

struct Bond {
  int iatom, jatom;
};

struct Angle {
  int iatom, jatom, katom;
};

struct Dihedral {
  int iatom, jatom, katom, latom;
};

struct CrossTerm {
  int iatom1, jatom1, katom1, latom1;
  int iatom2, jatom2, katom2, latom2;
};

struct InclusionExclusion {
  std::vector<int> sizes;
  std::vector<int> in14_ex14;

  InclusionExclusion(const std::vector<int> &_sizes,
                     const std::vector<int> &_in14_ex14)
      : sizes(_sizes), in14_ex14(_in14_ex14) {}
};

class CharmmPSF {
public:
  /**
   * @brief Base constructor
   */
  CharmmPSF(void);

  /**
   * @brief Construct a CharmmPSF object from a CHARMM formatted PSF.
   */
  CharmmPSF(const std::string &fileName);

  /**
   * @brief Copy constructor
   */
  CharmmPSF(const CharmmPSF &other);

  /**
   * @brief Move constructor
   */
  CharmmPSF(const CharmmPSF &&other);

public:
  void setNumAtoms(const int numAtoms);
  void setAtomCharges(const std::vector<double> &charges);

public:
  int getNumAtoms(void) const;
  int getNumBonds(void) const;
  int getNumAngles(void) const;
  int getNumDihedrals(void) const;
  int getNumImpropers(void) const;
  int getNumCrossTerms(void) const;

  const std::vector<double> &getMasses(void) const;
  const std::vector<double> &getCharges(void) const;
  const std::vector<std::string> &getAtomNames(void) const;
  const std::vector<std::string> &getAtomTypes(void) const;
  const std::vector<Bond> &getBonds(void) const;
  const std::vector<Angle> &getAngles(void) const;
  const std::vector<Dihedral> &getDihedrals(void) const;
  const std::vector<Dihedral> &getImpropers(void) const;
  const std::vector<CrossTerm> &getCrossTerms(void) const;
  const std::vector<std::set<int>> &getConnected12(void) const;
  const std::vector<std::set<int>> &getConnected13(void) const;
  const std::vector<std::set<int>> &getConnected14(void) const;
  const std::vector<int> &getIblo14(void) const;
  const std::vector<int> &getInb14(void) const;
  const CudaContainer<int4> &getWaterMolecules(void) const;
  const CudaContainer<int2> &getResidues(void) const;
  const CudaContainer<int2> &getGroups(void) const;
  const std::string &getFileName(void) const;

  std::vector<double> &getMasses(void);
  std::vector<double> &getCharges(void);
  std::vector<std::string> &getAtomNames(void);
  std::vector<std::string> &getAtomTypes(void);
  std::vector<Bond> &getBonds(void);
  std::vector<Angle> &getAngles(void);
  std::vector<Dihedral> &getDihedrals(void);
  std::vector<Dihedral> &getImpropers(void);
  std::vector<CrossTerm> &getCrossTerms(void);
  std::vector<std::set<int>> &getConnected12(void);
  std::vector<std::set<int>> &getConnected13(void);
  std::vector<std::set<int>> &getConnected14(void);
  std::vector<int> &getIblo14(void);
  std::vector<int> &getInb14(void);
  CudaContainer<int4> &getWaterMolecules(void);
  CudaContainer<int2> &getResidues(void);
  CudaContainer<int2> &getGroups(void);
  std::string &getFileName(void);

  double getTotalMass(void) const;
  InclusionExclusion getInclusionExclusionLists(void) const;

private:
  void initializeWaterMolecules(void);
  void createConnectedComponents(void);
  void buildTopologicalExclusions(void);
  void readCharmmPSF(const std::string &fileName);

private:
  int m_NumAtoms;
  std::vector<double> m_Masses;
  std::vector<double> m_Charges;
  std::vector<std::string> m_AtomNames;
  std::vector<std::string> m_AtomTypes;

  int m_NumBonds;
  std::vector<Bond> m_Bonds;

  int m_NumAngles;
  std::vector<Angle> m_Angles;

  int m_NumDihedrals;
  std::vector<Dihedral> m_Dihedrals;

  int m_NumImpropers;
  std::vector<Dihedral> m_Impropers;

  int m_NumCrossTerms;
  std::vector<CrossTerm> m_CrossTerms;

  std::vector<std::set<int>> m_Connected12;
  std::vector<std::set<int>> m_Connected13;
  std::vector<std::set<int>> m_Connected14;

  std::vector<int> m_Iblo14;
  std::vector<int> m_Inb14;

  CudaContainer<int4> m_WaterMolecules;
  CudaContainer<int2> m_Residues;
  CudaContainer<int2> m_Groups;

  std::string m_FileName;
};
