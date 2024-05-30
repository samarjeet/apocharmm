// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include "CudaContainer.h"
#include <set>
#include <string>
#include <vector>

// forward decalation of CharmmResidueTopology
class CharmmResidueTopology;

class Bond {
public:
  int atom1, atom2;
};

class Angle {
public:
  int atom1, atom2, atom3;
};

class Dihedral {
public:
  int atom1, atom2, atom3, atom4;
};

class CrossTerm {
public:
  int atomi1, atomj1, atomk1, atoml1;
  int atomi2, atomj2, atomk2, atoml2;
};

struct InclusionExclusion {
  std::vector<int> sizes;
  std::vector<int> in14_ex14;

  InclusionExclusion(std::vector<int> s, std::vector<int> list)
      : sizes(s), in14_ex14(list) {}
};

/**
 * @brief CHARMM topology file (PSF)
 *
 * CHARMM topology file (PSF) representation. Contains N-sized vector of all
 * the following entries:
 * - Bond bonds (getBonds())
 * - Angle angles (getAngles())
 * - Dihedral dihedrals (getDihedrals())
 * - Dihedral improper dihedrals (getImpropers())
 * - charges (see getAtomCharges())
 *
 * Used to generate ForceManager objects.
 *
 * @note in this documentation page, the system number of atoms is referred to
 * as N.
 *
 */
class CharmmPSF {
public:
  /** @brief Base constructor */
  CharmmPSF();
  /** @brief Constructor from a .psf file name */
  CharmmPSF(const std::string &fileName);

  /** @brief Basic copy-constructor
   * @todo unittest this: same attribute vlaues for the copy, acting on copy
   * does not change the original
   */

  CharmmPSF(const CharmmPSF &psfIn);

  /**
   * @brief Increase the mass of the hydrogen atoms to a given amount
   * Note that function is not 'repartitioning' hydrogen mass to the neighbor
   * atoms, but rather increasing the mass of the hydrogen atoms to a given
   * mass
   * @param _newHyrogenMass (in a.m.u.)
   */
  void setHydrogenMass(double _newHyrogenMass);

  int getNumAtoms() { return numAtoms; }
  int getNumBonds() { return numBonds; }
  int getNumAngles() { return numAngles; }
  int getNumDihedrals() { return numDihedrals; }
  int getNumImpropers() { return numImpropers; }
  int getNumCrossTerms() { return numCrossTerms; }

  /** @brief Returns a vector containing all Bond objects */
  std::vector<Bond> getBonds() { return bonds; }
  /** @brief Returns a vector containing all Angle objects */
  std::vector<Angle> getAngles() { return angles; }
  /** @brief Returns a vector containing all Dihedral objects */
  std::vector<Dihedral> getDihedrals() { return dihedrals; }
  /** @brief Returns a vector containing all improper dihedral objects */
  std::vector<Dihedral> getImpropers() { return impropers; }

  std::vector<CrossTerm> getCrossTerms() { return crossTerms; }

  /** @brief Returns a N-sized vector containing all atomic masses */
  std::vector<double> getAtomMasses() { return masses; }
  /** @brief Returns a N-sized vector containing all atomic charges */
  std::vector<double> getAtomCharges() { return charges; }

  /** @brief Returns a N-sized string-vector containing all atomic names */
  std::vector<std::string> getAtomNames() { return atomNames; }
  /** @brief Returns a N-sized string-vector containing all atomic types */
  std::vector<std::string> getAtomTypes() { return atomTypes; }

  std::string getAtomType(int index) const { return atomTypes[index]; }

  std::vector<int> getInb14() { return inb14; }
  std::vector<int> getIblo14() { return iblo14; }

  /** @brief Computes number of water molecules.
   *
   * Searches for a sequence of one "OT" type atom followed by two "HT" type
   * atoms.
   */
  CudaContainer<int4> getWaterMolecules();
  CudaContainer<int2> getResidues();
  CudaContainer<int2> getGroups();

  /** @brief Computes number of degrees of freedom (3N-6)
   *
   * Handling constraints is not done here. Should be done by the
   * ForceManager.
   */

  int getDegreesOfFreedom();

  int getMass();

  InclusionExclusion getInclusionExclusionLists();

  /** @brief Set charges value */
  void setAtomCharges(std::vector<double> chargesIn);

  /**
   * @brief Generates a PSF structure for the sequence of residues
   * @param rtf : should contain the residues in the `@param` sequence
   * @param sequence : each element should have an entry in the `@param` rtf
   * @param segment :
   */
  void generate(const CharmmResidueTopology &rtf,
                const std::vector<std::string> &sequence, std::string segmnet);

  /**
   * @brief Appends the sequence of residues to the PSF structure using the
   * RTF object
   * @param same as generate
   * If @param segment is same already present in the psf, the residues will
   * be appeneded to it. Otherwise, a segment in the psf will be created
   */
  void append(const CharmmResidueTopology &rtf,
              const std::vector<std::string> &sequence, std::string segmnet);

  std::string getOriginalPSFFileName() { return originalPSFFileName; }

private:
  int numAtoms;
  int numBonds;
  int numAngles;
  int numDihedrals;
  int numImpropers;
  int numCrossTerms;
  /** @brief N-sized vector containing the atomic masses */
  std::vector<double> masses;

  /** @brief N-sized vector containing the atomic charges */
  std::vector<double> charges;
  /** @brief N-sized string-vector of atom names */
  std::vector<std::string> atomNames;
  /** @brief N-sized string-vector of atom types */
  std::vector<std::string> atomTypes;

  // std::vector<std::pair<int, int>> bonds;
  /** @brief vector containing all Bond objects */
  std::vector<Bond> bonds;
  std::vector<Angle> angles;
  std::vector<Dihedral> dihedrals;
  std::vector<Dihedral> impropers;
  std::vector<CrossTerm> crossTerms;

  CudaContainer<int4> waterMolecules;
  CudaContainer<int2> residues;
  CudaContainer<int2> groups;

  std::vector<std::set<int>> connected12, connected13, connected14;
  std::vector<int> iblo14, inb14;

  void readCharmmPSFFile(std::string fileName);
  void buildTopologicalExclusions();
  void createConnectedComponents();

  /** @brief Keep the original file name to be loggable later. */
  std::string originalPSFFileName;
};
