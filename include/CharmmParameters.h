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

/* This class contains CHARMM parameters
 * It is filled up by reading the CHARMM .prm file
 *
 */
#pragma once
#include "CharmmPSF.h"
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct BondKey {
public:
  BondKey(std::string a1, std::string a2) : atom1(a1), atom2(a2) {}
  std::string atom1, atom2;
  friend bool operator<(const BondKey &first, const BondKey &second) {
    if (second.atom1 != first.atom1)
      return second.atom1 < first.atom1;
    else
      return second.atom2 < first.atom2;
  }
  bool operator==(const BondKey &other) const {
    return this->atom2 == other.atom2 && this->atom1 == other.atom1;
  }
  friend std::ostream &operator<<(std::ostream &output, const BondKey &key) {
    output << key.atom1 << " " << key.atom2 << " " << " ";
    return output;
  }
};

class BondValues {
public:
  BondValues() = default;
  BondValues(float kb, float b0) : kb(kb), b0(b0) {}
  BondValues(const BondValues &bv) = default;
  float kb, b0;
  friend std::ostream &operator<<(std::ostream &output, const BondValues &bv) {
    output << "(" << bv.kb << "," << bv.b0 << ")\n";
    return output;
  }
};

struct AngleKey {
  AngleKey(std::string a1, std::string a2, std::string a3)
      : atom1(a1), atom2(a2), atom3(a3) {}
  std::string atom1, atom2, atom3;
  friend bool operator<(const AngleKey &first, const AngleKey &second) {
    if (second.atom1 != first.atom1)
      return second.atom1 < first.atom1;
    else if (second.atom2 != first.atom2)
      return second.atom2 < first.atom2;
    else
      return second.atom3 < first.atom3;
  }

  bool operator==(const AngleKey &other) const {
    return this->atom1 == other.atom1 && this->atom2 == other.atom2 &&
           this->atom3 == other.atom3;
  }
  friend std::ostream &operator<<(std::ostream &output, const AngleKey &key) {
    output << key.atom1 << " " << key.atom2 << " " << key.atom3 << " ";
    return output;
  }
};

struct AngleValues {
public:
  AngleValues() = default;
  AngleValues(float kTheta, float theta0) : kTheta(kTheta), theta0(theta0) {}
  AngleValues(const AngleValues &av) = default;
  float kTheta, theta0;
};

struct DihedralKey {
  DihedralKey(std::string a1, std::string a2, std::string a3, std::string a4)
      : atom1(a1), atom2(a2), atom3(a3), atom4(a4) {}
  std::string atom1, atom2, atom3, atom4;
  friend bool operator<(const DihedralKey &first, const DihedralKey &second) {
    if (second.atom1 != first.atom1)
      return second.atom1 < first.atom1;
    else if (second.atom2 != first.atom2)
      return second.atom2 < first.atom2;
    else if (second.atom3 != first.atom3)
      return second.atom3 < first.atom3;
    else
      return second.atom4 < first.atom4;
  }
  bool operator==(const DihedralKey &other) const {
    return this->atom1 == other.atom1 && this->atom2 == other.atom2 &&
           this->atom3 == other.atom3 && this->atom4 == other.atom4;
  }
  friend std::ostream &operator<<(std::ostream &output, const DihedralKey &dk) {
    output << dk.atom1 << " " << dk.atom2 << " " << dk.atom3 << " " << dk.atom4
           << " ";
    return output;
  }
};

struct DihedralValues {
public:
  DihedralValues() = default;
  DihedralValues(float kChi, int n, float delta)
      : kChi(kChi), n(n), delta(delta) {}
  float kChi, delta;
  int n;
  friend std::ostream &operator<<(std::ostream &output,
                                  const DihedralValues &dv) {
    output << "(kChi: " << dv.kChi << ", delta : " << dv.delta
           << ", n : " << dv.n << ")\n";
    return output;
  }
};

struct ImDihedralValues {
public:
  ImDihedralValues() = default;
  ImDihedralValues(float kpsi, float psi0) : kpsi(kpsi), psi0(psi0) {}
  float kpsi, psi0;
  friend std::ostream &operator<<(std::ostream &output,
                                  const ImDihedralValues &idv) {
    output << "(kpsi : " << idv.kpsi << ", psi0 : " << idv.psi0 << ")\n";
    return output;
  }
};

struct CmapKey {
  // CmapKey(std::string a1i, std::string a1j, std::string a1k, std::string a1l,
  //         std::string a2i, std::string a2j, std::string a2k, std::string a2l)
  //     : atom1i(a1i), atom1j(a1j), atom1k(a1k), atom1l(a1l), atom2i(a2i),
  //       atom2j(a2j), atom2k(a2k), atom2l(a2l) {}
  // std::string atom1i, atom1j, atom1k, atom1l, atom2i, atom2j, atom2k, atom2l;
  CmapKey(DihedralKey d1, DihedralKey d2) : dih1(d1), dih2(d2) {}
  DihedralKey dih1, dih2;
};

class VdwParameters {
public:
  VdwParameters() = default;
  VdwParameters(double e, double r) : epsilon(e), rmin_2(r) {}

  double epsilon, rmin_2;
  friend std::ostream &operator<<(std::ostream &output,
                                  const VdwParameters &vdw) {
    output << "(" << vdw.epsilon << "," << vdw.rmin_2 << ")\n";
    return output;
  }
};

class NBFixParameters {
public:
  std::string atom1, atom2;
  double emin, rmin, emin14, rmin14;
  // NBFixParameters(std::string a1, std::string a2, double e, double r,
  //                 double e14, double r14)
  //     : atom1(a1), atom2(a2), emin(e), rmin(r), emin14(e14), rmin14(r14) {}
  // // copy constructor
  // NBFixParameters(const NBFixParameters &nbfix) = default;
  // NBFixParameters(const NBFixParameters &nbfix)
  //     : atom1(nbfix.atom1), atom2(nbfix.atom2), emin(nbfix.emin),
  //       rmin(nbfix.rmin), emin14(nbfix.emin14), rmin14(nbfix.rmin14) {}

  // // assignment operator
  // NBFixParameters &operator=(const NBFixParameters &nbfix) {
  //   atom1 = nbfix.atom1;
  //   atom2 = nbfix.atom2;
  //   emin = nbfix.emin;
  //   rmin = nbfix.rmin;
  //   emin14 = nbfix.emin14;
  //   rmin14 = nbfix.rmin14;
  //   return *this;
  // }

  friend std::ostream &operator<<(std::ostream &output,
                                  const NBFixParameters &nbfix) {
    output << "(" << nbfix.atom1 << "," << nbfix.atom2 << "," << nbfix.emin
           << "," << nbfix.rmin << "," << nbfix.emin14 << "," << nbfix.rmin14
           << ")\n";
    return output;
  }
};

/** @brief Contains bonded interactions parameters and list
 * @todo improve doc
 *
 * Contains:
 * - paramsSize: vector of int, containing the number of types of bond,
 * Urey-Bradley, angle, dihedral and (in the future ?) CMAP
 * - paramsVal: vector of vectors, containing the parameters for each
 * interaction
 * - listsSize: vector of int, contains the number of interactions for each type
 * - listVal: vector of vectors of int, contains the atom indices for each
 * interaction
 */
struct BondedParamsAndLists {
  std::vector<int> paramsSize;
  std::vector<std::vector<float>> paramsVal;

  std::vector<int> listsSize;
  std::vector<std::vector<int>> listVal;

  BondedParamsAndLists(std::vector<int> pSize,
                       std::vector<std::vector<float>> pVal,
                       std::vector<int> lSize,
                       std::vector<std::vector<int>> lVal)
      : paramsSize(pSize), paramsVal(pVal), listsSize(lSize), listVal(lVal) {}
};

/** @brief Contains van der Waals interactions parameters and lists */
struct VdwParamsAndTypes {
  std::vector<float> vdwParams, vdw14Params;
  std::vector<int> vdwTypes, vdw14Types;

  VdwParamsAndTypes(std::vector<float> vdwParams,
                    std::vector<float> vdw14Params, std::vector<int> vdwTypes,
                    std::vector<int> vdw14Types)
      : vdwParams(vdwParams), vdw14Params(vdw14Params), vdwTypes(vdwTypes),
        vdw14Types(vdw14Types) {}
};

/**
 * @brief Set of CHARMM parameters
 *
 * CHARMM parameters. Initialized from a filename (.prm, .str) or a list of
 * file names. Does not contain charges.
 */
class CharmmParameters {
public:
  /** @brief Constructor. Uses a single .prm or .str input file.
   * @param fileName String, .prm or .str file
   */
  CharmmParameters(const std::string &fileName);
  /** @brief Constructor. Uses a list of .prm and .str input files.
   * @param fileNames List of filename strings
   */
  CharmmParameters(const std::vector<std::string> &fileNames);

  /** @brief Returns a map of the bond interactions. Key is a BondKey object
   * (couple of atom names), value is a BondValues object (kb and b0) */
  std::map<BondKey, BondValues> getBonds();
  std::map<AngleKey, AngleValues> getAngles();
  std::map<DihedralKey, std::vector<DihedralValues>> getDihedrals();
  std::map<DihedralKey, ImDihedralValues> getImpropers();
  std::map<AngleKey, BondValues> getUreyBradleys();

  std::map<BondKey, BondValues> getBondParams() { return bondParams; }

  std::map<std::string, VdwParameters> getVdwParameters() { return vdwParams; }
  std::map<std::string, VdwParameters> getVdw14Parameters() {
    return vdw14Params;
  }

  // BondedParamsAndLists getBondedParamsAndLists(const
  // std::unique_ptr<CharmmPSF> & psf);

  /** @brief Returns bonded interactions list and parameters as
   * BondedParamsAndLists object
   *
   */
  BondedParamsAndLists
  getBondedParamsAndLists(const std::shared_ptr<CharmmPSF> &psf);
  // VdwParamsAndTypes getVdwParamsAndTypes(std::unique_ptr<CharmmPSF> & psf);

  /** @brief Returns vdW interaction lists and parameters as VdwParamsAndTypes
   * object */
  VdwParamsAndTypes getVdwParamsAndTypes(std::shared_ptr<CharmmPSF> &psf);

  /**
   * @brief Read prm file
   *
   * @param[in] fileName Input file name
   */
  void readCharmmParameterFile(std::string fileName);

  /** @brief Get original names of prm files imported */
  std::vector<std::string> getOriginalPrmFileNames() { return prmFileNames; }

private:
  std::map<BondKey, BondValues> bondParams;
  // std::map<BondKey, BondValues> ureybParams;
  std::map<AngleKey, BondValues> ureybParams;
  std::map<AngleKey, AngleValues> angleParams;
  std::map<DihedralKey, std::vector<DihedralValues>> dihedralParams;
  std::map<DihedralKey, ImDihedralValues> improperParams;

  std::map<std::tuple<std::string, std::string>, NBFixParameters>
      // std::map<BondKey, NBFixParameters>
      nbfixParams; // will be filled in the  vdw(14)Params

  std::map<std::string, VdwParameters> vdwParams;
  std::map<std::string, VdwParameters> vdw14Params;
  /** @brief original prm file names */
  std::vector<std::string> prmFileNames;
};
