// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmParameters.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "cpp_utils.h"

CharmmParameters::CharmmParameters(const std::string &fileName) {
  prmFileNames.push_back(fileName);
  readCharmmParameterFile(fileName);
}

CharmmParameters::CharmmParameters(const std::vector<std::string> &fileNames) {
  for (auto const &fileName : fileNames) {
    prmFileNames.push_back(fileName);
    readCharmmParameterFile(fileName);
  }
}
/*
std::string ltrim(const std::string &str) {
  size_t start = str.find_first_not_of(" \n\t");
  return (start == std::string::npos) ? "" : str.substr(start);
}

static std::string removeComments(std::string line) {
  line = trim(line);
  auto npos = line.find_first_of('!');
  return trim(line.substr(0, npos));
}

std::vector<std::string> split(const std::string &str) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  tokenStream >> token;
  while (token.size() && tokenStream) {
    tokens.push_back(token);
    tokenStream >> token;
  }
  return tokens;
}
*/
void CharmmParameters::readCharmmParameterFile(std::string fileName) {
  enum State {
    NONE,
    ATOMS,
    BONDS,
    ANGLES,
    DIHEDRALS,
    IMPROPERS,
    CMAP,
    NONBONDED,
    NBFIX,
    HBOND
  };
  State state = NONE;
  std::vector<std::string> tokens;

  enum FileType { PAR, TOPPAR };
  FileType fileType;
  int pos = fileName.find_last_of('/');
  if (pos != std::string::npos) {
    int topparPos = fileName.find("toppar", pos);
    if (topparPos != std::string::npos)
      fileType = TOPPAR;
    else
      fileType = PAR;
  }

  std::ifstream prmFile(fileName);
  std::string line;
  if (!prmFile.is_open()) {
    // std::cerr << "ERROR: Cannot open the file " << fileName << "\nExiting\n";
    throw std::invalid_argument("ERROR: Cannot open the file " + fileName +
                                "\nExiting\n");
    exit(0);
  }

  // If the file is toppar, skip the rtf portion
  // if (fileName.find_first_of("toppar") == 0 && fileName.find_last_of(".str")
  // != std::string::npos ) {

  if (fileType == TOPPAR) {

    while (!prmFile.eof()) {
      std::getline(prmFile, line);
      // std::cout << "toppar --" << line << "\n";
      line = removeComments(line);
      line = trim(line);
      std::transform(line.begin(), line.end(), line.begin(),
                     [](unsigned char c) { return std::toupper(c); });
      if (line.find_first_of('*') == 0 || line.find_first_of('!') == 0 ||
          line.size() == 0) {
        // Skip the line
      } else {
        if (line.find("READ") != std::string::npos &&
            line.find("PARA") != std::string::npos) {
          break;
        }
      }
    }
  }

  const float pi_180 = std::acos(-1) / 180.0;
  while (!prmFile.eof()) {
    std::getline(prmFile, line);
    // line = ltrim(line);
    line = removeComments(line);
    line = trim(line);
    // line = std::toupper(line);
    std::transform(line.begin(), line.end(), line.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    // std::cout << line << "\n";
    if (line.find_first_of('*') == 0 || line.find_first_of('!') == 0 ||
        line.size() == 0) {
      // Skip the line
    } else {
      if (line.find("ATOMS") == 0)
        state = ATOMS;
      if (line.find("BONDS") == 0)
        state = BONDS;
      if (line.find("ANGLES") == 0)
        state = ANGLES;
      if (line.find("DIHEDRALS") == 0)
        state = DIHEDRALS;
      if (line.find("IMPR") == 0)
        state = IMPROPERS;
      if (line.find("CMAP") == 0)
        state = CMAP;
      if (line.find("NONBONDED") == 0)
        state = NONBONDED;
      if (line.find("END") == 0)
        state = NONE;
      if (line.find("HBOND") == 0)
        state = HBOND;
      if (line.find("NBFIX") == 0)
        state = NBFIX;

      if (state == BONDS) {
        tokens = split(line);
        if (tokens.size() >= 4) {
          if (tokens[0] > tokens[1])
            std::swap(tokens[0], tokens[1]);
          bondParams.insert(
              {BondKey(tokens[0], tokens[1]),
               BondValues(std::stof(tokens[2]), std::stof(tokens[3]))});
        }
      }
      if (state == ANGLES) {
        tokens = split(line);
        // std::cout << "Tokens size : " << tokens.size() << "\n";
        if (tokens.size() >= 5) {
          if (tokens[0] > tokens[2])
            std::swap(tokens[0], tokens[2]);
          angleParams.insert({AngleKey(tokens[0], tokens[1], tokens[2]),
                              AngleValues(std::stof(tokens[3]),
                                          pi_180 * std::stof(tokens[4]))});
          /*
          try {
            // std::cout << tokens[5] << "\n";
            float kub = std::stof(tokens[5]);
            float s0 = std::stof(tokens[6]);
            ureybParams.insert({AngleKey(tokens[0], tokens[1], tokens[2]),
                                BondValues(kub, s0)});
            // if (tokens[0] == "CT1" && tokens[2]=="CT2") std::cout << line <<
            // "\n";
          } catch (const std::exception &e) {
            ureybParams.insert({AngleKey(tokens[0], tokens[1], tokens[2]),
                                BondValues(0.0, 0.0)});
            // std::cout << line << " "  << e.what() << "\n";
          }
          */
          //
          if (tokens.size() > 5) {
            // std::cout << tokens[5] << "\n";
            float kub = std::stof(tokens[5]);
            float s0 = std::stof(tokens[6]);
            ureybParams.insert({AngleKey(tokens[0], tokens[1], tokens[2]),
                                BondValues(kub, s0)});
          } else {
            ureybParams.insert({AngleKey(tokens[0], tokens[1], tokens[2]),
                                BondValues(0.0, 0.0)});
          }

          //
        }
      }

      if (state == DIHEDRALS) {
        tokens = split(line);
        if (tokens.size() >= 7) {
          if (tokens[0] > tokens[3]) {
            std::swap(tokens[0], tokens[3]);
            std::swap(tokens[2], tokens[1]);
          } else if ((tokens[0] == tokens[3]) && (tokens[1] > tokens[2]))
            std::swap(tokens[2], tokens[1]);

          auto key = DihedralKey(tokens[0], tokens[1], tokens[2], tokens[3]);
          auto elem = DihedralValues(std::stof(tokens[4]), std::stoi(tokens[5]),
                                     std::stof(tokens[6]));
          dihedralParams[key].push_back(elem);
          // std::cout << "new size " << dihedralParams[key].size() << "\n";
        }
      }

      if (state == IMPROPERS) {
        tokens = split(line);
        if (tokens.size() >= 7) {
          if (tokens[0] > tokens[3]) {
            std::swap(tokens[0], tokens[3]);
            std::swap(tokens[2], tokens[1]);
          } else if ((tokens[0] == tokens[3]) && (tokens[1] > tokens[2]))
            std::swap(tokens[2], tokens[1]);
          auto success = improperParams.insert(
              {DihedralKey(tokens[0], tokens[1], tokens[2], tokens[3]),
               ImDihedralValues(std::stof(tokens[4]), std::stof(tokens[6]))});
        }
      }

      if (state == NONBONDED) {
        tokens = split(line);
        if (tokens[0] == "NONBONDED") {
          while (*(tokens.end() - 1) == "-") {
            std::getline(prmFile, line);
            line = ltrim(line);
            auto tokens1 = split(line);
            tokens.insert(tokens.end(), tokens1.begin(), tokens1.end());
          }
        } else {
          line = removeComments(line);
          tokens = split(line);
          // if (tokens[0] == "HGA2")
          //   std::cout << line << " " << tokens.size() << "\n";
          //  std::cout << line << " " << tokens.size() << "\n";
          if (tokens.size() == 4) {
            if (vdwParams.find(tokens[0]) == vdwParams.end()) {
              vdwParams.insert(
                  {tokens[0],
                   VdwParameters(std::stod(tokens[2]), std::stod(tokens[3]))});
            } else {
              std::cerr << "Duplicate entry for " << tokens[0] << "\n";
              vdwParams[tokens[0]] =
                  VdwParameters(std::stod(tokens[2]), std::stod(tokens[3]));
              // throw std::invalid_argument("Duplicate entry for " + tokens[0]
              // +
              //                             "\n");
              // exit(0);
            }
            /*vdwParams.insert({tokens[0], VdwParameters(std::stod(tokens[2]),
                                                       std::stod(tokens[3]))});*/
          } else if (tokens.size() == 7) {
            if (vdwParams.find(tokens[0]) == vdwParams.end()) {
              vdwParams.insert(
                  {tokens[0],
                   VdwParameters(std::stod(tokens[2]), std::stod(tokens[3]))});
              vdw14Params.insert(
                  {tokens[0],
                   VdwParameters(std::stod(tokens[5]), std::stod(tokens[6]))});
            } else {
              std::cerr << "Duplicate entry for " << tokens[0] << "\n";
              vdwParams[tokens[0]] =
                  VdwParameters(std::stod(tokens[2]), std::stod(tokens[3]));
              vdw14Params[tokens[0]] =
                  VdwParameters(std::stod(tokens[5]), std::stod(tokens[6]));
              // throw std::invalid_argument("Duplicate entry for " + tokens[0]
              // +
              //                             "\n");
              // exit(0);
            }
            /*vdwParams.insert({tokens[0], VdwParameters(std::stod(tokens[2]),
                                                       std::stod(tokens[3]))});
            vdw14Params.insert(
                {tokens[0],
                 VdwParameters(std::stod(tokens[5]), std::stod(tokens[6]))});
            */

          } else {
            // std::cerr << "Extra tokens in line " << line << "\n";
            throw std::invalid_argument("Extra tokens in line " + line + "\n");
            exit(0);
          }
        }

      } // state : NONBONDED

      if (state == CMAP) {
        // std::cout << line << std::endl;
      }

      if (state == NBFIX) {
        // std::cout << "NBFIX : " << line << "\t";
        tokens = split(line);
        if (tokens.size() >= 4) {
          if (tokens[0] > tokens[1])
            std::swap(tokens[0], tokens[1]);
          double emin = std::abs(std::stod(tokens[2]));
          double rmin = std::stod(tokens[3]);
          double emin14 = emin;
          double rmin14 = rmin;
          if (tokens.size() >= 5) {
            emin14 = std::stod(tokens[4]);
            rmin14 = std::stod(tokens[5]);
          }

          // std::cout << emin << " " << rmin << " " << emin14 << " " << rmin14
          //           << std::endl;
          NBFixParameters nbf{tokens[0], tokens[1], emin, rmin, emin14, rmin14};
          std::tuple<std::string, std::string> key{tokens[0], tokens[1]};
          // BondKey key{tokens[0], tokens[1]};
          //  nbfixParams[key] = nbf;
          nbfixParams.insert({key, nbf});
        }
      }
    }
  }
}

std::map<BondKey, BondValues> CharmmParameters::getBonds() {
  return bondParams;
}

std::map<AngleKey, AngleValues> CharmmParameters::getAngles() {
  return angleParams;
}

std::map<AngleKey, BondValues> CharmmParameters::getUreyBradleys() {
  return ureybParams;
}

std::map<DihedralKey, std::vector<DihedralValues>>
CharmmParameters::getDihedrals() {
  return dihedralParams;
}

std::map<DihedralKey, ImDihedralValues> CharmmParameters::getImpropers() {
  return improperParams;
}

BondedParamsAndLists CharmmParameters::getBondedParamsAndLists(
    const std::shared_ptr<CharmmPSF> &psf) {
  std::vector<int> paramsSize;
  std::vector<std::vector<float>> paramsVal;

  std::vector<int> listsSize;
  std::vector<std::vector<int>> listVal;

  auto atomTypes = psf->getAtomTypes();
  auto atomNames = psf->getAtomNames();
  auto bonds = psf->getBonds();
  auto angles = psf->getAngles();
  auto dihedrals = psf->getDihedrals();
  auto impropers = psf->getImpropers();
  auto cmaps = psf->getCrossTerms();

  std::vector<BondKey> bondKeysPresent;
  std::vector<AngleKey> ureybKeysPresent;
  std::vector<AngleKey> angleKeysPresent;
  std::vector<DihedralKey> dihedralKeysPresent;
  std::vector<DihedralKey> improperKeysPresent;

  for (int bond = 0; bond < psf->getNumBonds(); ++bond) {
    std::string atom1 = atomTypes[bonds[bond].atom1];
    std::string atom2 = atomTypes[bonds[bond].atom2];
    if (atom1 > atom2)
      std::swap(atom1, atom2);
    auto key = BondKey(atom1, atom2);

    if (bondParams.count(key)) {
      auto findResult =
          std::find(bondKeysPresent.begin(), bondKeysPresent.end(), key);
      if (findResult == std::end(bondKeysPresent)) {
        bondKeysPresent.push_back(key);
        auto value = bondParams[key];
        paramsVal.push_back({value.b0, value.kb});
      }
      findResult =
          std::find(bondKeysPresent.begin(), bondKeysPresent.end(), key);
      int bondType = findResult - std::begin(bondKeysPresent);
      listVal.push_back({bonds[bond].atom1, bonds[bond].atom2, bondType, 13});

    } else {
      std::stringstream tmpexc;
      tmpexc << "bond not found " << bond << " " << key << " "
             << bonds[bond].atom1 << " " << bonds[bond].atom2 << "\n";
      throw std::invalid_argument(tmpexc.str());
    }
  }
  paramsSize.push_back(bondKeysPresent.size());
  listsSize.push_back(listVal.size());

  /*
  In case of Urey-Bradley, most of the angles will not have a energy
  contribution. So, we need to push in only the terms which have non-zero k
  value
  */
  int ureybCount = 0;
  for (int angle = 0; angle < psf->getNumAngles(); ++angle) {
    std::string atom1 = atomTypes[angles[angle].atom1];
    std::string atom2 = atomTypes[angles[angle].atom2];
    std::string atom3 = atomTypes[angles[angle].atom3];
    if (atom1 > atom3)
      std::swap(atom1, atom3);

    auto key = AngleKey(atom1, atom2, atom3);
    if (ureybParams.count(key)) {
      if (std::abs(ureybParams[key].kb - 0.0) <= 0.01)
        continue;
      ureybCount++;
      // ureybKeysPresent.insert(BondKey(atom1,atom3));
      auto findResult =
          std::find(ureybKeysPresent.begin(), ureybKeysPresent.end(), key);
      if (findResult == ureybKeysPresent.end()) {
        ureybKeysPresent.push_back(key);
        auto value = ureybParams[key];
        paramsVal.push_back({value.b0, value.kb});
      }
      findResult =
          std::find(ureybKeysPresent.begin(), ureybKeysPresent.end(), key);
      int ureybType = findResult - ureybKeysPresent.begin();
      listVal.push_back(
          {angles[angle].atom1, angles[angle].atom3, ureybType, 13});
    } else {
      std::stringstream tmpexc;
      tmpexc << "Ureyb not found " << angle << " " << key << "\n";
      throw std::invalid_argument(tmpexc.str());
    }
  }
  paramsSize.push_back(ureybKeysPresent.size());
  listsSize.push_back(listVal.size() - listsSize[0]);

  for (int angle = 0; angle < psf->getNumAngles(); ++angle) {
    std::string atom1 = atomTypes[angles[angle].atom1];
    std::string atom2 = atomTypes[angles[angle].atom2];
    std::string atom3 = atomTypes[angles[angle].atom3];
    if (atom1 > atom3)
      std::swap(atom1, atom3);

    auto key = AngleKey(atom1, atom2, atom3);
    if (angleParams.count(key)) {
      auto findResult =
          std::find(angleKeysPresent.begin(), angleKeysPresent.end(), key);
      if (findResult == angleKeysPresent.end()) {
        angleKeysPresent.push_back(key);
        auto value = angleParams[key];
        paramsVal.push_back({value.theta0, value.kTheta});
      }
      findResult =
          std::find(angleKeysPresent.begin(), angleKeysPresent.end(), key);
      int angleType = findResult - angleKeysPresent.begin();
      listVal.push_back({angles[angle].atom1, angles[angle].atom2,
                         angles[angle].atom3, angleType, 13, 13});

    } else {
      std::stringstream tmpexc;
      tmpexc << "Angle not found " << angle << " " << key << "\n";
      throw std::invalid_argument(tmpexc.str());
    }
  }
  paramsSize.push_back(angleKeysPresent.size());
  listsSize.push_back(listVal.size() - listsSize[0] - listsSize[1]);

  int dihedralParamsPresent = 0;
  int startParamDihedral = paramsVal.size();
  std::map<DihedralKey, int> indexOfKeyInParamsVal;
  const float pi_180 = std::acos(-1) / 180.0;
  for (int dihedral = 0; dihedral < psf->getNumDihedrals(); ++dihedral) {
    std::string atom1 = atomTypes[dihedrals[dihedral].atom1];
    std::string atom2 = atomTypes[dihedrals[dihedral].atom2];
    std::string atom3 = atomTypes[dihedrals[dihedral].atom3];
    std::string atom4 = atomTypes[dihedrals[dihedral].atom4];

    if (atom1 > atom4) {
      std::swap(atom1, atom4);
      std::swap(atom2, atom3);
    }
    if ((atom1 == atom4) && (atom2 > atom3)) {
      std::swap(atom2, atom3);
    }

    auto key = DihedralKey(atom1, atom2, atom3, atom4);
    if (dihedralParams.count(key)) {
      // dihedralKeysPresent.insert(DihedralKey(atom1, atom2, atom3, atom4));
      auto findResult = std::find(dihedralKeysPresent.begin(),
                                  dihedralKeysPresent.end(), key);
      if (findResult == dihedralKeysPresent.end()) {
        dihedralKeysPresent.push_back(key);
        // if (dihedralParams[key].size() > 1) std::cout << "size larger than
        // 1\n";
        for (int i = 0; i < dihedralParams[key].size(); ++i) {
          if (i > 0) {
            paramsVal[paramsVal.size() - 1][0] *= -1;
          } else {
            indexOfKeyInParamsVal[key] = paramsVal.size() - startParamDihedral;
          }
          auto value = dihedralParams[key][i];
          float cpsin = std::sin(value.delta * pi_180);
          float cpcos = std::cos(value.delta * pi_180);
          paramsVal.push_back({(float)value.n, value.kChi, cpsin, cpcos});
          dihedralParamsPresent++;
        }
        dihedralKeysPresent.push_back(key);
      }
      // findResult = std::find(dihedralKeysPresent.begin(),
      //                       dihedralKeysPresent.end(), key);
      // int dihedralType = findResult - dihedralKeysPresent.begin();
      int dihedralType = indexOfKeyInParamsVal[key];
      listVal.push_back({dihedrals[dihedral].atom1, dihedrals[dihedral].atom2,
                         dihedrals[dihedral].atom3, dihedrals[dihedral].atom4,
                         dihedralType, 13, 13, 13});
    } else {
      if (atom2 > atom3)
        std::swap(atom2, atom3);

      key = DihedralKey("X", atom2, atom3, "X");
      if (dihedralParams.count(key)) {
        // dihedralKeysPresent.insert(DihedralKey("X", atom2, atom3, "X"));
        auto findResult = std::find(dihedralKeysPresent.begin(),
                                    dihedralKeysPresent.end(), key);
        if (findResult == dihedralKeysPresent.end()) {
          if (dihedralParams[key].size() > 1)
            std::cout << "CharmmParameters: Dihedral key size larger than 1\n";
          for (int i = 0; i < dihedralParams[key].size(); ++i) {
            if (i > 0) {
              paramsVal[paramsVal.size() - 1][0] *= -1;
            } else {
              indexOfKeyInParamsVal[key] =
                  paramsVal.size() - startParamDihedral;
            }
            auto value = dihedralParams[key][i];
            float cpsin = std::sin(value.delta * pi_180);
            float cpcos = std::cos(value.delta * pi_180);
            paramsVal.push_back({(float)value.n, value.kChi, cpsin, cpcos});
            dihedralParamsPresent++;
          }
          dihedralKeysPresent.push_back(key);
        }
        // findResult = std::find(dihedralKeysPresent.begin(),
        //                       dihedralKeysPresent.end(), key);
        // int dihedralType = findResult - dihedralKeysPresent.begin();
        int dihedralType = indexOfKeyInParamsVal[key];
        listVal.push_back({dihedrals[dihedral].atom1, dihedrals[dihedral].atom2,
                           dihedrals[dihedral].atom3, dihedrals[dihedral].atom4,
                           dihedralType, 13, 13, 13});

      } else {
        std::stringstream tmpexc;
        tmpexc << "dihedral not found " << dihedral << " "
               << atomTypes[dihedrals[dihedral].atom1] << " "
               << atomTypes[dihedrals[dihedral].atom2] << " "
               << atomTypes[dihedrals[dihedral].atom3] << " "
               << atomTypes[dihedrals[dihedral].atom4] << "\t"
               << atomNames[dihedrals[dihedral].atom1] << " "
               << atomNames[dihedrals[dihedral].atom2] << " "
               << atomNames[dihedrals[dihedral].atom3] << " "
               << atomNames[dihedrals[dihedral].atom4] << "\n";
        throw std::invalid_argument(tmpexc.str());
      }
    }
  }
  paramsSize.push_back(dihedralParamsPresent);
  listsSize.push_back(listVal.size() - listsSize[0] - listsSize[1] -
                      listsSize[2]);

  // int startDihedral=listsSize[2] +listsSize[1] + listsSize[0];
  // std::cout << "number of dihedrals : " << listsSize[3] << "\n";
  // for(int i=0; i < listsSize[3] ; ++i){
  //  int paramPos = listVal[startDihedral + i][4];
  //  std::cout << i << " " << listVal[startDihedral+i][0] << " " <<
  //  listVal[startDihedral+i][4] << "\t";
  //  //std::cout << paramsVal[paramPos][0] << "\n";
  //  std::cout << "\n";
  //}

  for (int improper = 0; improper < psf->getNumImpropers(); ++improper) {
    std::string atom1 = atomTypes[impropers[improper].atom1];
    std::string atom2 = atomTypes[impropers[improper].atom2];
    std::string atom3 = atomTypes[impropers[improper].atom3];
    std::string atom4 = atomTypes[impropers[improper].atom4];

    if (atom1 > atom4) {
      std::swap(atom1, atom4);
      std::swap(atom2, atom3);
    }
    if ((atom1 == atom4) && (atom2 > atom3)) {
      std::swap(atom2, atom3);
    }

    auto key = DihedralKey(atom1, atom2, atom3, atom4);
    if (improperParams.count(key)) {
      // DihedralKeysPresent.insert(DihedralKey(atom1, atom2, atom3, atom4));
      auto findResult = std::find(improperKeysPresent.begin(),
                                  improperKeysPresent.end(), key);
      if (findResult == improperKeysPresent.end()) {
        improperKeysPresent.push_back(key);
        auto value = improperParams[key];
        // paramsVal.push_back({value.kChi, (float)value.n, value.delta});
        paramsVal.push_back({value.psi0, value.kpsi, 0, 1});
      }
      findResult = std::find(improperKeysPresent.begin(),
                             improperKeysPresent.end(), key);
      int improperType = findResult - improperKeysPresent.begin();
      listVal.push_back({impropers[improper].atom1, impropers[improper].atom2,
                         impropers[improper].atom3, impropers[improper].atom4,
                         improperType, 13, 13, 13});
    } else {
      // if (atom2 > atom3) std::swap(atom2, atom3);

      key = DihedralKey(atom1, "X", "X", atom4);
      if (improperParams.count(key)) {
        auto findResult = std::find(improperKeysPresent.begin(),
                                    improperKeysPresent.end(), key);
        if (findResult == improperKeysPresent.end()) {
          improperKeysPresent.push_back(key);
          auto value = improperParams[key];
          // paramsVal.push_back({value.kChi, (float)value.n, value.delta});
          paramsVal.push_back({value.psi0, value.kpsi, 0, 1});
        }
        findResult = std::find(improperKeysPresent.begin(),
                               improperKeysPresent.end(), key);
        int improperType = findResult - improperKeysPresent.begin();
        listVal.push_back({impropers[improper].atom1, impropers[improper].atom2,
                           impropers[improper].atom3, impropers[improper].atom4,
                           improperType, 13, 13, 13});

      } else {
        std::stringstream tmpexc;
        tmpexc << "improper not found " << improper << " "
               << atomTypes[impropers[improper].atom1] << " "
               << atomTypes[impropers[improper].atom2] << " "
               << atomTypes[impropers[improper].atom3] << " "
               << atomTypes[impropers[improper].atom4] << "\t"
               << atomNames[impropers[improper].atom1] << " "
               << atomNames[impropers[improper].atom2] << " "
               << atomNames[impropers[improper].atom3] << " "
               << atomNames[impropers[improper].atom4] << "\n";
        throw std::invalid_argument(tmpexc.str());
      }
    }
  }

  paramsSize.push_back(improperKeysPresent.size());
  listsSize.push_back(listVal.size() - listsSize[0] - listsSize[1] -
                      listsSize[2] - listsSize[3]);

  for (int i = 0; i < psf->getNumCrossTerms(); ++i) {
    auto cmap = cmaps[i];
    // std::cout << cmap.atomi1 << " " << cmap.atomj1 << " " << cmap.atomk1 <<
    // "
    // "
    //           << cmap.atoml1 << " " << cmap.atomi2 << " " << cmap.atomj2 <<
    //           "
    //           "
    //           << cmap.atomk2 << " " << cmap.atoml2 << "\n";
    // std::cout << atomTypes[cmap.atomi1] << " " << atomTypes[cmap.atomj1] <<
    // "
    // "
    //           << atomTypes[cmap.atomk1] << " " << atomTypes[cmap.atoml1] <<
    //           "
    //           "
    //           << atomTypes[cmap.atomi2] << " " << atomTypes[cmap.atomj2] <<
    //           "
    //           "
    //           << atomTypes[cmap.atomk2] << " " << atomTypes[cmap.atoml2]
    //           << "\n";
    auto dihe1 = DihedralKey(atomTypes[cmap.atomi1], atomTypes[cmap.atomj1],
                             atomTypes[cmap.atomk1], atomTypes[cmap.atoml1]);
    auto dihe2 = DihedralKey(atomTypes[cmap.atomi2], atomTypes[cmap.atomj2],
                             atomTypes[cmap.atomk2], atomTypes[cmap.atoml2]);
    // std::cout << dihe1 << " " << dihe2 << "\n";
    auto key = CmapKey(dihe1, dihe2);
    // auto key = CmapKey(atomTypes[cmap.atom1], atomTypes[cmap.atom2],
    //                    atomTypes[cmap.atom3], atomTypes[cmap.atom4],
    //                    atomTypes[cmap.atom5]);
    // if (cmapParams.count(key)) {
    //   auto findResult =
    //       std::find(cmapKeysPresent.begin(), cmapKeysPresent.end(), key);
    //   if (findResult == cmapKeysPresent.end()) {
    //     cmapKeysPresent.push_back(key);
    //     auto value = cmapParams[key];
    //     paramsVal.push_back(value);
    //   }
    //   findResult =
    //       std::find(cmapKeysPresent.begin(), cmapKeysPresent.end(), key);
    //   int cmapType = findResult - cmapKeysPresent.begin();
    //   listVal.push_back({cmap.atom1, cmap.atom2, cmap.atom3, cmap.atom4,
    //                      cmap.atom5, cmapType, 13, 13, 13, 13});
    // } else {
    //   std::stringstream tmpexc;
    //   tmpexc << "cmap not found " << i << " " << key << "\n";
    //   throw std::invalid_argument(tmpexc.str());
    // }
  }

  // CMAP are currently not being used
  paramsSize.push_back(0);
  listsSize.push_back(0);
  return BondedParamsAndLists(paramsSize, paramsVal, listsSize, listVal);
}

VdwParamsAndTypes
// CharmmParameters::getVdwParamsAndTypes(std::unique_ptr<CharmmPSF> &psf) {
CharmmParameters::getVdwParamsAndTypes(std::shared_ptr<CharmmPSF> &psf) {
  std::vector<float> psfVdwParams, psfVdw14Params;
  std::vector<int> psfVdwTypes, psfVdw14Types;

  std::set<std::string> vdwAtomTypesMap, vdw14AtomTypesMap;
  for (const auto &atomType : psf->getAtomTypes()) {
    vdwAtomTypesMap.insert(atomType);

    // auto findResult = std::find(vdw14Params.begin(),
    // vdw14Params.end(),atomType);
    auto findResult = vdw14Params.find(atomType);
    if (findResult != vdw14Params.end())
      vdw14AtomTypesMap.insert(atomType);
  }

  std::vector<std::string> vdwAtomTypes(vdwAtomTypesMap.begin(),
                                        vdwAtomTypesMap.end());
  std::vector<std::string> vdw14AtomTypes(vdw14AtomTypesMap.begin(),
                                          vdw14AtomTypesMap.end());

  // int count = 0;
  for (int i = 0; i < vdwAtomTypes.size(); ++i) {
    for (int j = 0; j <= i; ++j) {
      std::string iType = vdwAtomTypes[i];
      std::string jType = vdwAtomTypes[j];

      double epsilon, rmin;

      std::tuple<std::string, std::string> nbfixKey{jType, iType};
      // BondKey nbfixKey{jType, iType};
      if (nbfixParams.find(nbfixKey) != nbfixParams.end()) {
        // std::cout << "NBFIX : " << iType << " " << jType << "\n";

        NBFixParameters nbf{nbfixParams[nbfixKey]};

        epsilon = nbfixParams[nbfixKey].emin;
        rmin = nbfixParams[nbfixKey].rmin;
      } else {

        double epsilonI = vdwParams[iType].epsilon;
        double epsilonJ = vdwParams[jType].epsilon;

        double rmin_2I = vdwParams[iType].rmin_2;
        double rmin_2J = vdwParams[jType].rmin_2;

        epsilon = std::sqrt(epsilonI * epsilonJ);
        rmin = rmin_2I + rmin_2J;
      }

      float c12 = epsilon * std::pow(rmin, 12);
      float c6 = 2 * epsilon * std::pow(rmin, 6);

      psfVdwParams.push_back(c6);
      psfVdwParams.push_back(c12);

      // std::cout << count++ << "(" << i << "," << j << ") : " << 6*c6 <<
      // "\t"
      // << 12*c12 << "\n";
    }
  }

  // psfVdw14Params = psfVdwParams;

  for (int i = 0; i < vdwAtomTypes.size(); ++i) {
    for (int j = 0; j <= i; ++j) {
      std::string iType = vdwAtomTypes[i];
      std::string jType = vdwAtomTypes[j];

      double epsilon, rmin;

      std::tuple<std::string, std::string> nbfixKey{jType, iType};
      // BondKey nbfixKey{jType, iType};
      if (nbfixParams.find(nbfixKey) != nbfixParams.end()) {
        // std::cout << "NBFIX : " << iType << " " << jType << "\n";

        NBFixParameters nbf{nbfixParams[nbfixKey]};

        epsilon = nbfixParams[nbfixKey].emin;
        rmin = nbfixParams[nbfixKey].rmin;
      } else {

        double epsilonI = vdwParams[iType].epsilon;
        double epsilonJ = vdwParams[jType].epsilon;

        double rmin_2I = vdwParams[iType].rmin_2;
        double rmin_2J = vdwParams[jType].rmin_2;

        if (std::find(vdw14AtomTypes.begin(), vdw14AtomTypes.end(), iType) !=
            vdw14AtomTypes.end()) {
          epsilonI = vdw14Params[iType].epsilon;
          rmin_2I = vdw14Params[iType].rmin_2;
        }

        if (std::find(vdw14AtomTypes.begin(), vdw14AtomTypes.end(), jType) !=
            vdw14AtomTypes.end()) {
          epsilonJ = vdw14Params[jType].epsilon;
          rmin_2J = vdw14Params[jType].rmin_2;
        }

        epsilon = std::sqrt(epsilonI * epsilonJ);
        rmin = rmin_2I + rmin_2J;
      }

      float c12 = epsilon * std::pow(rmin, 12);
      float c6 = 2 * epsilon * std::pow(rmin, 6);

      psfVdw14Params.push_back(c6);
      psfVdw14Params.push_back(c12);
    }
  }

  int index = 0;
  for (const auto &atomType : psf->getAtomTypes()) {
    auto result = std::find(vdwAtomTypes.begin(), vdwAtomTypes.end(), atomType);
    int pos = result - vdwAtomTypes.begin();
    psfVdwTypes.push_back(pos);
    psfVdw14Types.push_back(pos);
    // std::cout << "index :" << index << " pos : " << pos << " " << atomType
    // <<
    // "\n";

    index++;
  }

  return VdwParamsAndTypes(psfVdwParams, psfVdw14Params, psfVdwTypes,
                           psfVdw14Types);
}
