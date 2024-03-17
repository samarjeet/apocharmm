// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE
#ifndef NOCUDAC
#ifndef BONDED_STRUCT_H
#define BONDED_STRUCT_H

#include <iostream>
#include <vector>

enum {
  BOND,
  UREYB,
  ANGLE,
  DIHE,
  IMDIHE,
  CMAP,
  IN14,
  EX14,
  PAIR,
  TRIP,
  QUAD,
  SOLVENT
};
const int CONST_START = PAIR;

struct bond_t {
  int i, j, itype;
  static int type() { return 0; }
  static int size() { return 2; }
  void getAtoms(std::vector<int> &atoms) {
    atoms.resize(size());
    atoms.at(0) = i;
    atoms.at(1) = j;
  }
  void printAtoms() { std::cout << i << " " << j; }
};

struct angle_t {
  int i, j, k, itype;
  static int type() { return 1; }
  static int size() { return 3; }
  void getAtoms(std::vector<int> &atoms) {
    atoms.resize(size());
    atoms.at(0) = i;
    atoms.at(1) = j;
    atoms.at(2) = k;
  }
  void printAtoms() { std::cout << i << " " << j << " " << k; }
};

struct dihe_t {
  int i, j, k, l, itype;
  static int type() { return 2; }
  static int size() { return 4; }
  void getAtoms(std::vector<int> &atoms) {
    atoms.resize(size());
    atoms.at(0) = i;
    atoms.at(1) = j;
    atoms.at(2) = k;
    atoms.at(3) = l;
  }
  void printAtoms() { std::cout << i << " " << j << " " << k << " " << l; }
};

struct cmap_t {
  int i1, j1, k1, l1, i2, j2, k2, l2, itype;
  static int type() { return 3; }
  static int size() { return 8; }
  void getAtoms(std::vector<int> &atoms) {
    atoms.resize(size());
    atoms.at(0) = i1;
    atoms.at(1) = j1;
    atoms.at(2) = k1;
    atoms.at(3) = l1;
    atoms.at(4) = i2;
    atoms.at(5) = j2;
    atoms.at(6) = k2;
    atoms.at(7) = l2;
  }
  void printAtoms() {
    std::cout << i1 << " " << j1 << " " << k1 << " " << l1 << " " << i2 << " "
              << j2 << " " << k2 << " " << l2;
  }
};

struct xx14_t {
  int i, j;
  static int type() { return 4; }
  static int size() { return 2; }
  void getAtoms(std::vector<int> &atoms) {
    atoms.resize(size());
    atoms.at(0) = i;
    atoms.at(1) = j;
  }
  void printAtoms() { std::cout << i << " " << j; }
};

struct solvent_t {
  int i, j, k;
  static int type() { return 5; }
  static int size() { return 3; }
  void getAtoms(std::vector<int> &atoms) {
    atoms.resize(size());
    atoms.at(0) = i;
    atoms.at(1) = j;
    atoms.at(2) = k;
  }
  void printAtoms() { std::cout << i << " " << j << " " << k; }
};

// Data structures for bonds, angles, dihedrals, and cmap
struct bondlist_t {
  int i, j, itype, ishift;
};

struct anglelist_t {
  int i, j, k, itype, ishift1, ishift2;
};

struct dihelist_t {
  int i, j, k, l, itype, ishift1, ishift2, ishift3;
};

struct cmaplist_t {
  int i1, j1, k1, l1, i2, j2, k2, l2, itype, ishift1, ishift2, ishift3;
};

struct xx14list_t {
  int i, j, ishift;
};

#endif // BONDED_STRUCT_H
#endif // NOCUDAC
