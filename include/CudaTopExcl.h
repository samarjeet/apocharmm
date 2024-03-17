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
#ifndef CUDATOPEXCL_H
#define CUDATOPEXCL_H
//
// Class for topological exclusions
//
// (c) Antti-Pekka Hynninen, 2014
// aphynninen@hotmail.com
//

#include <string>
#include <vector>

class CudaTopExcl {
private:
  // Atom-atom exclusions:
  // For global atom index i, excluded atoms are in
  // atomExcl[ atomExclPos[i] ... atomExclPos[i+1]-1 ]
  int atomExclPosLen;
  int *atomExclPos;

  int atomExclLen;
  int *atomExcl;

  // Maximum number of exclusions per atom
  int maxNumExcl;

  // Number of coordinates (global)
  // const int ncoord;
  int ncoord;

  // Global -> Local index mapping
  int *glo2loc = 0;

  void setup(const int *iblo14, const int *inb14);

public:
  CudaTopExcl();
  CudaTopExcl(const int ncoord, const int *iblo14, const int *inb14);
  CudaTopExcl(const int ncoord, std::string iblo14File, std::string inb14File);
  ~CudaTopExcl();

  void setFromFile(const int numAtoms, std::string iblo14File,
                   std::string inb14File);
  void setFromVector(int numAtoms, const std::vector<int> &iblo14,
                     const std::vector<int> &inb14);
  int getAtomExclPosLen() { return atomExclPosLen; }
  int getAtomExclLen() { return atomExclLen; }
  int getAtomExclPosLen() const { return atomExclPosLen; }
  int getAtomExclLen() const { return atomExclLen; }

  int *getAtomExclPos() { return atomExclPos; }
  int *getAtomExcl() { return atomExcl; }
  const int *getAtomExclPos() const { return atomExclPos; }
  const int *getAtomExcl() const { return atomExcl; }

  int getMaxNumExcl() { return maxNumExcl; }
  int getMaxNumExcl() const { return maxNumExcl; }

  int get_ncoord() { return ncoord; }
  int get_ncoord() const { return ncoord; }

  int *get_glo2loc() { return glo2loc; }
  int *get_glo2loc() const { return glo2loc; }
};

#endif // CUDATOPEXCL_H
#endif // NOCUDAC
