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
#ifndef CUDADIRECTFORCETYPES_H
#define CUDADIRECTFORCETYPES_H

struct DirectEnergyVirial_t {
  // Energies
  double energy_vdw;
  double energy_elec;
  double energy_excl;

  // Finished virial
  double vir[9];

  // DP Shift forces for virial calculation
  double sforce[27 * 3];

  // FP Shift forces for virial calculation
  long long int sforce_fp[27 * 3];
};

struct DirectSettings_t {
  float kappa;
  float kappa2;

  float boxx;
  float boxy;
  float boxz;

  float roff, roff2, roff3, roff5;
  float ron, ron2;

  float roffinv;
  float roffinv2;
  float roffinv3;
  float roffinv4;
  float roffinv5;
  float roffinv6;
  float roffinv12;
  float roffinv18;

  float inv_roff2_ron2_3;

  float k6, k12, dv6, dv12;

  float ga6, gb6, gc6;
  float ga12, gb12, gc12;
  float GAconst, GBcoef;

  float Aconst, Bconst, Cconst, Dconst;
  float dvc;

  float Acoef, Bcoef, Ccoef;
  float Denom, Eaddr, Constr;

  float e14fac;

  float hinv;
  float *ewald_force;
  bool q_p21;
  float lambda;
};

// Enum for VdW and electrostatic models
enum {
  NONE = 0,
  VDW_VSH = 1,
  VDW_VSW = 2,
  VDW_VFSW = 3,
  VDW_VGSH = 4,
  VDW_CUT = 5,
  VDW_DBEXP = 6,
  VDW_SC = 7,
  EWALD = 101,
  CSHIFT = 102,
  CFSWIT = 103,
  CSHFT = 104,
  CSWIT = 105,
  RSWIT = 106,
  RSHFT = 107,
  RSHIFT = 108,
  RFSWIT = 109,
  GSHFT = 110,
  EWALD_LOOKUP = 111
};

// Enum for vdwparam
enum { VDW_MAIN, VDW_IN14 };

#endif // CUDADIRECTFORCETYPES_H
#endif // NOCUDAC
