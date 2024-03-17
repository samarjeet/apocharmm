// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include "CudaEnergyVirial.h"
#include "Force.h"
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/*
GEO [MAXGEO integer] [shape_specification] [position_spec] [RCM]
              [potential_spec] [atom_selection] [ DISTANCE atom_selection]
                        [ ADISTANCE atom_selection atom_selection ] [PERP]
                        [ ANGLE atom_selection atom_selection ]
                        [ DIHEDRAL 3 X atom_selection ]


shape_specification:==  { [SPHERE] } [XREF real] [YREF real] [ZREF real] -
                                     [TREF real]
                        { CYLINDER } [XDIR real] [YDIR real] [ZDIR real]
                        { PLANAR   }

potential_spec:== { HARMonic } { INSIDE    } [FORCE real] -
                                             [DROFF real] [DTOFF real]
                  { QUARtic  } { OUTSIDE   } [P1 real] [P2 real]
                  { EXPOnent } { SYMMETRIC }
                  { GAUSsian }
                  { SAWOod   }



atom-selection:== (see *note select:(chmdoc/select.doc).)

*/

enum class RestraintShape { SPHERE, CYLINDER, PLANE };

enum class PotentialFunction {
  HARMONIC,
  QUARTIC,
  EXPONENTIAL,
  GAUSSIAN,
  SAWOOD
};

struct Restraint {
  RestraintShape shape;
  PotentialFunction potential;
  bool isCenterOfMass;
  float3 origin;
  bool relativeToBox; // 0.5 to -0.5
  float3 orientation;
  bool insideOnly;
  float forceConstant;
  float offsetDistance;
  //__host__ __device__ std::vector<int> atoms;
  int atoms[100];
  int size;
};

//
// Calculates geometric restraint forces
//
template <typename AT, typename CT> class GeometricRestraintForce {
public:
  // GeometricRestraintForce();

  GeometricRestraintForce(CudaEnergyVirial &energyVirial);

  void setForce(std::shared_ptr<Force<long long int>> &forceValIn);

  void addRestraint(RestraintShape shape, PotentialFunction potential,
                    bool isCenterOfMass, float3 origin, bool relativeToBox,
                    float3 oritentation, bool insideOnly, float forceConstant,
                    float offsetDistance, std::vector<int> atoms);

  void initialize();

  void calc_force(const float4 *xyzq, bool calcEnergy, bool calcVirial);

private:
  CudaEnergyVirial &energyVirial;
  std::shared_ptr<Force<long long int>> forceVal;
  thrust::device_vector<Restraint> restraints; // use a unified memory vector
  // thrust::device_vector<thrust::device_vector<int>> restraintAtoms;
};
