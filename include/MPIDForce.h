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

/**
 * This version of MPIDForce is highly inspired by the version implemented by AS
 * in OpenMM.
 */

#include "CudaContainer.h"
#include "CudaEnergyVirial.h"
#include <memory>

class Multipole {
  int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
  double charge, thole, dampingFactor;
  std::vector<double> polarity;

  std::vector<double> molecularDipole;     // Ordered as X Y Z
  std::vector<double> molecularQuadrupole; // Ordered as XX  XY  YY  XZ  YZ  ZZ
  std::vector<double>
      molecularOctopole; // Ordered as XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ
  std::vector<std::vector<int>> covalentInfo;
};

template <typename AT, typename CT> class MPIDForce {

public:
  MPIDForce(CudaEnergyVirial &energyVirial);

  void setStream(std::shared_ptr<cudaStream_t> _stream);
  void setBoxDimensions(std::vector<double> dim) { boxDimensions = dim; }

  enum NonBondedMethod { NoCutoff, PME };

  enum InducedDipoleMethod { Mutual, Direct, Extrapolated };

  enum AxisType {
    ZThenX,
    Bisector,
    ZBisect,
    ThreeFold,
    ZOnly,
    NoAxisType,
    LastAxisTypeIndex
  };

  void setNonBondedMethod(NonBondedMethod _nonBondedMethod);

  void setInducedDipoleMethod(InducedDipoleMethod _polarizationType);

  void setCutoff(double _cutoff);

  void setFFTGrid(int _nx, int _ny, int _nz);

  void setKappa(double _kappa);

  void setDefaultTholeWidth(double _defaultTholeWidth);

  void setNumAtoms(int _numAtoms);

  int getNumAtoms() const;

  void setDipoles(const std::vector<float> &dipoles);
  void setQuadrupoles(const std::vector<float> &quadrupoles);
  void setOctopoles(const std::vector<float> &octopoles);

  void setup();

  void calculateForce(const float4 *xyzq, bool calcEnergy, bool calcVirial);
  void calc_force(const float4 *xyzq, bool calcEnergy, bool calcVirial);

  // for debugging
  void printSphericalDipoles(void);
  void printSphericalQuadrupoles(void) ;

private:
  CudaEnergyVirial &energyVirial;
  std::shared_ptr<cudaStream_t> stream;
  std::vector<double> boxDimensions;

  std::vector<Multipole> multipoles;

  NonBondedMethod nonBondedMethod;

  int nx, ny, nz;
  double kappa, cutoff, defaultTholeWidth;
  InducedDipoleMethod polarizationType;

  int numAtoms;

  CudaContainer<float> molecularDipoles, molecularQuadrupoles,
      molecularOctopoles;

  CudaContainer<float> sphericalDipoles, sphericalQuadrupoles,
      sphericalOctopoles;

  CudaContainer<float> labFrameDipoles, labFrameQuadrupoles, labFrameOctopoles;

  CudaContainer<int4> multipoleParticles;
};
