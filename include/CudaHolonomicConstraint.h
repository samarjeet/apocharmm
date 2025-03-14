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

#include "CudaContainer.h"
#include "XYZQ.h"
#include <memory>
#include <vector>

class CharmmContext;

/**
 * @brief Class for holonomic constraint
 *
 */
class CudaHolonomicConstraint {
public:
  /**
   * @brief Construct a new Cuda Holonoma object
   *
   */
  CudaHolonomicConstraint();
  // CudaHolonomicConstraint(CharmmContext &ctx);
  void setCharmmContext(std::shared_ptr<CharmmContext> context);
  void setup(double timeStep);

  // Fix this : don't pass raw pointers
  void handleHolonomicConstraints(const double4 *ref);

  void removeForceAlongHolonomicConstraints();

  void setStream(std::shared_ptr<cudaStream_t> _stream) { stream = _stream; }

  void setMemcpyStream(std::shared_ptr<cudaStream_t> _stream) {
    memcpyStream = _stream;
  }

private:
  // int numSettleMolecules;
  //  std::vector<int4> settleIndex;
  //  double tol; // float ?
  std::shared_ptr<CharmmContext> context;
  int numWaterMolecules;
  CudaContainer<int4> settleWaterIndex;
  CudaContainer<int4> shakeAtoms;
  CudaContainer<int2> allConstrainedAtomPairs; // for projecting out the forces
                                               // during minimization
  CudaContainer<float4> shakeParams;

  // CudaContainer<int2> shakeAtomPairs, shakeAtomTriples, shakeAtomQuads;

  double mO, mH, mH2O, mO_div_mH2O, mH_div_mH2O, rHHsq, rOHsq, ra, ra_inv, rb,
      rc, rc2;

  // XYZQ xyzq_stored;
  CudaContainer<double4> coords_stored;
  double timeStep;
  void constrainWaterMolecules(const double4 *ref);
  void constrainShakeAtoms(const double4 *ref);
  void updateVelocities();

  std::shared_ptr<cudaStream_t> stream, memcpyStream;
};
