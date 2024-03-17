// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CudaVerletIntegrator.h"
#include "catch.hpp"
#include <iostream>
#include <vector>

// Check that propagating from a restart file yields the same results as a
// continuous propagation
TEST_CASE("deterministic") { CHECK(false); }

TEST_CASE("verlet integrator", "[dynamics]") {
  /*CharmmContext sim;
  int numAtoms = 9;
  sim.setNumAtoms(numAtoms);
  sim.setPeriodicBoxSizes(50., 50., 50.);
  sim.setDirectSpaceParameters(9.0);
  std::vector<std::vector<float>> coordsCharges ={
      {0.0,0.0,0.0,-0.83},
      {0.50, 0.0, 0.0, 0.085},
      {-.25, 0.35,0.0,0.085},
      {0.0,0.0,3.0,-0.83},
      {1.0, 0.0, 3.0, 0.085},
      {-.25, 0.85,3.0,0.085},
      {0.0,2.0,3.0,-0.83},
      {1.0, 2.0, 3.0, 0.085},
      {-.25, 2.85,3.0,0.085}
      };
  sim.setCoordsCharges(coordsCharges);
  std::vector<double> masses =
  {15.994, 1.008, 1.008, 15.994, 1.008, 1.008, 15.994, 1.008, 1.008};
  sim.setMasses(masses);
  std::vector<float> vdwParams = {0.0007538169157, 3.088260428e-06, 10.47430038,
  327.9039001, 595.0144043,     581923.375,      0,           9.79418993e+25};
  sim.setVdwType({1,0,0,1,0,0,1,0,0});
  sim.setVdwParam(vdwParams);
  std::vector<int> iblo14={0,0,0,0,0,0,0,0,0}, inb14;
  sim.setTopologicalExclusions(iblo14, inb14);
  //std::vector<int> inExSize = {0,9};std::vector<int> inEx
  ={0,1,1,2,0,2,3,4,3,5,4,5,6,7,6,8,7,8}; std::vector<int> inExSize =
  {0,0};std::vector<int> inEx ; sim.set14InclusionExclusion(inExSize, inEx);
  sim.setReciprocalSpaceParameters(50, 50, 50, 4, 0.34);
  sim.assignVelocitiesAtTemperature(298.13);
  // set masses
  CudaVerletIntegrator integrator(0.001);
  // a shared pointer must be passed in
  integrator.setSimulationContext(&sim);
  sim.calculateReciprocalSpace();
  integrator.initialize();
  integrator.propagate(10);

*/
}
