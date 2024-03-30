// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

/**\file*/

#pragma once
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "CharmmPSF.h"
#include "Coordinates.h"
#include "CudaContainer.h"
// #include "CudaHolonomicConstraint.h"
#include "CudaMinimizer.h"
#include "Force.h"
#include "ForceManager.h"
#include "Logger.h"
#include "NonEquilibriumForceManager.h"
#include "Subscriber.h"
#include "XYZQ.h"
#include "cuda_utils.h"
#include <random_utils.h>

// #include "Checkpoint.h"

// Forward declaration
class Logger;
class Checkpoint;

/**
 * @brief Mediator class for the MD simulation.
 * @attention Requires a call to assignVelocitiesAtTemperature !!!
 *
 * Requires a ForceManager object to be initialized.
 */
class CharmmContext : public std::enable_shared_from_this<CharmmContext> {
public:
  /**
   * @brief Creates a CharmmContext object, based on a ForceManager.
   *
   * If the input ForceManager is not initialized, calls
   * ForceManager::initialize(). Also calls forceManager setCharmmContext.
   *
   * @param[in] fmIn the ForceManager.
   */
  CharmmContext(std::shared_ptr<ForceManager> fmIn);
  ~CharmmContext() = default;

  /**
   * @brief Tentative copy constructor. Does **not** setup ForceManager.
   *
   * @todo unittest this: same attribute vlaues for the copy, acting on copy
   * does not change the original
   */
  CharmmContext(const CharmmContext &ctxIn);

  void setupFromCheckpoint(std::shared_ptr<Checkpoint> checkpoint);

  /**
   * @brief Set the coordinates <b>and the charges</b>
   *
   * Fills in the coordsCharge Container with atom coordinates extracted from a
   * Coordinates object, and the charges extracted from the ForceManager
   * CharmmPSF object.  Also resets the neighborlist
   * @param crd  shared_ptr to the CharmmCrd object
   *
   * @note Using crd instead of &crd for pybind11 reasons
   */
  void setCoordinates(const std::shared_ptr<Coordinates> crd);

  /**
   * @brief Set the coordinates <b>and the charges</b>
   *
   * Fills in the coordsCharge Container with atom coordinates extracted from a
   * vector<vector<double>> object, and the charges extracted from the
   * ForceManager CharmmPSF object.  Also resets the neighborlist
   * @param crd  vector of vector (dimensions N,3) of atom coordinates
   */
  void setCoordinates(const std::vector<std::vector<double>> crd);

  /**
   * @brief Returns the coordinates as a vector of vector of double
   *
   * @return std::vector<std::vector<double>>
   */
  std::vector<std::vector<double>> getCoordinates();

  /**
   * @brief Sets the temperature.
   * @param[in] temp The temperature.
   * @remark Does not initialize velocities: it is a pure "setter".
   *
   * @todo remove this: the `temperature` variable isn't used anywhere ?
   */
  void setTemperature(const float temp);

  /**
   * @brief  Returns the temperature variable (does **not** compute it)
   * @return The temperature.
   *
   * See computeTemperature for temperature computation.
   *
   * @todo remove this: the `temperature` variable isn't used anywhere ?
   */
  float getTemperature();

  /**
   * @brief Compute temperature from kinetic energy
   *
   * Computes (on GPU) the total kinetic energy, returns the average
   * temperature as 2*E_kin/(n_dof*k_B)
   */
  float computeTemperature();

  /** @brief Set periodic boundary condition (PBC) to the ForceManager
   * @param[in] _pbc: PBC object
   */
  void setPeriodicBoundaryCondition(const PBC _pbc);

  /** @brief Returns periodic boundary condition (PBC) */
  PBC getPeriodicBoundaryCondition();

  // for getting minimization working
  // will improve the semantics of the setters and getters later
  void setCoords(const std::vector<float> &coords);
  std::vector<float> getCoords();

  /** @brief Sets systems atomic charges
   *
   * Currently, there are *two* data containers for the atomic charges :
   *    _ an XYZQ object (deprecated) xyzq
   *    _ a CudaContainer object coordsCharge (such a good name <3)
   *
   * Sets the fourth column of both of these containers to be a N-sized vector
   * containing the value of the atomic charges.
   *
   * @param[in] &charges: pointer to vector<float> containing atomic charge
   * values
   *
   * @todo not impl/not used
   */
  void setCharges(std::vector<float> &charges);

  // void setPeriodicBoxSizes(double x, double y, double z);
  void setMasses(const char *fileName);

  /**
   * @brief Calculates kinetic energy
   * @return No return
   *
   * Computation happens on the device (GPU), through
   * calculateKineticEnergyKernel.
   */
  void calculateKineticEnergy();
  /**
   * @brief Calculates kinetic energy and transfers it to host memory
   * @return Kinetic energy : double
   */
  double getKineticEnergy();

  /**
   * @brief Calculates kinetic energy and puts it on device memory
   * DOES NOT transfers the KE to the host memory
   * @return CudaContainer with kinetic energy on host memory
   */
  CudaContainer<double> getKineticEnergy_();

  // float getKineticEnergy();
  // std::shared_ptr<Force<long long int>> getForcesLLI();

  /**
   * @brief Get forces from the ForceManager
   */
  std::shared_ptr<Force<double>> getForces();

  /**
   * @brief Get number of atoms
   */
  int getNumAtoms() const;

  /** @brief Returns a pointer to the XYZQ */
  XYZQ *getXYZQ();

  int *get_loc2glo() const;
  int getForceStride() const;
  //
  // Getters and setters

  /**
   * @brief sets numAtoms AND MASSES !!! Not a simple setter.
   *
   * @attention This function does too much given its name.
   * It should only set the numAtoms variables, but is also allocating
   * velocityMass array and setting masses.
   * @todo split it in parts.
   */
  void setNumAtoms(const int num);

  void setCoordsCharges(const std::vector<float4> &coordsChargesIn);
  void setCoordsCharges(const std::vector<std::vector<float>> &coordsChargesIn);

  void resetNeighborList();
  float calculatePotentialEnergy(bool reset = false, bool print = false);
  /** @brief Compute the potential energy (using the ForceManager calc_force
   * function)
   * @todo Param "reset" not implemented in ForceManager::calc_force
   *       Param "print" doesnt do anything
   * @attention Does NOT return the potential energy value, but 0.0.
   *       The potential energy is stored in CudaContainer<double>
   *       totalPotentialEnergy.
   *
   * @return 0.0
   *
   * Computation is done on the GPU device.
   */
  // float calculatePotentialEnergy(bool reset = false);
  /** @brief Call calc_force on each ForceManager.
   */
  float calculateForces(bool reset = false, bool calcEnergy = false,
                        bool calcVirial = false);

  void setMasses(const std::vector<double> &masses);

  /**
   * @brief Generates velocities following Boltzmann distribution
   *
   * @attention If using holonomic constraint (e.g. TIP3 water, SHAKE), this
   * will result in much hotter temperature.
   */
  void assignVelocitiesAtTemperature(float temp);

  /**
   * @brief Set the Random Seed For Velocities object
   * If using this, make sure that it is called before
   * assignVelocitiesAtTemperature
   * @param seed
   */
  void setRandomSeedForVelocities(uint64_t _seed) { seed = _seed; }

  uint64_t getRandomSeedForVelocities() { return seed; }

  /***@brief Removes the center of mass motion **/
  void removeCenterOfMassMotion();

  /**
   * @brief Reads a CHARMM velocity frame and uses it to set velocities
   * It is mostly aimed for debug purposes and should not be exposed through
   * Python interface.
   *
   * @param fileName
   */
  void assignVelocitiesFromCHARMMVelocityFile(std::string fileName);

  /** @brief Assign velocities from a vector of values (dimensions (3*N),
   * [v1x,v1y,v1z,v2x,v2y...]) */
  void assignVelocities(const std::vector<double> velIn);

  /** @brief Assign velocities from a (N,3) vector of vectors */
  void assignVelocities(const std::vector<std::vector<double>> velIn);

  /**
   * @brief Get the Velocity Mass CudaContainer.
   *
   * @return CudaContainer<double4>&
   */
  CudaContainer<double4> &getVelocityMass();

  /**
   * @brief Get the Coordinates and Charge CudaContainer.
   *
   * @return CudaContainer<double4>&
   */
  CudaContainer<double4> &getCoordinatesCharges();

  const std::vector<double> &getBoxDimensions();

  void setBoxDimensions(const std::vector<double> &boxDimensionsIn);

  std::vector<Bond> getBonds();

  /** @brief Uses CharmmPSF getDegreesOfFreedom, removes holonomic constraints
   *
   * For now, only removes dofs coming from TIP3 water model.
   *
   * @todo Should take into account other possible constraints. Should remove
   * 3*n_waters **only if using rigid water** model.
   * Should be renamed as "computeDegreesOfFreedom" maybe ? As it's not just a
   * simple getter.
   */
  int getDegreesOfFreedom();

  /** @brief Returns the int numDegreesOfFreedom.  */
  int getNumDegreesOfFreedom();

  void imageCentering();

  std::shared_ptr<ForceManager> getForceManager() { return forceManager; }

  /** @brief
   *
   * Calls ForceManager's getPotentialEnergies
   */
  std::vector<float> getPotentialEnergies();

  /**
   * @brief Get the Potential Energy value.
   * It does not calculate the potential energy only returns it from the last
   * potential energy calculation.
   *
   * @return CudaContainer<double>
   */
  CudaContainer<double> getPotentialEnergy();

  double getVolume() const;

  /**
   * @brief Calculates pressure using kinetic energy from on-step velocity.
   * @warning Currently not working
   * @todo Fix it
   */
  void computePressure();

  CudaContainer<double> getPressure() { return pressure; }
  /**
   * @brief Returns the virial (as a CudaContainer) from the ForceManager
   */
  CudaContainer<double> getVirial();
  /**
   * @brief Get water molecules for setting up SETTLE constraints
   *
   * @return CudaContainer<int4> water molecules
   */
  CudaContainer<int4> getWaterMolecules();

  /**
   * @brief Get the atoms involved in H bonds : pairs, triplets, quads
   * (atom1, atom2, atom3, atom4)
   * atom3 and atom4 will be -1 for cases where pairs and triplets are
   * enumerated
   *
   * @return CudaContainer<int4> list of 4 atoms involved in a SHAKE constraint
   */
  CudaContainer<int4> getShakeAtoms();

  CudaContainer<float4> getShakeParams();

  /** @brief recomputes the number of degrees of freedom depending on the use
   * -or not- of holonomic constraints, sets up the usingHolonomicConstraints
   * flag
   * @todo Rename or split in two functions
   */
  void useHolonomicConstraints(bool set);
  bool isUsingHolonomicConstraints() const { return usingHolonomicConstraints; }

  void orient();
  /**
   * @brief Set ForceManager object
   *
   * Typically done by the constructor.
   */
  void setForceManager(std::shared_ptr<ForceManager> fm);

  /**
   * @brief Links this CharmmContext to forceManager. Has to be called after
   * constructor.
   *
   * This function has to be called AFTER the constructor.
   */
  void linkBackForceManager();

  // TODO : Only for debug
  // remove this
  void writeCrd(std::string fileName);

  /**
   * @brief Set the coordinates using numpy array
   * @warning Not implemented yet
   * @todo
   */
  void setCoordinatesNumpy(pybind11::array_t<double> input_array);

  /**
   * @brief Reads coordinates and velocities from a restart file.
   *
   * @todo This needs to be tested
   */
  void readRestart(std::string fileName);

  void setLogger();
  void setLogger(std::string loggerFileName);
  std::shared_ptr<Logger> getLogger();
  bool hasLoggerSet() const;

private:
  // std::vector<std::vector<int>> groups;
  uint64_t seed;

  /**
   * @brief ForceManager object
   *
   * Requires to be linked AFTER the constructor is called. See
   * either linkBackForceManager() or ForceManager 's setCharmmContext
   * functions.
   */
  std::shared_ptr<ForceManager> forceManager = nullptr;

  /** @brief Number of atoms in the system. Usually set upon calling
   * SetCoordinates.
   *
   * Initially set to -1 to catch any initialization missing.
   */
  int numAtoms = -1;

  /**
   * @brief DOFs
   * Updated in #useHolonomicConstraints
   */
  int numDegreesOfFreedom;

  /** @brief Periodic Boundary Conditions
   */
  PBC pbc;
  // CudaContainer<float> charges;

  /** @brief Contains positions and charges as four columns
   */
  XYZQ xyzq;

  /**
   * @brief Double precision container of the positions (x,y,z coords) and
   * atomic charge value
   */
  CudaContainer<double4> coordsCharge;

  CudaContainer<double4> velocityMass;

  CudaContainer<double> kineticEnergy;

  // pressure
  CudaContainer<double> pressure;
  CudaContainer<double> virialKineticEnergyTensor;

  /** @todo Remove this, isn't used anywhere ? */
  float temperature;

  bool usingHolonomicConstraints;

  /** @brief Logger attached to the context. Initialized upon first call to
   * propagate by any integrator.
   * @todo Should probably be a shared_ptr */
  std::shared_ptr<Logger> logger;
  bool hasLogger;
};
