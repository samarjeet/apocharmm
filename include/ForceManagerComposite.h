// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

/*
 * This header file contains the interface for Force management in CHARMM.
 * This class ensures that the Forces are destroyed when the ForceManager
 * is destroyed.
 *
 */

#pragma once

#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaBondedForce.h"
#include "CudaNeighborList.h"
#include "CudaNeighborListBuild.h"
#include "CudaPMEDirectForce.h"
#include "CudaPMEReciprocalForce.h"
#include "CudaTopExcl.h"
#include "ForceManager.h"
#include "PBC.h"
#include "TestForce.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief Composite class of several ForceManager
 *
 * Designed to deal with several different PSFs, allowing for example the use
 * of EDS methods (see EDSForceManager)
 */
class ForceManagerComposite : public ForceManager {
public:
  /**
   * @brief Base constructor.
   */
  ForceManagerComposite(void);

  /**
   * @brief Constructor from a list of ForceManager objects
   * @param fmList Vector of ForceManager objects
   *
   * Adds sequentially each member of the vector to the composite.
   */
  ForceManagerComposite(
      const std::vector<std::shared_ptr<ForceManager>> &fmList);

public:
  /** @deprecated Seems only adapted for a ForceManagerComposite with two
   * children FMs */
  void setLambda(const float lambda);

  /** @brief The selector vector contains all 0 but one 1. The index of the 1
   * (true) value is the index of the child ForceManager to be used as a
   * "driver" for the simulation */
  virtual void setSelectorVec(const std::vector<float> &lambda);

  /**
   * @brief Add a ForceManager to the composite
   *
   * @param fm ForceManager to add
   *
   * Adds force manager to the composite, creates a new XYZQ object initialized
   * with the child ForceManager's CharmmPSF (this allows for a different set
   * of atomic charges to be used)
   *
   */
  void addForceManager(std::shared_ptr<ForceManager> fm);

  void setBoxDimensions(const std::vector<double> &size) override;

  void setKappa(const float kappa) override;
  void setCutoff(const float cutoff) override;
  void setCtonnb(const float ctonnb) override;
  void setCtofnb(const float ctofnb) override;

  void setFFTGrid(const int nfftx, const int nffty, const int nfftz) override;

  void setPmeSplineOrder(const int pmeSplineOrder) override;

public:
  /**
   * @brief Returns CharmmPSF of the first child
   */
  std::shared_ptr<CharmmPSF> getPsf(void);

  /**
   * @brief Returns true if all FM children are initialized, false otherwise
   */
  bool isInitialized(void) const override;

  virtual std::shared_ptr<Force<double>> getForces(void) override;

  /**
   * @brief Returns the Force (*i.e.* force values) of a children, given its
   * index
   *
   * @param[in] childIdx index of the child whose forces should be returned
   *
   */
  std::shared_ptr<Force<double>> getForcesInChild(const int childIdx);

  const std::vector<double> &getBoxDimensions(void) const override;
  std::vector<double> &getBoxDimensions(void) override;

  /**
   * @brief Get the total potential energies of all the children
   *
   * @return CudaContainer<double> &
   */
  virtual CudaContainer<double> &getPotentialEnergy(void) override;

  virtual float getPotentialEnergies(void) override;

  virtual CudaContainer<double> &getVirial(void) override;

  /**
   * @return True
   */
  bool isComposite(void) const override;

  /**
   * @deprecated Seems only adapted for a ForceManagerComposite with two
   * children FMs
   */
  float getLambda(void) const;

  /**
   * @brief Returns the number of ForceManager within this composite
   *
   * Corresponds to the size of vector children
   */
  int getCompositeSize(void) const;

  const std::vector<std::shared_ptr<ForceManager>> &
  getChildren(void) const override;
  std::vector<std::shared_ptr<ForceManager>> &getChildren(void) override;

public:
  /**
   * @brief Initializes all children, prepare containers
   */
  void initialize(void) override;

  void resetNeighborList(const float4 *xyzq) override;

  /**
   * @brief We need a method to compute energies for ALL the children and return
   * all (for FEP uses for example)
   */
  CudaContainer<double> computeAllChildrenPotentialEnergy(const float4 *xyzq);

  /**
   * @brief Calls calc_force on each child ForceManager
   *
   * Calculate the forces for each children ForceManager.
   * Fills in the coordinates in the individual fm's XYZQ
   * and invokes their force calculations
   */
  virtual void calcForce(const float4 *xyzq, const bool reset = false,
                         const bool calcEnergy = false,
                         const bool calcVirial = false) override;

  std::shared_ptr<ForceManagerComposite> shared_from_this(void) {
    return std::static_pointer_cast<ForceManagerComposite>(
        ForceManager::shared_from_this());
  }

protected:
  /**
   * @brief List of XYZQs, one per child
   */
  std::vector<CudaContainer<float4>> m_XYZQs;

  std::vector<std::shared_ptr<Force<double>>> m_ChildrenTotalForceValues;

  /** @brief Weighing factor, if two states are combined
   *
   * Energy is computed following ?
   */
  float m_Lambda;

  CudaContainer<double> m_ChildrenPotentialEnergy;
  std::vector<float> m_Lambdas;
  std::shared_ptr<cudaStream_t> m_CompositeStream;
};
