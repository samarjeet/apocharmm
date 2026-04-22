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
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaBondedForce.h"
#include "CudaNeighborList.h"
#include "CudaNeighborListBuild.h"
#include "CudaPMEDirectForce.h"
#include "CudaPMEReciprocalForce.h"
#include "CudaTopExcl.h"
#include "PBC.h"
#include "TestForce.h"

// Forward declaration
class CharmmContext;

class ForceView {
public:
  template <typename ForceType>
  ForceView(ForceType *inputForce)
      : m_Force(static_cast<void *>(inputForce)),
        clear_impl{[](void *force) -> void {
          ForceType *ptr = static_cast<ForceType *>(force);
          return ptr->clear();
        }},
        calc_force_impl{[](void *force, const float4 *xyzq, bool calcEnergy,
                           bool calcVirial) -> void {
          ForceType *ptr = static_cast<ForceType *>(force);
          return ptr->calc_force(xyzq, calcEnergy, calcVirial);
        }},
        getForceImpl{[](void *force) -> std::shared_ptr<Force<long long int>> {
          ForceType *ptr = static_cast<ForceType *>(force);
          return ptr->getForce();
        }} {}
  //       ,
  // getEnergyVirialImpl{[](void *force) -> CudaEnergyVirial {
  //   ForceType *ptr = static_cast<ForceType *>(force);
  //   return ptr->getEnergyVirial();
  // }} {}

  void clear(void) { return this->clear_impl(m_Force); }

  void calc_force(const float4 *xyzq, bool calcEnergy, bool calcVirial) {
    return this->calc_force_impl(m_Force, xyzq, calcEnergy, calcVirial);
  }

  std::shared_ptr<Force<long long int>> getForce(void) {
    return this->getForceImpl(m_Force);
  }

  // void add()

private:
  void *m_Force;
  void (*clear_impl)(void *force);
  void (*calc_force_impl)(void *force, const float4 *xyzq, bool calcEnergy,
                          bool calcVirial);
  std::shared_ptr<Force<long long int>> (*getForceImpl)(void *force);
  // void (*getEnergyVirialImpl)(void *force);
};

/**
 * @brief Class handling simulation forces (bonded and non-bonded)
 *
 * Contains all forces (bonded and non bonded), parameters for the non-bonded
 * force treatments (cutoff distances), PBC and parameters.
 * Initialized from a topology entry (CharmmPSF) and a set of parameters
 * (CharmmParameters).
 */
class ForceManager : public std::enable_shared_from_this<ForceManager> {
public:
  ForceManager(void);

  /**
   * @brief Create a ForceManager, using psf and prm objects
   *
   * Creates a ForceManager (not yet initialized !) using a CharmmPSF and
   * CharmmParameters as inputs.
   *
   * @param[in] psf CharmmPSF
   * @param[in] prm CharmmParameters
   *
   */
  ForceManager(std::shared_ptr<CharmmPSF> psf,
               std::shared_ptr<CharmmParameters> prm);

  /** @brief Copy-constructor
   *
   * Copies any member of fmIn that are not setup by initialize()
   * This includes :
   *   - CharmmPSF psf,
   *   - CharmmParameters prm,
   *   - non-bonded parameters (nfftx/y/z, boxx/y/z)
   * Members NOT copied are thus cudaStream_t, Force, CudaForce objects.
   * Neither are objects CudaEnergyVirial
   *
   * @attention  CharmmContext is **not** set by the copy constructor.
   *
   * @param[in] fmIn ForceManager object to be copied
   */
  ForceManager(const ForceManager &other);

  // /** @brief WIP Tentative "copy-constructor-from-a-pointer". Will require
  // some
  //  * testing.
  //  *
  //  * Given a std::shared_ptr<ForceManager> object, generates a *deep* copy of
  //  * the fm pointed at. Two parts :
  //  *  - copy all attributes
  //  *  - do any "initialization" work done by a standard constructor
  //  */
  // ForceManager(std::shared_ptr<ForceManager> fmIn);

  ~ForceManager(void);

public:
  /**
   * @brief Sets this ForceManager's CharmmContext and vice-versa
   *
   * Sets context to input CharmmContext, and if context' forceManager is not
   * this object, sets context' forceManager to this ForceManager.
   *
   * @param ctx CharmmContext object
   */
  void setCharmmContext(std::shared_ptr<CharmmContext> ctx);

  /** @brief Sets "psf" variable to a given CharmmPSF object
   *
   * Sets initialized to false.
   * @param[in] psfIn CharmmPSF object
   */
  void setPsf(std::shared_ptr<CharmmPSF> psf);

  /**
   * @brief Setup psf using a file name
   *
   * If used, one should expect to have to re-initialize the ForceManager.
   */
  void addPsf(const std::string &psfFile);

  /**
   * @brief Setup parameters using a (single) file name.
   *
   * If used, one should expect to have to re-initialize the ForceManager.
   */
  void addPrm(const std::string &prmFile);
  void addPrm(const std::vector<std::string> &prmList);

  /**
   * @brief Given a vector, sets values of boxx,y,x and vector boxDimensions
   * Also heuristically sets the PME grid to an even number just smaller than
   * the box dimension,
   * @param size x,y,z vector
   */
  virtual void setBoxDimensions(const std::vector<double> &size);

  virtual void setKappa(const float kappa);
  virtual void setCutoff(const float cutoff);
  virtual void setCtonnb(const float ctonnb);
  virtual void setCtofnb(const float ctofnb);

  /**
   * @brief Set FFT grid size.
   *
   * @todo we should have an in-house method to do select FFT grid size
   * automatically
   */
  virtual void setFFTGrid(const int nfftx, const int nffty, const int nfftz);

  virtual void setPmeSplineOrder(const int pmeSplineOrder);

  /**
   * @brief Sets periodic boundary conditions (PBC)
   */
  void setPeriodicBoundaryCondition(const PBC _pbc);

  /**
   * @brief set Vdw model to use
   */
  virtual void setVdwType(const int vdwType);

  /**
   * @brief Sets the printEnergyDecomposition flag to the bool value given as
   * input (default true).
   */
  void setPrintEnergyDecomposition(const bool printEnergyDecomposition = true);

  virtual void addForceManager(std::shared_ptr<ForceManager> fm);

public:
  /**
   * @brief Returns current CharmmContext
   */
  std::shared_ptr<CharmmContext> getContext(void);

  /**
   * @brief Test if this ForceManager has a CharmmContext
   *
   * Returns false if context is nullptr, true otherwise
   */
  bool hasCharmmContext(void) const;

  /**
   * @brief Returns pointer to the CharmmPSF attached to this ForceManager
   */
  virtual std::shared_ptr<CharmmPSF> getPsf(void);

  /**
   * @brief Returns number of atoms from CharmmPSF
   *
   * Returns number of atoms extracted from the psf (via getNumAtoms)
   */
  int getNumAtoms(void) const;

  /**
   * @brief Returns current CharmmParameters
   */
  std::shared_ptr<CharmmParameters> getPrm(void);

  /**
   * @brief Returns value of initialized flag
   */
  virtual bool isInitialized(void) const;

  const CudaContainer<int4> &getShakeAtoms(void) const;
  CudaContainer<int4> &getShakeAtoms(void);

  const CudaContainer<float4> &getShakeParams(void) const;
  CudaContainer<float4> &getShakeParams(void);

  /**
   * @brief EnergyVirial getters...
   */
  const CudaEnergyVirial &getBondedEnergyVirial(void) const;
  CudaEnergyVirial &getBondedEnergyVirial(void);
  const CudaEnergyVirial &getReciprocalEnergyVirial(void) const;
  CudaEnergyVirial &getReciprocalEnergyVirial(void);
  const CudaEnergyVirial &getDirectEnergyVirial(void) const;
  CudaEnergyVirial &getDirectEnergyVirial(void);

  /**
   * @brief  returns decomposition of energy as vector of double. Intended as
   * output only.
   * @todo This shouldn't stay like that forever : what happens when we want to
   * add new forces ?
   */
  std::map<std::string, double> getEnergyComponents(void);

  /**
   * @brief Stream getters...
   */
  std::shared_ptr<cudaStream_t> getBondedStream(void);
  std::shared_ptr<cudaStream_t> getReciprocalStream(void);
  std::shared_ptr<cudaStream_t> getDirectStream(void);
  std::shared_ptr<cudaStream_t> getForceManagerStream(void);

  /**
   * @brief Force getters...
   */
  std::shared_ptr<Force<long long int>> getBondedForcevalues(void);
  std::shared_ptr<Force<long long int>> getReciprocalForcevalues(void);
  std::shared_ptr<Force<long long int>> getDirectForcevalues(void);
  std::shared_ptr<Force<double>> getTotalForcevalues(void);

  /**
   * @brief Returns pointer m_TotalForceValues
   */
  virtual std::shared_ptr<Force<double>> getForces(void);

  int getForceStride(void) const;

  /**
   * @brief Returns m_BoxDimensions
   */
  virtual const std::vector<double> &getBoxDimensions(void) const;
  virtual std::vector<double> &getBoxDimensions(void);

  /**
   * @brief Nonbonded param getters...
   */
  float getKappa(void) const;
  float getCutoff(void) const;
  float getCtonnb(void) const;
  float getCtofnb(void) const;

  /** @brief FFT grid size getter */
  std::vector<int> getFFTGrid(void) const;

  /**
   * @brief Returns current periodic boundary conditions (PBC)
   */
  PBC getPeriodicBoundaryCondition(void) const;

  virtual CudaContainer<double> &getPotentialEnergy(void);

  /**
   * @brief
   * @todo displace to Device side (right now done on host)
   */
  virtual float getPotentialEnergies(void);

  /**
   * @brief Sums all three components of the virial (direct, bonded,
   * reciprocal), transfers back to device, returns as a CudaContainer
   */
  virtual CudaContainer<double> &getVirial(void);

  int getVdwType(void) const;

  /**
   * @brief False if not composite ForceManager
   */
  virtual bool isComposite(void) const;

  // /**
  //  * @brief Returns CharmmPSF bonds. get rid of this ?
  //  *
  //  * @todo accessing psf bonds through psf->getBonds() seems like a less
  //  noisy
  //  * way to do things, and more systematic (avoids the need for getters
  //  * everywhere, and rather uses jumps from object to object as
  //  * obj1->obj2->obj3->getStuff()
  //  */
  // virtual const std::vector<Bond> &getBonds(void) const;
  // virtual std::vector<Bond> &getBonds(void);

  virtual const std::vector<std::shared_ptr<ForceManager>> &
  getChildren(void) const;
  virtual std::vector<std::shared_ptr<ForceManager>> &getChildren(void);

public:
  /**
   * @brief Initializes and allocates variables/arrays required to perform
   * simulation (Force, CudaForce, cudaStream_t objects)
   *
   * For each of the Bonded, Direct and Reciprocal parts (in that order), does
   * the following:
   *  - creates cudaStream_t to handle computation
   *  - gets param/topo/exclusion/... information
   *  - creates & allocates Force object
   *  - creates CudaBondedForce / CudaPMEDirectForce / CudaPMEReciprocalForce
   *    object
   *     * sets lists/parameters
   *     * links to the Force and Stream objects
   *
   * Then creates totalForceValues (pointer to Force, a storage class for force
   * values).
   *
   * cudaCheck(cudaDeviceSynchronize());
   *
   */
  virtual void initialize(void);

  /**
   * @brief Resets neighbor list
   *
   * @param xyzq Pointer to XYZQ
   */
  virtual void resetNeighborList(const float4 *xyzq);

  void calcForcePart1(const float4 *xyzq, const bool reset,
                      const bool calcEnergy, const bool calcVirial);
  void calcForcePart2(const float4 *xyzq, const bool reset,
                      const bool calcEnergy, const bool calcVirial);
  void calcForcePart3(const float4 *xyzq, const bool reset,
                      const bool calcEnergy, const bool calcVirial);
  /**
   * @brief Compute forces, energy, optionally virial
   *
   * Computes all forces (bonded, non-bonded direct space, non-bonded
   * reciprocal space) via CudaPMEDirectForce, CudaBondedForce,
   * reciprocalForcePtr
   *
   * NB: calcEnergy argument is hard-coded to true.
   *
   * @return Total **potential** energy (bonded + non bonded)
   * @callergraph
   */
  virtual void calcForce(const float4 *xyzq, bool reset = false,
                         bool calcEnergy = false, bool calcVirial = false);

  template <typename ForceType>
  void subscribe(std::shared_ptr<ForceType> force, const std::string &forceTag,
                 std::shared_ptr<cudaStream_t> forceStream,
                 std::shared_ptr<Force<long long int>> forceValues,
                 std::shared_ptr<CudaEnergyVirial> energyVirial) {
    m_ForcePtrs.push_back(static_cast<std::shared_ptr<void>>(force));
    m_ForceViews.push_back(ForceView(force.get()));
    m_ForceTags.push_back(forceTag);
    m_ForceStreams.push_back(forceStream);
    m_ForceValues.push_back(forceValues);
    m_EnergyVirials.push_back(energyVirial);
    return;
  }

  template <typename ForceType>
  void unsubscribe(std::shared_ptr<ForceType> force) {
    for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
      if (static_cast<void *>(force.get()) ==
          static_cast<void *>(m_ForcePtrs[i].get())) {
        m_ForcePtrs.erase(m_ForcePtrs.begin() + i);
        m_ForceViews.erase(m_ForceViews.begin() + i);
        m_ForceTags.erase(m_ForceTags.begin() + i);
        m_ForceStreams.erase(m_ForceStreams.begin() + i);
        m_ForceValues.erase(m_ForceValues.begin() + i);
        m_EnergyVirials.erase(m_EnergyVirials.begin() + i);
        break;
      }
    }
    return;
  }

  void unsubscribe(const std::string &forceTag) {
    for (std::size_t i = 0; i < m_ForceTags.size(); i++) {
      if (m_ForceTags[i] == forceTag) {
        m_ForcePtrs.erase(m_ForcePtrs.begin() + i);
        m_ForceViews.erase(m_ForceViews.begin() + i);
        m_ForceTags.erase(m_ForceTags.begin() + i);
        m_ForceStreams.erase(m_ForceStreams.begin() + i);
        m_ForceValues.erase(m_ForceValues.begin() + i);
        m_EnergyVirials.erase(m_EnergyVirials.begin() + i);
        break;
      }
    }
    return;
  }

  virtual CudaContainer<double>
  computeAllChildrenPotentialEnergy(const float4 *xyzq);

protected:
  /**
   * @todo refine the selection criteria
   * @todo doc this
   */
  void initializeHolonomicConstraintsVariables(void);

  /**
   * @brief Computes a FFT grid size by taking integers close to the box
   * dimensions. Requires the box dimension to be set.
   *
   * @return The FFT grid size as a vector of integers
   */
  std::vector<int> computeFFTGridSize(void);

  /**
   * @brief Check that a vector contains correct dimensions (positive, non-zero
   * numbers). Returns true if so, throws an error otherwise.
   */
  void checkBoxDimensions(const std::vector<double> &size);

private:
  void dealloc(void);

protected:
  /**
   * @brief CharmmContext object linked to this force manager
   */
  std::shared_ptr<CharmmContext> m_Context;

  /**
   * @brief PSF file (CharmmPSF object)
   */
  std::shared_ptr<CharmmPSF> m_Psf;

  /**
   * @brief CharmmParameters
   */
  std::shared_ptr<CharmmParameters> m_Prm;

  /**
   * @brief Flag tracking if the ForceManager has been initialized
   */
  bool m_IsInitialized;

  /**
   * @brief Atoms with SHAKE constraint
   */
  CudaContainer<int4> m_ShakeAtoms;

  /**
   * @brief Parameters of SHAKE constraint
   */
  CudaContainer<float4> m_ShakeParams;

  // TODO : these should not be directly here

  /** @brief energy-virial objects
   * @todo these should not be directly here
   *
   * Contain energy and virial terms
   */
  CudaEnergyVirial m_BondedEnergyVirial;
  CudaEnergyVirial m_ReciprocalEnergyVirial;
  CudaEnergyVirial m_DirectEnergyVirial;

  /**
   * @brief cudaStream_t object used to handle bonded forces
   */
  std::shared_ptr<cudaStream_t> m_BondedStream;
  std::shared_ptr<cudaStream_t> m_ReciprocalStream;
  std::shared_ptr<cudaStream_t> m_DirectStream;

  /**
   * @brief Pointer to cudaStream_t object, created in initialize().
   */
  std::shared_ptr<cudaStream_t> m_ForceManagerStream;

  /**
   * @brief Force object containig the bonded forces
   */
  std::shared_ptr<Force<long long int>> m_BondedForceValues;
  std::shared_ptr<Force<long long int>> m_ReciprocalForceValues;
  std::shared_ptr<Force<long long int>> m_DirectForceValues;

  /**
   * @brief Contains the total value of *minus* the Force on each atom (storage
   * class)
   *
   * It actually contains the gradient of the energy, hence the minus sign.
   */
  std::shared_ptr<Force<double>> m_TotalForceValues;

  /**
   * @brief Box dimension (Angstroms ?)
   * @todo We need to rationalize the box dimension variables (use only
   * boxDimensions for example ?)
   */
  float m_BoxX;
  float m_BoxY;
  float m_BoxZ;

  // box dimensions -- a vector this time ?
  /**
   * @brief Vector containing x,y,z dimensions of the box
   *
   * Contains boxx, boxy, boxz.
   * Setup by setBoxDimensions
   *
   */
  std::vector<double> m_BoxDimensions;

  // Long range and PME options

  /**
   * @brief PME Kappa parameter. Default: 0.34
   */
  float m_Kappa;

  /** @brief Cutoff value for generation of pair list. Default: 14 Angstroms.
   *
   * Equivalent of CUTNB in CHARMM.
   */
  float m_Cutoff;

  /**
   * @brief Cutoff distance for the non-bonded interactions at which the
   * smoothing function reaches 0. Default: 12 Angstroms
   */
  float m_Ctonnb;

  /**
   * @brief Cutoff distance for the non-bonded interactions at which the
   * smoothing starts scaling. Default: 10 Angstroms.
   */
  float m_Ctofnb;

  // FFT grid size
  int m_NfftX;
  int m_NfftY;
  int m_NfftZ;

  /**
   * @brief PME order of splines to use
   */
  int m_PmeSplineOrder;

  /** @brief PBC type (P1 or P21). Set to P1 by default upon construction.
   */
  PBC m_Pbc;

  // interface is slowing down my progress with the code, hence I am just
  // keeping unique_ptr to the forces here right now. Will fix this in a bit
  /**
   * @brief Bonded forces
   * @todo maybe shared_ptr now ?
   */
  std::unique_ptr<CudaBondedForce<long long int, float>> m_BondedForcePtr;

  /**
   * @brief Non-bonded indirect space forces
   */
  std::unique_ptr<CudaPMEReciprocalForce> m_ReciprocalForcePtr;

  /**
   * @brief Non-bonded direct space forces
   */
  std::unique_ptr<CudaPMEDirectForce<long long int, float>> m_DirectForcePtr;

  CudaContainer<double> m_TotalPotentialEnergy;

  CudaContainer<double> m_BondedVirial;
  CudaContainer<double> m_ReciprocalVirial;
  CudaContainer<double> m_DirectVirial;
  CudaContainer<double> m_TotalVirial;

  /**
   * @brief Testing graph implementation for minimizing clearing forces launch
   * overheads
   *
   */
  bool m_ClearGraphCreated;
  cudaGraph_t m_ClearGraph;
  cudaGraphExec_t m_CleargraphInstance;

  std::vector<std::shared_ptr<void>> m_ForcePtrs;
  std::vector<ForceView> m_ForceViews;
  std::vector<std::string> m_ForceTags;
  std::vector<std::shared_ptr<cudaStream_t>> m_ForceStreams;
  std::vector<std::shared_ptr<Force<long long int>>> m_ForceValues;
  std::vector<std::shared_ptr<CudaEnergyVirial>> m_EnergyVirials;

  /**
   * @brief just a hacky version right now to do more efficient EDS
   *
   */
  bool m_ComputeDirectSpaceForces;

  /** @brief Van der Waals model to use*/
  int m_VdwType;

  /**
   * @brief Only for compiling purposes.
   * @todo Add a getChildren getter to the ForceManager, returning Null (or sthg
   * like that) for a non-composite FM. REquired for MBARSubscriber
   */
  std::vector<std::shared_ptr<ForceManager>> m_Children;

private:
  /**
   * @brief Flag indicating whether to print energy decomposition or not
   */
  bool m_PrintEnergyDecomposition;
};
