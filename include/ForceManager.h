// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

/*
 * This header file contains the interface for Force management in CHARMM.
 * This class ensures that the Forces are destroyed when the ForceManager
 * is destroyed.
 *
 */

#pragma once
#include <memory>
#include <vector>
// #include <variant>
#include <algorithm>
#include <string>

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
#include "XYZQ.h"

// Forward declaration
class CharmmContext;

// /**
//  * @todo doc this...
//  */
// class ForceType {
// public:
//   template <class T>
//   ForceType(T t) noexcept : self{std::make_unique<model_t<T>>(std::move(t))}
//   {} void calc_force(const float4 *xyzq) { self->calc_force(xyzq); }

// private:
//   struct concept_t {
//     virtual ~concept_t() = default;
//     virtual void calc_force(const float4 *xyzq) = 0;
//   };

//   template <class T> struct model_t : concept_t {
//     model_t(T s) noexcept : self{std::move(s)} {}
//     // TODO : there should be other interface functions
//     void calc_force(const float4 *xyzq) override { self.calc_force(xyzq); }
//     T self;
//   };

//   std::unique_ptr<concept_t> self;
// };

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

// enum class PBC { P1, P21 };

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
  ForceManager();
  ~ForceManager() = default; //{std::cout << "In ForceManager destructor\n";}
  // ForceManager(std::unique_ptr<CharmmPSF> &psf,
  //             std::unique_ptr<CharmmParameters> &prm);

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
   *   - non-bonded parameters (nfftx/y/z, boxx/y/z
   * Members NOT copied are thus cudaStream_t, Force, CudaForce objects.
   * Neither are objects CudaEnergyVirial
   *
   * @attention  CharmmContext is **not** set by the copy constructor.
   *
   * @param[in] fmIn ForceManager object to be copied
   */
  ForceManager(const ForceManager &fmIn);

  /** @brief WIP Tentative "copy-constructor-from-a-pointer". Will require some
   * testing.
   *
   * Given a std::shared_ptr<ForceManager> object, generates a *deep* copy of
   * the fm pointed at. Two parts :
   *  - copy all attributes
   *  - do any "initialization" work done by a standard constructor
   */
  ForceManager(std::shared_ptr<ForceManager> fmIn);

  /**
   * @brief Setup psf using a file name
   *
   * If used, one should expect to have to re-initialize the ForceManager.
   */
  void addPSF(std::string psfFile);

  /**
   * @brief just a hacky version right now to do more efficient EDS
   *
   */
  bool computeDirectSpaceForces = true;
  /**
   * @brief Setup parameters using a (single) file name.
   *
   * If used, one should expect to have to re-initialize the ForceManager.
   */
  void addPRM(std::string prmFile);
  void addPRM(std::vector<std::string> prmList);

  /**
   * @brief Returns number of atoms from CharmmPSF
   *
   * Returns number of atoms extracted from the psf (via getNumAtoms)
   */
  int getNumAtoms() const;

  /**
   * @brief Returns value of initialized flag
   */
  virtual bool isInitialized() const;

  /**
   * @brief Resets neighbor list
   *
   * @param xyzq Pointer to XYZQ
   */
  virtual void resetNeighborList(const float4 *xyzq);

  /**
   * @brief Returns pointer to the CharmmPSF attached to this ForceManager
   */
  virtual std::shared_ptr<CharmmPSF> &getPSF() { return psf; }

  /** @brief Sets "psf" variable to a given CharmmPSF object
   *
   * Sets initialized to false.
   * @param[in] psfIn CharmmPSF object
   */
  void setPSF(std::shared_ptr<CharmmPSF> psfIn);

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
  virtual float calc_force(const float4 *xyzq, bool reset = false,
                           bool calcEnergy = false, bool calcVirial = false);
  // virtual std::shared_ptr<Force<long long int>> getForcesLLI();

  /**
   * @brief Returns pointer totalForceValues
   */
  virtual std::shared_ptr<Force<double>> getForces();

  /* FOR FUTURE USE
  void emplace_back(ForceType f) { forces_.emplace_back(std::move(f)); }
  */

  /**
   * @brief Given a vector, sets values of boxx,y,x and vector boxDimensions
   * Also heuristically sets the PME grid to an even number just smaller than
   * the box dimension,
   * @param size x,y,z vector
   */
  virtual void setBoxDimensions(const std::vector<double> &size);

  /**
   * @brief Returns boxDimensions
   */
  virtual const std::vector<double> &getBoxDimensions(void) const;
  virtual std::vector<double> &getBoxDimensions(void);
  virtual void setKappa(float kappaIn) { kappa = kappaIn; }
  virtual void setCutoff(float cutoffIn) { cutoff = cutoffIn; }
  virtual void setCtonnb(float ctonnbIn) { ctonnb = ctonnbIn; }
  virtual void setCtofnb(float ctofnbIn) { ctofnb = ctofnbIn; }
  virtual void setPmeSplineOrder(int order) { pmeSplineOrder = order; }

  /** @brief set Vdw model to use */
  virtual void setVdwType(int vdwTypeIn) { vdwType = vdwTypeIn; }

  int getVdwType() { return vdwType; }
  /** @brief Van der Waals model to use*/
  int vdwType;

  /**
   * @brief Set FFT grid size.
   *
   * @todo we should have an in-house method to do select FFT grid size
   * automatically
   */
  virtual void setFFTGrid(int nx, int ny, int nz) {
    nfftx = nx;
    nffty = ny;
    nfftz = nz;
  }
  /** @brief Sets periodic boundary conditions (PBC)
   */
  void setPeriodicBoundaryCondition(const PBC _pbc);
  /** @brief Returns current periodic boundary conditions (PBC) */
  PBC getPeriodicBoundaryCondition() const { return pbc; }

  int getForceStride();

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
  virtual void initialize();

  /**
   * @brief Returns CharmmPSF bonds. get rid of this ?
   *
   * @todo accessing psf bonds through psf->getBonds() seems like a less noisy
   * way to do things, and more systematic (avoids the need for getters
   * everywhere, and rather uses jumps from object to object as
   * obj1->obj2->obj3->getStuff()
   */
  virtual std::vector<Bond> getBonds();

  /**
   * @brief False if not composite ForceManager
   */
  virtual bool isComposite() const;
  virtual void addForceManager(std::shared_ptr<ForceManager> fm);
  virtual CudaContainer<double>
  computeAllChildrenPotentialEnergy(const float4 *xyzq);

  // float getPotentialEnergy(); // Does not calculate PE
  //  virtual std::vector<float> getAllPotentialEnergies(const float4* xyzq);
  /** @brief
   * @todo displace to Device side (right now done on host)
   * @todo Change return type to float (instead of a vector<float> of size 1)
   */
  virtual std::vector<float> getPotentialEnergies();
  /*
  virtual float calc_all_forces(const float4 *xyzq, bool reset = false,
                                bool calcEnergy = false,
                                bool calcVirial = false);
  */
  /**
   * @brief Sums all three components of the virial (direct, bonded,
   * reciprocal), transfers back to device, returns as a CudaContainer
   */
  virtual CudaContainer<double> getVirial();

  virtual CudaContainer<double> &getPotentialEnergy(void);

  /**
   * @brief  returns decomposition of energy as vector of double. Intended as
   * output only.
   * @todo This shouldn't stay like that forever : what happens when we want to
   * add new forces ?
   */
  std::map<std::string, double> getEnergyComponents();

  /**
   * @brief Sets this ForceManager's CharmmContext and vice-versa
   *
   * Sets context to input CharmmContext, and if context' forceManager is not
   * this object, sets context' forceManager to this ForceManager.
   *
   * @param ctx CharmmContext object
   */
  void setCharmmContext(std::shared_ptr<CharmmContext> ctx);

  /**
   * @brief Test if this ForceManager has a CharmmContext
   *
   * Returns false if context is nullptr, true otherwise
   */
  bool hasCharmmContext();

  CudaContainer<int4> getShakeAtoms() { return shakeAtoms; }
  CudaContainer<float4> getShakeParams() { return shakeParams; }

  /////////////////////////////////
  // COPY CONSTRUCTOR SHENNANIGANS
  ////////////////////////////////

  /** @brief Returns current CharmmContext */
  std::shared_ptr<CharmmContext> getContext() { return context; }
  /** @brief Returns current CharmmParameters */
  std::shared_ptr<CharmmParameters> getPrm() { return prm; }
  /** @brief Virial getters... */
  CudaEnergyVirial getBondedEnergyVirial() { return bondedEnergyVirial; }
  CudaEnergyVirial getDirectEnergyVirial() { return directEnergyVirial; }
  CudaEnergyVirial getReciprocalEnergyVirial() {
    return reciprocalEnergyVirial;
  }
  /** @brief Stream getters... */
  std::shared_ptr<cudaStream_t> getBondedStream() { return bondedStream; }
  std::shared_ptr<cudaStream_t> getDirectStream() { return directStream; }
  std::shared_ptr<cudaStream_t> getReciprocalStream() {
    return reciprocalStream;
  }
  std::shared_ptr<cudaStream_t> getForceManagerStream() {
    return forceManagerStream;
  }
  /** @brief Force getters... */
  std::shared_ptr<Force<long long int>> getBondedForcevalues() {
    return bondedForceValues;
  }
  std::shared_ptr<Force<long long int>> getReciprocalForcevalues() {
    return reciprocalForceValues;
  }
  std::shared_ptr<Force<long long int>> getDirectForcevalues() {
    return directForceValues;
  }
  std::shared_ptr<Force<double>> getTotalForcevalues() {
    return totalForceValues;
  }
  /** @brief Nonbonded param getters... */
  float getKappa() { return kappa; }
  float getCutoff() { return cutoff; }
  // float getCutnb() { return cutnb; }
  float getCtonnb() { return ctonnb; }
  float getCtofnb() { return ctofnb; }
  /** @brief FFT grid size getter */
  std::vector<int> getFFTGrid() { return {nfftx, nffty, nfftz}; }

  /**
   * @brief Only for compiling purposes.
   * @todo Add a getChildren getter to the ForceManager, returning Null (or sthg
   * like that) for a non-composite FM. REquired for MBARSubscriber
   */
  std::vector<std::shared_ptr<ForceManager>> children;

  virtual std::vector<std::shared_ptr<ForceManager>> getChildren();

  void calc_force_part1(const float4 *xyzq, bool reset, bool calcEnergy,
                        bool calcVirial);
  void calc_force_part2(const float4 *xyzq, bool reset, bool calcEnergy,
                        bool calcVirial);
  void calc_force_part3(const float4 *xyzq, bool reset, bool calcEnergy,
                        bool calcVirial);

  /**
   * @brief Sets the printEnergyDecomposition flag to the bool value given as
   * input (default true).
   */
  void setPrintEnergyDecomposition(bool bIn = true);

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

  ///////////////
  // PROTECTED //
  ///////////////

protected:
  /** @brief CharmmContext object linked to this force manager */
  std::shared_ptr<CharmmContext> context; //= nullptr;

  /* FOR FUTURE USE
   * @todo Need to be implemented. Should be some kind of "force handler" to
   * initialize everything  (?)
   * @remark For now, does nothing.
  std::vector<ForceType> forces_;
  */

  /**
   * @brief Number of atoms. Extracted from PSF.
   */
  int numAtoms;

  /**
   * @brief Flag tracking if the ForceManager has been initialized
   */
  bool initialized;

  /**
   * @brief PSF file (CharmmPSF object)
   */
  std::shared_ptr<CharmmPSF> psf;
  /**
   * @brief CharmmParameters
   */
  std::shared_ptr<CharmmParameters> prm;

  /** @brief Atoms with SHAKE constraint */
  CudaContainer<int4> shakeAtoms;
  /** @brief Parameters of SHAKE constraint */
  CudaContainer<float4> shakeParams;

  // TODO : these should not be directly here

  /** @brief energy-virial objects
   * @todo these should not be directly here
   *
   * Contain energy and virial terms
   */
  CudaEnergyVirial bondedEnergyVirial, directEnergyVirial,
      reciprocalEnergyVirial;
  // cudaStream_t bondedStream, directStream, reciprocalStream ;

  /**
   * @brief cudaStream_t object used to handle bonded forces
   */
  std::shared_ptr<cudaStream_t> bondedStream, reciprocalStream, directStream;

  /** @brief Pointer to cudaStream_t object, created in initialize() .
   */
  std::shared_ptr<cudaStream_t> forceManagerStream;

  /**
   * @brief Force object containig the bonded forces
   */
  std::shared_ptr<Force<long long int>> bondedForceValues,
      reciprocalForceValues, directForceValues;

  /**
   * @brief Contains the total value of *minus* the Force on each atom (storage
   * class)
   *
   * It actually contains the gradient of the energy, hence the minus sign.
   */
  std::shared_ptr<Force<double>> totalForceValues;

  /**
   * @brief Box dimension (Angstroms ?)
   * @todo We need to rationalize the box dimension variables (use only
   * boxDimensions for example ?)
   */
  float boxx, boxy, boxz;

  // Long range and PME options

  /** @brief PME Kappa parameter. Default: 0.34 */
  float kappa;
  /** @brief Cutoff value for generation of pair list. Default: 14 Angstroms.
   *
   * Equivalent of CUTNB in CHARMM.
   */
  float cutoff;
  /** @brief Cutoff distance for the non-bonded interactions at which the
   * smoothing function reaches 0. Default: 12 Angstroms */
  float ctonnb;
  /** @brief Cutoff distance for the non-bonded interactions at which the
   * smoothing starts scaling. Default: 10 Angstroms. */
  float ctofnb;

  // box dimensions -- a vector this time ?
  /**
   * @brief Vector containing x,y,z dimensions of the box
   *
   * Contains boxx, boxy, boxz.
   * Setup by setBoxDimensions
   *
   * @todo Switch to doubles ?
   */
  std::vector<double> boxDimensions;

  // fft grid size
  int nfftx, nffty, nfftz;
  /**
   * @brief PME order of splines to use
   */
  int pmeSplineOrder;

  /** @brief PBC type (P1 or P21). Set to P1 by default upon construction.
   */
  PBC pbc;

  // interface is slowing down my progress with the code, hence I am just
  // keeping unique_ptr to the forces here right now. Will fix this in a bit
  /**
   * @brief Bonded forces
   * @todo maybe shared_ptr now ?
   */
  std::unique_ptr<CudaBondedForce<long long int, float>> bondedForcePtr;
  /**
   * @brief Non-bonded direct space forces
   */
  std::unique_ptr<CudaPMEDirectForce<long long int, float>> directForcePtr;
  /**
   * @brief Non-bonded indirect space forces
   */
  std::unique_ptr<CudaPMEReciprocalForce> reciprocalForcePtr;

  CudaContainer<double> totalPotentialEnergy;
  CudaContainer<double> virial;

  CudaContainer<double> directVirial, bondedVirial, reciprocalVirial;

  /**
   * @todo refine the selection criteria
   * @todo doc this
   */
  void initializeHolonomicConstraintsVariables();

  /**
   * @brief Testing graph implementation for minimizing clearing forces launch
   * overheads
   *
   */
  bool clearGraphCreated = false;
  cudaGraph_t clearGraph;
  cudaGraphExec_t cleargraphInstance;

  /** @brief Computes a FFT grid size by taking integers close to the box
   * dimensions. Requires the box dimension to be set.
   *
   * @return The FFT grid size as a vector of integers
   */
  std::vector<int> computeFFTGridSize();

  /** @brief Check that a vector contains correct dimensions (positive, non-zero
   * numbers). Returns true if so, throws an error otherwise. */
  bool checkBoxDimensions(const std::vector<double> &size);

  std::vector<std::shared_ptr<void>> m_ForcePtrs;
  std::vector<ForceView> m_ForceViews;
  std::vector<std::string> m_ForceTags;
  std::vector<std::shared_ptr<cudaStream_t>> m_ForceStreams;
  std::vector<std::shared_ptr<Force<long long int>>> m_ForceValues;
  std::vector<std::shared_ptr<CudaEnergyVirial>> m_EnergyVirials;

private:
  /**
   * @brief Flag indicating whether to print energy decomposition or not
   */
  bool printEnergyDecomposition = false;

  void removeNetReciprocalForce();
};

////////////////////////////
// ForceManagerComposite  //
////////////////////////////
/**
 * @brief Composite class of several ForceManager
 *
 * Designed to deal with several different PSFs, allowing for example the use
 * of EDS methods (see EDSForceManager)
 */
class ForceManagerComposite : public ForceManager {
public:
  /** @brief Base constructor. */
  ForceManagerComposite();

  /**
   * @brief Constructor from a list of ForceManager objects
   * @param fmList Vector of ForceManager objects
   *
   * Adds sequentially each member of the vector to the composite.
   */
  ForceManagerComposite(std::vector<std::shared_ptr<ForceManager>> fmList);

  /**
   * @return True
   */
  bool isComposite() const override;

  /** @brief Returns true if all FM children are initialized, false otherwise
   */
  bool isInitialized() const override;

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

  /** @brief We need a method to compute energies for ALL the children and
   * return all (for FEP uses for example)*/
  CudaContainer<double> computeAllChildrenPotentialEnergy(const float4 *xyzq);

  /**
   * @brief Initializes all children, prepare containers
   */
  void initialize() override;

  /**
   * @brief Returns CharmmPSF of the first child
   */
  std::shared_ptr<CharmmPSF> &getPSF() override {
    // Returns the first one !?
    return children[0]->getPSF();
  }

  std::vector<Bond> getBonds() override;

  void resetNeighborList(const float4 *xyzq) override;

  /**
   * @brief Calls calc_force on each child ForceManager
   *
   * Calculate the forces for each children ForceManager.
   * Fills in the coordinates in the individual fm's XYZQ
   * and invokes their force calculations
   */
  virtual float calc_force(const float4 *xyzq, bool reset = false,
                           bool calcEnergy = false,
                           bool calcVirial = false) override;

  /**
   * @brief WIP Should return each components of each force of each child
   *
   * @todo Not doing what it's supposed to !

  virtual float calc_all_forces(const float4 *xyzq, bool reset = false,
                                bool calcEnergy = false,
                                bool calcVirial = false) override;
  */

  virtual std::shared_ptr<Force<double>> getForces() override;

  virtual CudaContainer<double> getVirial() override;

  /**
   * @brief Returns the Force (*i.e.* force values) of a children, given its
   * index
   *
   * @param[in] childId index of the child whose forces should be returned
   *
   */
  std::shared_ptr<Force<double>> getForcesInChild(int childId);

  /** @deprecated Seems only adapted for a ForceManagerComposite with two
   * children FMs */
  float getLambda() const;
  /** @deprecated Seems only adapted for a ForceManagerComposite with two
   * children FMs */
  void setLambda(float lambdaIn);
  /** @brief The selector vector contains all 0 but one 1. The index of the 1
   * (true) value is the index of the child ForceManager to be used as a
   * "driver" for the simulation */
  virtual void setSelectorVec(std::vector<float> lambdaIn);

  void setBoxDimensions(const std::vector<double> &size) override;
  void setKappa(float kappaIn) override;
  void setCutoff(float cutoffIn) override;
  void setCtonnb(float ctonnbIn) override;
  void setCtofnb(float ctofnbIn) override;
  void setPmeSplineOrder(int order) override;
  void setFFTGrid(int nx, int ny, int nz) override;

  // std::vector<float> getAllPotentialEnergies(const float4* xyzq) override;

  /**
   * @brief Returns a vector containing the potential energy of each child
   * ForceManager
   */
  virtual std::vector<float> getPotentialEnergies() override;

  /**
   * @brief Get the total potential energies of all the children
   *
   * @return CudaContainer<double>
   */
  virtual CudaContainer<double> &getPotentialEnergy() override;
  const std::vector<double> &getBoxDimensions(void) const override;
  std::vector<double> &getBoxDimensions(void) override;

  std::shared_ptr<ForceManagerComposite> shared_from_this() {
    return std::static_pointer_cast<ForceManagerComposite>(
        ForceManager::shared_from_this());
  }

  /** @brief Returns the number of ForceManager within this composite
   *
   * Corresponds to the size of vector children
   */
  int getCompositeSize();

  /**
   * @brief List of ForceManager contained
   */
  std::vector<std::shared_ptr<ForceManager>> children;

  std::vector<std::shared_ptr<ForceManager>> getChildren() override;

protected:
  /**
   * @brief List of XYZQs, one per child
   */
  std::vector<std::shared_ptr<XYZQ>> xyzqs;
  // std::vector<float> forces;

  std::vector<std::shared_ptr<Force<double>>> totalChildrenForceValues;
  /** @brief Weighing factor, if two states are combined
   *
   * Energy is computed following ?
   */
  float lambda;

  CudaContainer<double> childrenPotentialEnergy;
  std::vector<float> lambdai;
  std::shared_ptr<cudaStream_t> compositeStream;
};
