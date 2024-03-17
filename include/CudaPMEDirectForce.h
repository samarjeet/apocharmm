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
#ifndef CUDAPMEDIRECTFORCE_H
#define CUDAPMEDIRECTFORCE_H
#include "Bonded_struct.h"
#include "CudaDirectForceTypes.h"
#include "CudaEnergyVirial.h"
#include "CudaNeighborList.h"
#include "CudaNeighborListBuild.h"
#include "CudaP21NeighborListBuild.h"
#include "CudaTopExcl.h"
#include "Force.h"
#include "PBC.h"
#include "XYZQ.h"
#include <cuda.h>
#include <memory>
#include <string>
#include <vector>

// If this variable is set, we'll use texture objects.
// Unset for now because of the problem with texture objects on GTX 750
// #define USE_TEXTURE_OBJECTS

//
// Calculates direct non-bonded interactions on GPU
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//
// AT = accumulation type
// CT = calculation type
//

/**
 * @brief Abstract base class to create CudaPMEDirectForce objects.
 *
 * Purely abstract base class. Used to declare the methods that are expected in
 * all derived classes
 */
template <typename AT, typename CT> class CudaPMEDirectForceBase {
  // class CudaPMEDirectForceBase  {
public:
  virtual ~CudaPMEDirectForceBase() {}
  virtual void setup(double boxx, double boxy, double boxz, double kappa,
                     double roff, double ron, double e14fac, int vdw_model,
                     int elec_model, bool q_p21) = 0;
  virtual void get_setup(float &boxx, float &boxy, float &boxz, float &kappa,
                         float &roff, float &ron, float &e14fac, int &vdw_model,
                         int &elec_model, bool &q_p21) = 0;

  virtual void get_box_size(CT &boxx, CT &boxy, CT &boxz) = 0;
  virtual void set_box_size(const CT boxx, const CT boxy, const CT boxz) = 0;

  virtual void set_calc_vdw(const bool calc_vdw) = 0;
  virtual void set_calc_elec(const bool calc_elec) = 0;
  virtual bool get_calc_vdw() = 0;
  virtual bool get_calc_elec() = 0;

  virtual void set_vdwparam(const int nvdwparam, const CT *h_vdwparam) = 0;
  // virtual void set_vdwparam(const int nvdwparam, const char *filename) = 0;
  virtual void set_vdwparam14(const int nvdwparam, const CT *h_vdwparam) = 0;
  // virtual void set_vdwparam14(const int nvdwparam, const char *filename) = 0;

  virtual void set_vdwtype(const int ncoord, const int *h_vdwtype) = 0;
  // virtual void set_vdwtype(const int ncoord, const char *filename) = 0;
  virtual void set_vdwtype(const int ncoord, const int *glo_vdwtype,
                           const int *loc2glo, cudaStream_t stream = 0) = 0;

  virtual void set_14_list(int nin14list, int nex14list, xx14list_t *h_in14list,
                           xx14list_t *h_ex14list, cudaStream_t stream = 0) = 0;

  virtual void set_14_list(const float4 *xyzq, const float boxx,
                           const float boxy, const float boxz,
                           const int *glo2loc_ind, const int nin14_tbl,
                           const int *in14_tbl, const xx14_t *in14,
                           const int nex14_tbl, const int *ex14_tbl,
                           const xx14_t *ex14, cudaStream_t stream = 0) = 0;

  virtual void calc_14_force(const float4 *xyzq, const bool calc_energy,
                             const bool calc_virial, const int stride,
                             AT *force, cudaStream_t stream = 0) = 0;

  virtual void calc_force(const int whichlist, const float4 *xyzq,
                          const CudaNeighborListBuild<32> &nlist,
                          const bool calc_energy, const bool calc_virial,
                          const int stride, AT *force,
                          cudaStream_t stream = 0) = 0;
  /*virtual void calc_force(const int whichlist, const float4 *xyzq,
                          const CudaP21NeighborListBuild &nlist,
                          const bool calc_energy, const bool calc_virial,
                          const int stride, AT *force,
                          cudaStream_t stream = 0) = 0;
  */
};

//
// Actual class
//
template <typename AT, typename CT>
class CudaPMEDirectForce : public CudaPMEDirectForceBase<AT, CT> {
protected:
  /** @brief Energy & Virial */
  CudaEnergyVirial &energyVirial;

  /** @brief Energy term names */
  std::string strVdw;
  std::string strElec;
  std::string strExcl;

  /** @brief  VdW parameters */
  int nvdwparam;
  int vdwparam_len;
  CT *vdwparam;
  const bool use_tex_vdwparam;
#ifdef USE_TEXTURE_OBJECTS
  bool vdwParamTexObjActive;
  cudaTextureObject_t vdwParamTexObj;
#endif

  /** @brief VdW 1-4 parameters */
  int nvdwparam14;
  int vdwparam14_len;
  CT *vdwparam14;
  const bool use_tex_vdwparam14;
#ifdef USE_TEXTURE_OBJECTS
  bool vdwParam14TexObjActive;
  cudaTextureObject_t vdwParam14TexObj;
#endif

  /** @brief 1-4 interaction and exclusion lists */
  int nin14list;
  int in14list_len;
  xx14list_t *in14list;

  int nex14list;
  int ex14list_len;
  xx14list_t *ex14list;

  /** @brief VdW types */
  int vdwtype_len;
  int *vdwtype;
  int *vdwtype14;
  int *vdwtypeTemp;

  /** @brief Type of VdW and electrostatic models (see above: NONE, VDW_VSH,
   * VDW_VSW ...) */
  int vdw_model;
  int elec_model;

  /** @brief Flag noting if vdW terms are calculated.
   *
   * These flags are true if the vdw/elec terms are calculated.
   * true by default */
  bool calc_vdw;
  bool calc_elec;

  bool q_p21;

  /** @brief Lookup table for Ewald. Used if elec_model == EWALD_LOOKUP */
  CT *ewald_force;
  int n_ewald_force;

  /** @brief Host version of setup */
  DirectSettings_t *h_setup;

  void setup_ewald_force(CT h);
  /** @brief Sets method for calculating electrostatic force and energy */
  void set_elec_model(int elec_model, CT h = 0.01);
  void update_setup();

  void setup_vdwparam(const int type, const int nvdwparam,
                      const CT *h_vdwparam);
  // void load_vdwparam(const char *filename, const int nvdwparam, CT
  // **h_vdwparam);

  XYZQ xyzq_sorted;
  /**
   * @brief Topological exclusion
   */
  CudaTopExcl topExcl;
  std::shared_ptr<CudaNeighborList<32>> neighborList;
  CudaNeighborListBuild<32> *nlist[2];
  int zone_patom[9];
  int *loc2glo = 0;
  std::vector<double> boxDimensions;
  std::shared_ptr<Force<long long int>> forceVal;
  std::shared_ptr<Force<long long int>> forceSortedVal;
  std::shared_ptr<cudaStream_t> directStream;
  int numAtoms;
  float cutoff;
  int *glo_vdwtype;

public:
  CudaPMEDirectForce(CudaEnergyVirial &energyVirial, const char *nameVdw,
                     const char *nameElec, const char *nameExcl);
  // Move constructor
  CudaPMEDirectForce(CudaPMEDirectForce &&other);
  ~CudaPMEDirectForce();

  /** @brief Sets parameters for the nonbonded computation */
  void setup(double boxx, double boxy, double boxz, double kappa, double roff,
             double ron, double e14fac, int vdw_model, int elec_model,
             bool q_p21);
  /** @brief Returns parameters for the nonbonded computation */
  void get_setup(float &boxx, float &boxy, float &boxz, float &kappa,
                 float &roff, float &ron, float &e14fac, int &vdw_model,
                 int &elec_model, bool &q_p21);

  void clearTextures();

  /** @brief Returns box sizes */
  void get_box_size(CT &boxx, CT &boxy, CT &boxz);
  /** @brief Sets box size from three numbers */
  void set_box_size(const CT boxx, const CT boxy, const CT boxz);
  /** @brief Sets box size from box size in global memory */
  void set_box_size(const double3 *d_boxSize);

  void set_calc_vdw(const bool calc_vdw);
  void set_calc_elec(const bool calc_elec);
  bool get_calc_vdw();
  bool get_calc_elec();

  void set_vdwparam(const int nvdwparam, const CT *h_vdwparam);
  // void set_vdwparam(const int nvdwparam, const char *filename);
  void set_vdwparam(const std::vector<CT> vdwparam);
  void set_vdwparam14(const int nvdwparam, const CT *h_vdwparam);
  // void set_vdwparam14(const int nvdwparam, const char *filename);
  void set_vdwparam14(const std::vector<CT> vdwparam);

  void set_vdwtype(const int ncoord, const int *h_vdwtype);
  void set_vdwtype14(const int ncoord, const int *h_vdwtype);
  void set_vdwtype(const std::vector<int> &vdwTypeVec);
  void set_vdwtype14(const std::vector<int> &vdwTypeVec);
  // void set_vdwtype(const int ncoord, const char *filename);
  // void set_vdwtype14(const int ncoord, const char *filename);
  void set_vdwtype(const int ncoord, const int *glo_vdwtype, const int *loc2glo,
                   cudaStream_t stream = 0);
  void set_sort_vdwtype(const int ncoord, const int *ind_sorted,
                        cudaStream_t stream = 0);
  void set_sort_vdwtype(const int ncoord, const int *ind_sorted);

  void set_14_list(int nin14list, int nex14list, xx14list_t *h_in14list,
                   xx14list_t *h_ex14list, cudaStream_t stream = 0);
  void set_14_list(std::string sizeFile, std::string valFile,
                   cudaStream_t stream = 0);
  void set_14_list(const std::vector<int> &size, const std::vector<int> &val,
                   cudaStream_t stream = 0);
  void set_14_list(const float4 *xyzq, const float boxx, const float boxy,
                   const float boxz, const int *glo2loc_ind,
                   const int nin14_tbl, const int *in14_tbl, const xx14_t *in14,
                   const int nex14_tbl, const int *ex14_tbl, const xx14_t *ex14,
                   cudaStream_t stream = 0);

  void calc_14_force(const float4 *xyzq, const bool calc_energy,
                     const bool calc_virial, const int stride, AT *force,
                     cudaStream_t stream = 0);

  /** @brief Calculates direct force
   * @todo */
  void calc_force(const int whichlist, const float4 *xyzq,
                  const CudaNeighborListBuild<32> &nlist,
                  const bool calc_energy, const bool calc_virial,
                  const int stride, AT *force, cudaStream_t stream = 0);
  /** @brief Calculates direct force. Uses calc_force parent function.
   * @todo (comment l1391) resetNeighborList(xyzq, numAtoms) "has to be under a
   * heuristic control". Is it the case yet ?
   * @todo uses cuda function "setSortedCoords", noted "TODO:testing for now"
   */

  void calc_force(const float4 *xyzq, bool calcEnergy, bool calcVirial);
  // void resetNeighborList() {}
  void setupSorted(int numAtoms);

  /**
   * @brief
   */
  void setupTopologicalExclusions(int numAtoms, std::vector<int> &iblo14,
                                  std::vector<int> &inb14);
  void setupNeighborList(int numAtoms);
  void resetNeighborList(const float4 *xyzq, int numAtoms);
  void setBoxDimensions(std::vector<double> dim);
  void setStream(std::shared_ptr<cudaStream_t> streamIn) {
    directStream = streamIn;
  }
  void setNumAtoms(int n) {
    numAtoms = n;
    // TODO : this should be in the constructor
    forceSortedVal->realloc(numAtoms, 1.5f);
    forceVal->realloc(numAtoms, 1.5f);
  }

  void setCutoff(float cutoffIn) { cutoff = cutoffIn; }

  void setForce(std::shared_ptr<Force<long long int>> &forceValIn) {
    forceVal = forceValIn;
  }
};

#endif // CUDAPMEDIRECTFORCE_H
#endif // NOCUDAC
