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
#ifndef CUDABONDEDFORCE_H
#define CUDABONDEDFORCE_H

#include "Bonded_struct.h"
#include "CudaEnergyVirial.h"
#include "Force.h"
#include <memory>
#include <string>

//
// Calculates bonded interactions on GPU
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//
// AT = accumulation type
// CT = calculation type
//

// #define USE_DP_SFORCE

/** @brief Representation of the (bonded) force objects.
 *
 * Contains all informations regarding the bonded forces. Can compute (bonded)
 * energy, forces, virial via cuda kernels.
 *
 * @note Does NOT contain the force *values* (see Force).
 */
template <typename AT, typename CT> class CudaBondedForce {
private:
  // BondedSettings_t *h_setup;

  // BondedEnergyVirial_t *h_energy_virial;

  /** @brief Energy and virial storage for the bonded forces */
  CudaEnergyVirial &energyVirial;

  // Energy term names
  std::string strBond;
  std::string strUreyb;
  std::string strAngle;
  std::string strDihe;
  std::string strImdihe;
  std::string strCmap;

  // ------
  // Bonds
  // ------
  int nbondlist;
  int nbondcoef;

  int bondlist_len;
  bondlist_t *bondlist;

  int bondcoef_len;
  float2 *bondcoef;

  // -------------
  // Urey-Bradley
  // -------------
  int nureyblist;
  int nureybcoef;

  int ureyblist_len;
  bondlist_t *ureyblist;

  int ureybcoef_len;
  float2 *ureybcoef;

  // -------
  // Angles
  // -------
  int nanglelist;
  int nanglecoef;

  int anglelist_len;
  anglelist_t *anglelist;

  int anglecoef_len;
  float2 *anglecoef;

  // ----------
  // Dihedrals
  // ----------
  int ndihelist;
  int ndihecoef;

  int dihelist_len;
  dihelist_t *dihelist;

  int dihecoef_len;
  float4 *dihecoef;

  // -------------------
  // Improper Dihedrals
  // -------------------
  int nimdihelist;
  int nimdihecoef;

  int imdihelist_len;
  dihelist_t *imdihelist;

  int imdihecoef_len;
  float4 *imdihecoef;

  // ------
  // CMAPs
  // ------
  int ncmaplist;
  int ncmapcoef;

  int cmaplist_len;
  cmaplist_t *cmaplist;

  int cmapcoef_len;
  float2 *cmapcoef;

  std::shared_ptr<Force<long long int>> forceVal;
  std::shared_ptr<cudaStream_t> bondedStream;
  std::vector<double> boxDimensions;

public:
  CudaBondedForce();

  // Move constructor
  CudaBondedForce(CudaBondedForce &&other);
  CudaBondedForce(CudaEnergyVirial &energyVirial, const char *nameBond,
                  const char *nameUreyb, const char *nameAngle,
                  const char *nameDihe, const char *nameImdihe,
                  const char *nameCmap);
  ~CudaBondedForce();

  void setup_coef(const std::vector<int> &size,
                  const std::vector<std::vector<float>> &val);
  void setup_list(const std::vector<int> &bondedListSize,
                  const std::vector<std::vector<int>> &bondedListVal,
                  cudaStream_t stream = 0);

  void setup_coef(const int nbondcoef, const float2 *h_bondcoef,
                  const int nureybcoef, const float2 *h_ureybcoef,
                  const int nanglecoef, const float2 *h_anglecoef,
                  const int ndihecoef, const float4 *h_dihecoef,
                  const int nimdihecoef, const float4 *h_imdihecoef,
                  const int ncmapcoef, const float2 *h_cmapcoef);

  void setup_list(const int nbondlist, const bondlist_t *h_bondlist,
                  const int nureyblist, const bondlist_t *h_ureyblist,
                  const int nanglelist, const anglelist_t *h_anglelist,
                  const int ndihelist, const dihelist_t *h_dihelist,
                  const int nimdihelist, const dihelist_t *h_imdihelist,
                  const int ncmaplist, const cmaplist_t *h_cmaplist,
                  cudaStream_t stream = 0);

  void
  setup_list(const float4 *xyzq, const CT boxx, const CT boxy, const CT boxz,
             const int *glo2loc_ind, const int nbond_tbl, const int *bond_tbl,
             const bond_t *bond, const int nureyb_tbl, const int *ureyb_tbl,
             const bond_t *ureyb, const int nangle_tbl, const int *angle_tbl,
             const angle_t *angle, const int ndihe_tbl, const int *dihe_tbl,
             const dihe_t *dihe, const int nimdihe_tbl, const int *imdihe_tbl,
             const dihe_t *imdihe, const int ncmap_tbl, const int *cmap_tbl,
             const cmap_t *cmap, cudaStream_t stream = 0);

  void clear(void);

  /** @brief Compute (bonded) forces
   *
   * Takes booleans for each bonded term (bond, U-B, angle, dihedral,
   * improper).
   * Actual computation is done on cuda kernels.
   *
   * Energy terms are stored in CudaEnergyVirial object
   */
  void calc_force(const float4 *xyzq, const CT boxx, const CT boxy,
                  const CT boxz, const bool calc_energy, const bool calc_virial,
                  const int stride, AT *force, const bool calc_bond,
                  const bool calc_ureyb, const bool calc_angle,
                  const bool calc_dihe, const bool calc_imdihe,
                  const bool calc_cmap, cudaStream_t stream = 0);

  void print();
  void setForce(std::shared_ptr<Force<long long int>> &forceValIn);
  void setStream(std::shared_ptr<cudaStream_t> streamIn);
  void calc_force(const float4 *xyzq, bool calcEnergy, bool calcVirial);
  void setBoxDimensions(std::vector<double> dim) { boxDimensions = dim; }

  std::shared_ptr<Force<long long int>> getForce(void);
};

#endif // CUDABONDEDFORCE_H
#endif // NOCUDAC
