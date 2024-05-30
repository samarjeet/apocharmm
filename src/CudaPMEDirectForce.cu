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
#include "CudaDirectForceKernels.h"
#include "CudaPMEDirectForce.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cassert>
#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <thread>

//
// Sets vdwtype from a global list
//
__global__ void set_vdwtype_kernel(const int ncoord,
                                   const int *__restrict__ glo_vdwtype,
                                   const int *__restrict__ loc2glo,
                                   int *__restrict__ vdwtype) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < ncoord) {
    int j = loc2glo[i];
    vdwtype[i] = glo_vdwtype[j];
    // vdwtype[j] = glo_vdwtype[i];
  }
}

//
// Sets vdwtype from a sorting index ind_sorted and itself
//
__global__ void set_sorted_vdwtype_kernel(const int ncoord,
                                          int *__restrict__ vdwtypeTemp,
                                          const int *__restrict__ ind_sorted,
                                          int *__restrict__ vdwtype) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < ncoord) {
    vdwtypeTemp[i] = vdwtype[i];
  }
  __syncthreads();
  if (i < ncoord) {
    int j = ind_sorted[i];
    vdwtype[i] = vdwtypeTemp[j];
  }
}

__global__ void set_14_list_kernel(
    const int nin14_tbl, const int *__restrict__ in14_tbl,
    const xx14_t *__restrict__ in14, xx14list_t *__restrict__ in14list,
    const int nex14_tbl, const int *__restrict__ ex14_tbl,
    const xx14_t *__restrict__ ex14, xx14list_t *__restrict__ ex14list,
    const float4 *__restrict__ xyzq, const float3 half_box,
    const int *__restrict__ glo2loc_ind) {
  int pos = threadIdx.x + blockIdx.x * blockDim.x;
  if (pos < nin14_tbl) {
    int j = in14_tbl[pos];
    xx14_t in14v = in14[j];
    xx14list_t in14listv;
    in14listv.i = glo2loc_ind[in14v.i];
    in14listv.j = glo2loc_ind[in14v.j];
    float4 xyzq_i = xyzq[in14listv.i];
    float4 xyzq_j = xyzq[in14listv.j];
    in14listv.ishift = calc_ishift(xyzq_i, xyzq_j, half_box);
    in14list[pos] = in14listv;
  } else if (pos < nin14_tbl + nex14_tbl) {
    pos -= nin14_tbl;
    int j = ex14_tbl[pos];
    xx14_t ex14v = ex14[j];
    xx14list_t ex14listv;
    ex14listv.i = glo2loc_ind[ex14v.i];
    ex14listv.j = glo2loc_ind[ex14v.j];
    float4 xyzq_i = xyzq[ex14listv.i];
    float4 xyzq_j = xyzq[ex14listv.j];
    ex14listv.ishift = calc_ishift(xyzq_i, xyzq_j, half_box);
    ex14list[pos] = ex14listv;
  }
}

// ########################################################################################
// ########################################################################################
// ########################################################################################

// Move constructor
template <typename AT, typename CT>
CudaPMEDirectForce<AT, CT>::CudaPMEDirectForce(CudaPMEDirectForce &&other)
    : use_tex_vdwparam(other.use_tex_vdwparam),
      use_tex_vdwparam14(other.use_tex_vdwparam14),
      energyVirial(other.energyVirial), strVdw(other.strVdw),
      strElec(other.strElec), strExcl(other.strExcl) {

  h_setup = other.h_setup;
  other.h_setup = 0;

  vdwparam = other.vdwparam;
  nvdwparam = other.nvdwparam;
  vdwparam_len = other.vdwparam_len;
  vdwparam14 = other.vdwparam14;
  nvdwparam14 = other.nvdwparam14;
  vdwparam14_len = other.vdwparam14_len;

  other.vdwparam = 0;
  other.nvdwparam = 0;
  other.vdwparam_len = 0;
  other.vdwparam14 = 0;
  other.nvdwparam14 = 0;
  other.vdwparam14_len = 0;

  nin14list = other.nin14list;
  in14list_len = other.in14list_len;
  in14list = other.in14list;

  other.nin14list = 0;
  other.in14list_len = 0;
  other.in14list = NULL;

  nex14list = other.nex14list;
  ex14list_len = other.ex14list_len;
  ex14list = other.ex14list;

  other.nex14list = 0;
  other.ex14list_len = 0;
  other.ex14list = NULL;

  vdwtype = other.vdwtype;
  vdwtype14 = other.vdwtype14;
  vdwtypeTemp = other.vdwtypeTemp;
  vdwtype_len = other.vdwtype_len;

  glo_vdwtype = other.vdwtype;

  other.vdwtype = NULL;
  other.vdwtype14 = NULL;
  other.vdwtypeTemp = NULL;
  other.vdwtype_len = 0;

  ewald_force = other.ewald_force;
  n_ewald_force = other.n_ewald_force;

  other.ewald_force = NULL;
  other.n_ewald_force = 0;

  cutoff = other.cutoff;
}
//
// Class creator
//
template <typename AT, typename CT>
CudaPMEDirectForce<AT, CT>::CudaPMEDirectForce(CudaEnergyVirial &energyVirial,
                                               const char *nameVdw,
                                               const char *nameElec,
                                               const char *nameExcl)
    : // use_tex_vdwparam(true), use_tex_vdwparam14(true),
      // energyVirial(energyVirial) {
      use_tex_vdwparam(false), use_tex_vdwparam14(false),
      energyVirial(energyVirial) {
  assert(nameVdw != NULL);
  assert(nameElec != NULL);
  assert(nameExcl != NULL);

  // Insert energy terms
  energyVirial.insert(nameVdw);
  strVdw = nameVdw;

  energyVirial.insert(nameElec);
  strElec = nameElec;

  energyVirial.insert(nameExcl);
  strExcl = nameExcl;

#ifdef USE_TEXTURE_OBJECTS
  vdwParamTexObjActive = false;
  vdwParam14TexObjActive = false;
#else
  // Assert that texture references must be unbound
  assert(!get_vdwparam_texref_bound());
  assert(!get_vdwparam14_texref_bound());
#endif

  vdwparam = NULL;
  nvdwparam = 0;
  vdwparam_len = 0;

  vdwparam14 = NULL;
  nvdwparam14 = 0;
  vdwparam14_len = 0;

  nin14list = 0;
  in14list_len = 0;
  in14list = NULL;

  nex14list = 0;
  ex14list_len = 0;
  ex14list = NULL;

  vdwtype = NULL;
  vdwtype14 = NULL;
  vdwtypeTemp = NULL;
  vdwtype_len = 0;

  // new
  glo_vdwtype = NULL;
  // new

  ewald_force = NULL;
  n_ewald_force = 0;

  set_calc_vdw(true);
  set_calc_elec(true);

  // allocate<DirectEnergyVirial_t>(&d_energy_virial, 1);
  // allocate_host<DirectEnergyVirial_t>(&h_energy_virial, 1);

  allocate_host<DirectSettings_t>(&h_setup, 1);
  forceSortedVal = std::make_shared<Force<long long int>>();

  // clear_energy_virial();
}

//
// Class destructor
//
template <typename AT, typename CT>
CudaPMEDirectForce<AT, CT>::~CudaPMEDirectForce() {
  this->clearTextures();
  if (vdwparam != NULL)
    deallocate<CT>(&vdwparam);
  if (vdwparam14 != NULL)
    deallocate<CT>(&vdwparam14);
  if (in14list != NULL)
    deallocate<xx14list_t>(&in14list);
  if (ex14list != NULL)
    deallocate<xx14list_t>(&ex14list);
  if (vdwtype != NULL)
    deallocate<int>(&vdwtype);
  if (ewald_force != NULL)
    deallocate<CT>(&ewald_force);
  // if (d_energy_virial != NULL)
  // deallocate<DirectEnergyVirial_t>(&d_energy_virial); if (h_energy_virial
  // != NULL) deallocate_host<DirectEnergyVirial_t>(&h_energy_virial);
  if (h_setup != NULL)
    deallocate_host<DirectSettings_t>(&h_setup);

  if (vdwtype14 != NULL)
    deallocate<int>(&vdwtype14);
  if (glo_vdwtype != NULL)
    deallocate<int>(&glo_vdwtype);

  neighborList.reset();
  if (loc2glo != NULL)
    deallocate<int>(&loc2glo);
}

//
// Copies h_setup -> d_setup
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::update_setup() {
  updateDirectForceSetup(h_setup);
}

//
// Sets parameters for the nonbonded computation
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setup(double boxx, double boxy, double boxz,
                                       double kappa, double roff, double ron,
                                       double e14fac, int vdw_model,
                                       int elec_model, bool q_p21) {
  double ron2 = ron * ron;
  double ron4 = ron2 * ron2;
  double ron3 = ron * ron * ron;
  double ron5 = ron3 * ron2;
  double ron6 = ron3 * ron3;

  double roff2 = roff * roff;
  double roff3 = roff * roff * roff;
  double roff4 = roff2 * roff2;
  double roff5 = roff4 * roff;
  double roff6 = roff3 * roff3;
  double roff8 = roff6 * roff2;
  double roff12 = roff6 * roff6;
  double roff14 = roff12 * roff2;

  h_setup->boxx = boxx;
  h_setup->boxy = boxy;
  h_setup->boxz = boxz;
  h_setup->kappa = kappa;
  h_setup->kappa2 = kappa * kappa;
  h_setup->roff = roff;
  h_setup->roff2 = roff2;
  h_setup->roff3 = roff3;
  h_setup->roff5 = roff5;
  h_setup->ron = ron;
  h_setup->ron2 = ron2;

  h_setup->roffinv = ((CT)1.0) / roff;
  h_setup->roffinv2 = ((CT)1.0) / roff2;
  h_setup->roffinv3 = ((CT)1.0) / roff3;
  h_setup->roffinv4 = ((CT)1.0) / (roff2 * roff2);
  h_setup->roffinv5 = ((CT)1.0) / (roff * roff2 * roff2);
  h_setup->roffinv6 = ((CT)1.0) / (roff2 * roff2 * roff2);
  h_setup->roffinv12 = h_setup->roffinv6 * h_setup->roffinv6;
  h_setup->roffinv18 = h_setup->roffinv12 * h_setup->roffinv6;

  double roff2_min_ron2 = roff2 - ron2;
  double inv_roff2_ron2_3 =
      1.0 / (roff2_min_ron2 * roff2_min_ron2 * roff2_min_ron2);
  h_setup->inv_roff2_ron2_3 = (CT)inv_roff2_ron2_3;

  // Constants for VFSW
  if (ron < roff) {
    h_setup->k6 = roff3 / (roff3 - ron3);
    h_setup->k12 = roff6 / (roff6 - ron6);
    h_setup->dv6 = -((CT)1.0) / (ron3 * roff3);
    h_setup->dv12 = -((CT)1.0) / (ron6 * roff6);
  } else {
    h_setup->k6 = 1.0f;
    h_setup->k12 = 1.0f;
    h_setup->dv6 = -((CT)1.0) / (roff6);
    h_setup->dv12 = -((CT)1.0) / (roff6 * roff6);
  }

  double g = (roff2 - ron2) * (roff2 - ron2) * (roff2 - ron2);
  h_setup->Aconst = roff4 * (roff2 - 3.0 * ron2) / g;
  h_setup->Bconst = 6.0 * roff2 * ron2 / g;
  h_setup->Cconst = -(ron2 + roff2) / g;
  h_setup->Dconst = 2.0 / (5.0 * g);
  h_setup->dvc = 8.0 * (ron2 * roff2 * (roff - ron) - (roff5 - ron5) / 5.0) / g;
  if (ron < roff) {
    double Denom = 1.0 / g;
    h_setup->Denom = Denom;
    double Acoef = roff4 * (roff2 - 3.0 * ron2) * Denom;
    h_setup->Acoef = Acoef;
    double Bcoef = 6.0 * roff2 * ron2 * Denom;
    h_setup->Bcoef = Bcoef;
    double Ccoef = -3.0 * (ron2 + roff2) * Denom;
    h_setup->Ccoef = Ccoef;
    h_setup->Constr =
        2.0 * Bcoef * log(roff) - Acoef / roff2 + Ccoef * roff2 + Denom * roff4;
    h_setup->Eaddr =
        (12.0 * ron2 * roff2 * log(roff / ron) - 3.0 * (roff4 - ron4)) * Denom;
  } else {
    h_setup->Eaddr = -1.0 / roff2;
  }

  // Constants for Gromacs-style potentials
  // Copy-pasted from Michael G. Lerner code
  h_setup->GAconst = (CT)(5.0 / (3.0 * roff));
  h_setup->GBcoef = (CT)(5.0 / (3.0 * roff2 * roff2));
  double roff_ron = roff - ron;
  double roff_ron2 = roff_ron * roff_ron;
  double roff_ron3 = roff_ron * roff_ron2;
  double roff_ron4 = roff_ron2 * roff_ron2;
  // NOTE: ga6, gb6, gc6 are divided by 6
  //       ga12, gb12, gc12 are divided by 12
  double ga6 = -(10.0 * roff - 7.0 * ron) / (roff8 * roff_ron2);
  double gb6 = (9.0 * roff - 7.0 * ron) / (roff8 * roff_ron3);
  h_setup->ga6 = ga6;
  h_setup->gb6 = gb6;
  h_setup->gc6 =
      (1.0 / roff6) / 6.0 - (ga6 * roff_ron3) / 3.0 - (gb6 * roff_ron4) / 4.0;
  double ga12 = -(16.0 * roff - 13.0 * ron) / (roff14 * roff_ron2);
  double gb12 = (15.0 * roff - 13.0 * ron) / (roff14 * roff_ron3);
  h_setup->ga12 = ga12;
  h_setup->gb12 = gb12;
  h_setup->gc12 = (1.0 / roff12) / 12.0 - (ga12 * roff_ron3) / 3.0 -
                  (gb12 * roff_ron4) / 4.0;

  h_setup->e14fac = e14fac;

  this->vdw_model = vdw_model;
  set_elec_model(elec_model);

  set_calc_vdw(true);
  set_calc_elec(true);

  this->q_p21 = q_p21;
  h_setup->q_p21 = this->q_p21;

  update_setup();
}

//
// Returns parameters for the nonbonded computation
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::get_setup(float &boxx, float &boxy,
                                           float &boxz, float &kappa,
                                           float &roff, float &ron,
                                           float &e14fac, int &vdw_model,
                                           int &elec_model, bool &q_p21) {
  boxx = h_setup->boxx;
  boxy = h_setup->boxy;
  boxz = h_setup->boxz;
  kappa = h_setup->kappa;
  roff = h_setup->roff;
  ron = h_setup->ron;
  e14fac = h_setup->e14fac;
  vdw_model = this->vdw_model;
  elec_model = this->elec_model;
  q_p21 = this->q_p21;
}

//
// Returns box sizes
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::get_box_size(CT &boxx, CT &boxy, CT &boxz) {
  boxx = h_setup->boxx;
  boxy = h_setup->boxy;
  boxz = h_setup->boxz;
}

//
// Sets box sizes
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_box_size(const CT boxx, const CT boxy,
                                              const CT boxz) {
  h_setup->boxx = boxx;
  h_setup->boxy = boxy;
  h_setup->boxz = boxz;
  update_setup();
}

//
// Sets box sizes from box sizes in global memory
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_box_size(const double3 *d_boxSize) {
  double3 *h_boxSize = (double3 *)malloc(sizeof(double3));
  cudaCheck(cudaMemcpy(h_boxSize, d_boxSize, sizeof(double3),
                       cudaMemcpyDeviceToHost));
  h_setup->boxx = h_boxSize->x;
  h_setup->boxy = h_boxSize->y;
  h_setup->boxz = h_boxSize->z;
  update_setup();
  free(h_boxSize);
}

//
// Sets "calc_vdw" flag
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_calc_vdw(const bool calc_vdw) {
  this->calc_vdw = calc_vdw;
}

//
// Sets "calc_elec" flag
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_calc_elec(const bool calc_elec) {
  this->calc_elec = calc_elec;
}

//
// Returns "calc_vdw" flag
//
template <typename AT, typename CT>
bool CudaPMEDirectForce<AT, CT>::get_calc_vdw() {
  return this->calc_vdw;
}

//
// Returns "calc_elec" flag
//
template <typename AT, typename CT>
bool CudaPMEDirectForce<AT, CT>::get_calc_elec() {
  return this->calc_elec;
}

//
// Unbinds textures if they're bound
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::clearTextures() {
#ifdef USE_TEXTURE_OBJECTS
  if (vdwParamTexObjActive) {
    cudaCheck(cudaDestroyTextureObject(vdwParamTexObjActive));
    vdwParamTexObjActive = false;
  }
  if (vdwParam14TexObjActive) {
    cudaCheck(cudaDestroyTextureObject(vdwParam14TexObjActive));
    vdwParam14TexObjActive = false;
  }
#else
  if (get_vdwparam_texref_bound()) {
    cudaCheck(cudaUnbindTexture(*get_vdwparam_texref()));
    set_vdwparam_texref_bound(false);
  }
  if (get_vdwparam14_texref_bound()) {
    cudaCheck(cudaUnbindTexture(*get_vdwparam14_texref()));
    set_vdwparam14_texref_bound(false);
  }
#endif
}

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setup_vdwparam(const int type,
                                                const int h_nvdwparam,
                                                const CT *h_vdwparam) {
  assert(type == VDW_MAIN || type == VDW_IN14);

  int *nvdwparam_loc;
  int *vdwparam_len_loc;
  CT **vdwparam_loc;

  if (type == VDW_MAIN) {
    nvdwparam_loc = &this->nvdwparam;
    vdwparam_len_loc = &this->vdwparam_len;
    vdwparam_loc = &this->vdwparam;
  } else {
    nvdwparam_loc = &this->nvdwparam14;
    vdwparam_len_loc = &this->vdwparam14_len;
    vdwparam_loc = &this->vdwparam14;
  }

  *nvdwparam_loc = h_nvdwparam;

  // "Fix" vdwparam by multiplying c6 by 6.0 and c12 by 12.0
  // NOTE: this is done in order to avoid the multiplication in the inner loop
  CT *h_vdwparam_fixed = new CT[*nvdwparam_loc];
  for (int i = 0; i < *nvdwparam_loc / 2; i++) {
    h_vdwparam_fixed[i * 2] = ((CT)6.0) * h_vdwparam[i * 2];
    h_vdwparam_fixed[i * 2 + 1] = ((CT)12.0) * h_vdwparam[i * 2 + 1];
  }

  bool vdwparam_reallocated = false;
  if (*nvdwparam_loc > *vdwparam_len_loc) {
    reallocate<CT>(vdwparam_loc, vdwparam_len_loc, *nvdwparam_loc, 1.0f);
    vdwparam_reallocated = true;
  }
  copy_HtoD_sync<CT>(h_vdwparam_fixed, *vdwparam_loc, *nvdwparam_loc);
  delete[] h_vdwparam_fixed;

  const bool *use_tex_vdwparam_loc =
      (type == VDW_MAIN) ? &this->use_tex_vdwparam : &this->use_tex_vdwparam14;
#ifdef USE_TEXTURE_OBJECTS
  bool *texActive =
      (type == VDW_MAIN) ? &vdwParamTexObjActive : &vdwParam14TexObjActive;
  cudaTextureObject_t *tex =
      (type == VDW_MAIN) ? &vdwParamTexObj : &vdwParam14TexObj;
#else
  // bool* vdwparam_texref_bound_loc =
  //  (type == VDW_MAIN) ? get_vdwparam_texref_bound() :
  //  get_vdwparam14_texref_bound();
  texture<float2, 1, cudaReadModeElementType> *vdwparam_texref_loc =
      (type == VDW_MAIN) ? get_vdwparam_texref() : get_vdwparam14_texref();
#endif

  if (*use_tex_vdwparam_loc && vdwparam_reallocated) {
#ifdef USE_TEXTURE_OBJECTS
    if (*texActive) {
      cudaDestroyTextureObject(*tex);
      *texActive = false;
    }
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = *vdwparam_loc;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = sizeof(CT) * 8;
    resDesc.res.linear.desc.y = sizeof(CT) * 8;
    resDesc.res.linear.sizeInBytes = (*nvdwparam_loc) * sizeof(CT);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    cudaCheck(cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL));
    *texActive = true;
#else
    // Unbind texture
    if ((type == VDW_MAIN) ? get_vdwparam_texref_bound()
                           : get_vdwparam14_texref_bound()) {
      cudaCheck(cudaUnbindTexture(*vdwparam_texref_loc));
      (type == VDW_MAIN) ? set_vdwparam_texref_bound(false)
                         : set_vdwparam14_texref_bound(false);
    }
    // Bind texture
    memset(vdwparam_texref_loc, 0, sizeof(*vdwparam_texref_loc));
    vdwparam_texref_loc->normalized = 0;
    vdwparam_texref_loc->filterMode = cudaFilterModePoint;
    vdwparam_texref_loc->addressMode[0] = cudaAddressModeClamp;
    vdwparam_texref_loc->channelDesc.x = sizeof(CT) * 8;
    vdwparam_texref_loc->channelDesc.y = sizeof(CT) * 8;
    vdwparam_texref_loc->channelDesc.z = 0;
    vdwparam_texref_loc->channelDesc.w = 0;
    vdwparam_texref_loc->channelDesc.f = cudaChannelFormatKindFloat;
    cudaCheck(cudaBindTexture(NULL, *vdwparam_texref_loc, *vdwparam_loc,
                              (*nvdwparam_loc) * sizeof(CT)));
    (type == VDW_MAIN) ? set_vdwparam_texref_bound(true)
                       : set_vdwparam14_texref_bound(true);
#endif
  }
}

/*
//
// Loads vdwparam from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::load_vdwparam(const char *filename,
                                               const int nvdwparam,
                                               CT **h_vdwparam) {
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        // Open file
        file.open(filename);
        *h_vdwparam = new float[nvdwparam];
        for (int i = 0; i < nvdwparam; i++) {
            file >> (*h_vdwparam)[i];
        }
        file.close();
    } catch (std::ifstream::failure e) {
        std::cerr << "Error opening/reading/closing file " << filename
                  << std::endl;
        exit(1);
    }
}
*/

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam(const int nvdwparam,
                                              const CT *h_vdwparam) {
  setup_vdwparam(VDW_MAIN, nvdwparam, h_vdwparam);
}

//
// Sets VdW parameters by loading them from a file
//
/*template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam(const int nvdwparam,
                                              const char *filename) {
    CT *h_vdwparam;
    load_vdwparam(filename, nvdwparam, &h_vdwparam);
    setup_vdwparam(VDW_MAIN, nvdwparam, h_vdwparam);
    delete[] h_vdwparam;
}
*/

//
// Sets VdW parameters from a vector
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam(const std::vector<CT> vdwparam) {
  setup_vdwparam(VDW_MAIN, vdwparam.size(), vdwparam.data());
}

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam14(const int nvdwparam,
                                                const CT *h_vdwparam) {
  setup_vdwparam(VDW_IN14, nvdwparam, h_vdwparam);
}

/*
//
// Sets VdW parameters by loading them from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam14(const int nvdwparam,
                                                const char *filename) {
    CT *h_vdwparam;
    load_vdwparam(filename, nvdwparam, &h_vdwparam);
    setup_vdwparam(VDW_IN14, nvdwparam, h_vdwparam);
    delete[] h_vdwparam;
}
*/

//
// Sets VdW 14 parameters from a vector
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam14(
    const std::vector<CT> vdwparam) {
  setup_vdwparam(VDW_IN14, vdwparam.size(), vdwparam.data());
}

//
// Sets vdwtype array from global list in device memory memory
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype(const int ncoord,
                                             const int *glo_vdwtype,
                                             const int *loc2glo,
                                             cudaStream_t stream) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord - 1) / warpsize + 1) * warpsize;
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);

  int nthread = 512;
  int nblock = (ncoord - 1) / nthread + 1;
  set_vdwtype_kernel<<<nblock, nthread, 0, stream>>>(ncoord, glo_vdwtype,
                                                     loc2glo, vdwtype);
  cudaCheck(cudaGetLastError());
}

//
// Sets vdwtype array from itself with ind_sorted mapping
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_sort_vdwtype(const int ncoord,
                                                  const int *ind_sorted,
                                                  cudaStream_t stream) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord - 1) / warpsize + 1) * warpsize;
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);
  reallocate<int>(&vdwtypeTemp, &vdwtype_len, ncoord_aligned, 1.2f);

  int nthread = 512;
  int nblock = (ncoord - 1) / nthread + 1;
  set_sorted_vdwtype_kernel<<<nblock, nthread, 0, stream>>>(
      ncoord, vdwtypeTemp, ind_sorted, vdwtype);
  cudaCheck(cudaGetLastError());
}

//
// Sets vdwtype array from host memory
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype(const int ncoord,
                                             const int *h_vdwtype) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord - 1) / warpsize + 1) * warpsize;
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);
  copy_HtoD_sync<int>(h_vdwtype, vdwtype, ncoord);
}

//
// Sets vdwtype14 array from host memory
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype14(const int ncoord,
                                               const int *h_vdwtype) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord - 1) / warpsize + 1) * warpsize;
  reallocate<int>(&vdwtype14, &vdwtype_len, ncoord_aligned, 1.2f);
  copy_HtoD_sync<int>(h_vdwtype, vdwtype14, ncoord);
}

//
// Sets vdwtype array from host memory vector
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype(
    const std::vector<int> &vdwTypeVec) {
  int ncoord = vdwTypeVec.size();
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord - 1) / warpsize + 1) * warpsize;
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);
  // copy_HtoD_sync<int>(h_vdwtype, vdwtype, ncoord);
  copy_HtoD_sync<int>(vdwTypeVec.data(), vdwtype, ncoord);

  // global mappings
  // int ncoord_aligned = ((numAtoms - 1) / warpsize + 1) * warpsize;
  // int vdwtype_len = numAtoms;
  reallocate<int>(&glo_vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);
  copy_HtoD_sync<int>(vdwTypeVec.data(), glo_vdwtype, ncoord);
}

//
// Sets vdwtype14 array from host memory vector
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype14(
    const std::vector<int> &vdwTypeVec) {
  int ncoord = vdwTypeVec.size();
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord - 1) / warpsize + 1) * warpsize;
  reallocate<int>(&vdwtype14, &vdwtype_len, ncoord_aligned, 1.2f);
  // copy_HtoD_sync<int>(h_vdwtype, vdwtype, ncoord);
  copy_HtoD_sync<int>(vdwTypeVec.data(), vdwtype14, ncoord);
}

/*
//
// Sets vdwtype array by loading it from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype(const int ncoord,
                                             const char *filename) {
    int *h_vdwtype;

    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        // Open file
        file.open(filename);

        h_vdwtype = new int[ncoord];

        for (int i = 0; i < ncoord; i++) {
            file >> h_vdwtype[i];
        }

        file.close();
    } catch (std::ifstream::failure e) {
        std::cerr << "Error opening/reading/closing file " << filename
                  << std::endl;
        exit(1);
    }

    set_vdwtype(ncoord, h_vdwtype);

    delete[] h_vdwtype;
}

//
// Sets vdwtype14 array by loading it from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype14(const int ncoord,
                                               const char *filename) {
    int *h_vdwtype;

    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        // Open file
        file.open(filename);

        h_vdwtype = new int[ncoord];

        for (int i = 0; i < ncoord; i++) {
            file >> h_vdwtype[i];
        }

        file.close();
    } catch (std::ifstream::failure e) {
        std::cerr << "Error opening/reading/closing file " << filename
                  << std::endl;
        exit(1);
    }

    set_vdwtype14(ncoord, h_vdwtype);

    delete[] h_vdwtype;
}
*/

//
// Resets vdwtype after sorting
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_sort_vdwtype(const int ncoord,
                                                  const int *ind_sorted) {
  /*int *h_vdwtype;
  h_vdwtype = new int[ncoord];
  for (int i=0; i < ncoord; ++i){
    h_vdwtype[i] = ind_sorted[i]
  }
  */
}

//
// Sets 1-4 interaction and exclusion lists
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_14_list(int nin14list, int nex14list,
                                             xx14list_t *h_in14list,
                                             xx14list_t *h_ex14list,
                                             cudaStream_t stream) {
  this->nin14list = nin14list;
  this->nex14list = nex14list;

  if (nin14list > 0) {
    reallocate<xx14list_t>(&in14list, &in14list_len, nin14list);
    copy_HtoD<xx14list_t>(h_in14list, in14list, nin14list, stream);
  }

  if (nex14list > 0) {
    reallocate<xx14list_t>(&ex14list, &ex14list_len, nex14list);
    copy_HtoD<xx14list_t>(h_ex14list, ex14list, nex14list, stream);
  }
}

//
// Setup 1-3 interaction and exclusion lists from file input
//

template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_14_list(std::string sizeFile,
                                             std::string valFile,
                                             cudaStream_t stream) {
  std::ifstream file;
  xx14list_t *h_in14list;
  xx14list_t *h_ex14list;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(sizeFile);
    file >> nin14list;
    file >> nex14list;
    file.close();
  } catch (std::ifstream::failure e) {
    // std::cerr << "Error opening/reading/closing file " << sizeFile <<
    // std::endl;
    throw std::invalid_argument("Error opening/reading/closing file " +
                                sizeFile + "\n");
    exit(1);
  }
  h_in14list = new xx14list_t[nin14list];
  h_ex14list = new xx14list_t[nex14list];

  try {
    // Open file
    file.open(valFile);
    for (int i = 0; i < nin14list; i++) {
      file >> h_in14list[i].i >> h_in14list[i].j;
      h_in14list[i].ishift = 13;
    }
    for (int i = 0; i < nex14list; i++) {
      file >> h_ex14list[i].i >> h_ex14list[i].j;
      h_ex14list[i].ishift = 13;
      // std::cout << i << " "<< h_ex14list[i].i << " " << h_ex14list[i].j
      // <<  "\n";
    }
    file.close();
  } catch (std::ifstream::failure e) {
    // std::cerr << "Error opening/reading/closing file " << valFile <<
    // std::endl;
    throw std::invalid_argument("Error opening/reading/closing file " +
                                valFile + "\n");
    exit(1);
  }

  if (nin14list > 0) {
    reallocate<xx14list_t>(&in14list, &in14list_len, nin14list);
    copy_HtoD<xx14list_t>(h_in14list, in14list, nin14list, stream);
  }

  if (nex14list > 0) {
    reallocate<xx14list_t>(&ex14list, &ex14list_len, nex14list);
    copy_HtoD<xx14list_t>(h_ex14list, ex14list, nex14list, stream);
  }
  delete h_in14list;
  delete h_ex14list;
}

//
// Setup 1-3 interaction and exclusion lists from vectors
//

template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_14_list(const std::vector<int> &size,
                                             const std::vector<int> &val,
                                             cudaStream_t stream) {
  nin14list = size[0];
  nex14list = size[1];

  xx14list_t *h_in14list;
  xx14list_t *h_ex14list;
  h_in14list = new xx14list_t[nin14list];
  h_ex14list = new xx14list_t[nex14list];
  for (int i = 0; i < nin14list; i++) {
    h_in14list[i].i = val[2 * i];
    h_in14list[i].j = val[2 * i + 1];
    h_in14list[i].ishift = 13;
  }
  for (int i = 0; i < nex14list; i++) {
    h_ex14list[i].i = val[2 * nin14list + 2 * i];
    h_ex14list[i].j = val[2 * nin14list + 2 * i + 1];
    h_ex14list[i].ishift = 13;
  }
  if (nin14list > 0) {
    reallocate<xx14list_t>(&in14list, &in14list_len, nin14list);
    copy_HtoD<xx14list_t>(h_in14list, in14list, nin14list, stream);
  }

  if (nex14list > 0) {
    reallocate<xx14list_t>(&ex14list, &ex14list_len, nex14list);
    copy_HtoD<xx14list_t>(h_ex14list, ex14list, nex14list, stream);
  }
}
//
// Setup 1-4 interaction and exclusion lists from device memory using global
// data:
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_14_list(
    const float4 *xyzq, const float boxx, const float boxy, const float boxz,
    const int *glo2loc_ind, const int nin14_tbl, const int *in14_tbl,
    const xx14_t *in14, const int nex14_tbl, const int *ex14_tbl,
    const xx14_t *ex14, cudaStream_t stream) {
  this->nin14list = nin14_tbl;
  if (nin14list > 0)
    reallocate<xx14list_t>(&in14list, &in14list_len, nin14list, 1.2f);

  this->nex14list = nex14_tbl;
  if (nex14list > 0)
    reallocate<xx14list_t>(&ex14list, &ex14list_len, nex14list, 1.2f);

  float3 half_box;
  half_box.x = boxx * 0.5f;
  half_box.y = boxy * 0.5f;
  half_box.z = boxz * 0.5f;

  int nthread = 512;
  int nblock = (nin14_tbl + nex14_tbl - 1) / nthread + 1;
  set_14_list_kernel<<<nblock, nthread, 0, stream>>>(
      nin14_tbl, in14_tbl, in14, in14list, nex14_tbl, ex14_tbl, ex14, ex14list,
      xyzq, half_box, glo2loc_ind);
  cudaCheck(cudaGetLastError());
}

//
// Builds Ewald lookup table
// roff  = distance cut-off
// h     = the distance between interpolation points
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setup_ewald_force(CT h) {
  h_setup->hinv = ((CT)1.0) / h;

  n_ewald_force = (int)(sqrt(h_setup->roff2) * h_setup->hinv) + 2;

  CT *h_ewald_force = new CT[n_ewald_force];

  for (int i = 1; i < n_ewald_force; i++) {
    const CT two_sqrtpi = (CT)1.12837916709551; // 2/sqrt(pi)
    CT r = i * h;
    CT r2 = r * r;
    h_ewald_force[i] =
        two_sqrtpi * ((CT)h_setup->kappa) * exp(-((CT)h_setup->kappa2) * r2) +
        erfc(((CT)h_setup->kappa) * r) / r;
  }
  h_ewald_force[0] = h_ewald_force[1];

  allocate<CT>(&ewald_force, n_ewald_force);
  copy_HtoD_sync<CT>(h_ewald_force, ewald_force, n_ewald_force);

  h_setup->ewald_force = ewald_force;

  delete[] h_ewald_force;
}

//
// Sets method for calculating electrostatic force and energy
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_elec_model(int elec_model, CT h) {
  this->elec_model = elec_model;

  if (elec_model == EWALD_LOOKUP) {
    setup_ewald_force(h);
  }
}

//
// Calculates 1-4 exclusions and interactions
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::calc_14_force(const float4 *xyzq,
                                               const bool calc_energy,
                                               const bool calc_virial,
                                               const int stride, AT *force,
                                               cudaStream_t stream) {
  if (use_tex_vdwparam14) {
#ifdef USE_TEXTURE_OBJECTS
    if (!vdwParam14TexObjActive) {
      // std::cerr << "CudaPMEDirectForce<AT, CT>::calc_14_force, "
      //              "vdwParam14TexObj must be created"
      //           << std::endl;
      std::string invalar;
      invalar = "CudaPMEDirectForce<AT, CT>::calc_14_force, vdwParam14TexObj "
                "must be created\n";
      throw std::invalid_argument(invalar);
      exit(1);
    }
#else
    if (!get_vdwparam14_texref_bound()) {
      // std::cerr << "CudaPMEDirectForce<AT, CT>::calc_14_force, "
      //              "vdwparam14_texref must be bound"
      //           << std::endl;
      throw std::invalid_argument("CudaPMEDirectForce<AT, CT>::calc_14_force, "
                                  "vdwparam14_texref must be bound\n");
      exit(1);
    }
#endif
  }

  // int nthread = 512;
  int nthread = 128;
  int nin14block = (nin14list + nthread - 1) / nthread;
  int nex14block = (nex14list + nthread - 1) / nthread;
  int nblock = nin14block + nex14block;
  int shmem_size = 0;
  if (calc_energy) {
    shmem_size =
        nthread *
        sizeof(double3); // only need double2 here. Compile can't seem to
                         // handle distinction, (in
                         // CudaPMEDirectForceBlock.cu), see note by shpot
                         // declaration in CudaDirectForce14_util.h
  }

  int vdw_model_loc = calc_vdw ? vdw_model : NONE;
  int elec_model_loc = calc_elec ? elec_model : NONE;
  if (elec_model_loc == NONE && vdw_model_loc == NONE)
    return;

  calcForce14KernelChoice<AT, CT>(
      nblock, nthread, shmem_size, stream, vdw_model_loc, elec_model_loc,
      calc_energy, calc_virial, this->nin14list, this->in14list,
      this->nex14list, this->ex14list, nin14block, (int *)NULL, (int *)NULL,
      (int *)NULL, (int *)NULL, (int *)NULL, (int *)NULL,
      // this->vdwtype, this->vdwparam14,
      this->vdwtype14, this->vdwparam14,
#ifdef USE_TEXTURE_OBJECTS
      this->vdwParam14TexObj,
#endif
      xyzq, 1.0f, stride, force, energyVirial.getVirialPointer(),
      energyVirial.getEnergyPointer(strVdw),
      energyVirial.getEnergyPointer(strElec),
      energyVirial.getEnergyPointer(strExcl), (CudaBlock *)NULL);
}

__global__ void test_vdwparams(const int *vdwtype, const float *vdwparam) {

  for (int i = 0; i < 2500; ++i) {
    for (int j = 0; j < i; ++j) {
      int ia = vdwtype[i];
      int ja = vdwtype[j];

      // if ( !(ia == 26 || ja == 26 || ia == 38 || ja == 38)){
      int aa = max(ia, ja);
      int ivdw = aa * (aa - 1) + 2 * (ia + ja);
      float c6 = vdwparam[ivdw];
      float c12 = vdwparam[ivdw + 1];
      printf("%d\t%d\t%f\t%f\n", i, j, c6, c12);
      //}
    }
  }
}

float distance(const float4 *xyzq, int i, int j) {
  float4 xyzqi = xyzq[i];
  float4 xyzqj = xyzq[j];
  float dx = xyzqi.x - xyzqj.x;
  float dy = xyzqi.y - xyzqj.y;
  float dz = xyzqi.z - xyzqj.z;

  return dx * dx + dy * dy + dz * dz;
}

void calc_vdw_cpu(const CudaNeighborListBuild<32> &nlist, int nvdwparam,
                  const int *vdwtype, const float *vdwparam,
                  const float4 *xyzq) {
  // Only for test
  std::ofstream out;
  out.open("vdwparamtest.out");
  ientry_t *h_ientry = new ientry_t[nlist.get_n_ientry()];
  int *h_tile_indj = new int[nlist.get_n_tile()];
  int *h_vdwtype = new int[2500];
  float *h_vdwparam = new float[nvdwparam];
  float4 *h_xyzq = new float4[2500];
  copy_DtoH_T(nlist.get_ientry(), h_ientry, nlist.get_n_ientry(),
              sizeof(ientry_t));
  copy_DtoH_T(nlist.get_tile_indj(), h_tile_indj, nlist.get_n_tile(),
              sizeof(int));
  copy_DtoH_T(vdwtype, h_vdwtype, 2500, sizeof(int));
  copy_DtoH_T(vdwparam, h_vdwparam, nvdwparam, sizeof(float));
  copy_DtoH_T(xyzq, h_xyzq, 2500, sizeof(float4));
  cudaDeviceSynchronize();

  for (int i = 0; i < nlist.get_n_ientry(); ++i) {
    int indi = h_ientry[i].iatomStart;
    int startj = h_ientry[i].tileStart;
    int endj = h_ientry[i].tileEnd;

    for (int jtile = startj; jtile <= endj; ++jtile) {
      int indj = h_tile_indj[jtile];
      for (int it = 0; it < 32; ++it) {
        for (int jt = 0; jt < 32; ++jt) {
          int iatom = indi + it;
          int jatom = indj + jt;
          int ia = h_vdwtype[iatom];
          int ja = h_vdwtype[jatom];
          int aa = std::max(ia, ja);
          int ivdw = aa * (aa - 1) + 2 * (ia + ja);
          // if (distance(h_xyzq, iatom, jatom) <  100.0){
          out << iatom << " " << jatom << " " << h_vdwparam[ivdw] << " "
              << h_vdwparam[ivdw + 1] << "\n";
          // std::this_thread::sleep_for(std::chrono::microseconds(200));
          //}
        }
      }
    }
  }
  out.close();
  // std::cout << indi <<  " "  << h_tile_indj[startj] << " " <<
  // h_tile_indj[endj]  << "\n";

  delete h_ientry;
  delete h_tile_indj;
  delete h_vdwtype;
  delete h_vdwparam;
  delete h_xyzq;

  // only for test
}
// Calculates direct force
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::calc_force(
    const int whichlist, const float4 *xyzq,
    const CudaNeighborListBuild<32> &nlist, const bool calc_energy,
    const bool calc_virial, const int stride, AT *force, cudaStream_t stream) {

  // test_vdwparams<<<1,1>>>(vdwtype, vdwparam);
  // calc_vdw_cpu(nlist, nvdwparam, vdwtype, vdwparam, xyzq);
  if (use_tex_vdwparam) {
#ifdef USE_TEXTURE_OBJECTS
    if (!vdwParamTexObjActive) {
      // std::cerr << "CudaPMEDirectForce<AT, CT>::calc_force, "
      //              "vdwParamTexObj must be created"
      //           << std::endl;
      throw std::invalid_argument("CudaPMEDirectForce<AT, CT>::calc_force, "
                                  "vdwParamTexObj must be created\n");
      exit(1);
    }
#else
    if (!get_vdwparam_texref_bound()) {
      // std::cerr << "CudaPMEDirectForce<AT, CT>::calc_force, "
      //              "vdwparam_texref must be bound"
      //           << std::endl;
      throw std::invalid_argument("CudaPMEDirectForce<AT, CT>::calc_force, " +
                                  "vdwparam_texref must be bound" + "\n");
      exit(1);
    }
#endif
  }

  if (nlist.get_n_ientry() == 0)
    return;
  int vdw_model_loc = calc_vdw ? vdw_model : NONE;
  int elec_model_loc = calc_elec ? elec_model : NONE;
  if (elec_model_loc == NONE && vdw_model_loc == NONE)
    return;

  int nwarp = 2;
  if (get_cuda_arch() < 300) {
    nwarp = 2;
  } else {
    nwarp = 4;
  }
  nwarp = 8;
  int nthread = warpsize * nwarp;
  int nblock_tot = (nlist.get_n_ientry() - 1) / (nthread / warpsize) + 1;

  int shmem_size = 0;
  // (sh_xi, sh_yi, sh_zi, sh_qi, sh_vdwtypei)
  if (get_cuda_arch() < 300)
    shmem_size +=
        (nthread / warpsize) * tilesize * (sizeof(float) * 4 + sizeof(int));
  // (sh_fix, sh_fiy, sh_fiz)
  shmem_size += (nthread / warpsize) * warpsize * sizeof(AT) * 3;
  // If no texture fetch for vdwparam:
  if (!use_tex_vdwparam)
    shmem_size += nvdwparam * sizeof(float);

  if (calc_energy)
    shmem_size = max(shmem_size, (int)(nthread * sizeof(double) * 2));
  if (calc_virial)
    shmem_size = max(shmem_size, (int)(nthread * sizeof(double) * 3));
  //

  calcForceKernelChoice<AT, CT>(
      nblock_tot, nthread, shmem_size, stream, vdw_model_loc, elec_model_loc,
      calc_energy, calc_virial, nlist, this->vdwparam, this->nvdwparam,
      this->vdwtype,
#ifdef USE_TEXTURE_OBJECTS
      this->vdwParamTexObj,
#endif
      xyzq, stride, force, energyVirial.getVirialPointer(),
      energyVirial.getEnergyPointer(strVdw),
      energyVirial.getEnergyPointer(strElec));
}

template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setupSorted(int numAtoms) {
  xyzq_sorted.set_ncoord(numAtoms);
}

template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setupTopologicalExclusions(
    int numAtoms, std::vector<int> &iblo14, std::vector<int> &inb14) {
  topExcl.setFromVector(numAtoms, iblo14, inb14);
}

template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setupNeighborList(int numAtoms) {
  // TODO : assert that topExcl has been created
  neighborList = std::make_shared<CudaNeighborList<32>>(topExcl, 1, 1, 1);
  cudaCheck(cudaMalloc(&loc2glo, numAtoms * sizeof(int)));

  std::vector<int> numIntZones(8, 0);
  std::vector<std::vector<int>> intZones(8, std::vector<int>());
  numIntZones.at(0) = 1;
  intZones.at(0).push_back(0);

  if (q_p21)
    neighborList->setPBC(PBC::P21);
  else
    neighborList->setPBC(PBC::P1);

  neighborList->registerList(numIntZones, intZones);
  // nlist[0] = new CudaNeighborListBuild<32>(0, 0, 0);
  // nlist[1] = new CudaNeighborListBuild<32>(0, 0, 0);
  zone_patom[0] = 0;
  for (int i = 1; i < 9; i++)
    zone_patom[i] = numAtoms;
}

__global__ void set_loc2glo(int *array, int size) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size)
    array[id] = id;
}

template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::resetNeighborList(const float4 *xyzq,
                                                   int numAtoms) {
  int numThreads = 512;
  int numBlocks = (numAtoms - 1) / numThreads + 1;
  set_loc2glo<<<numBlocks, numThreads, 0, *directStream>>>(loc2glo, numAtoms);
  // cudaStreamSynchronize(*directStream);
  neighborList->sort(0, zone_patom, xyzq, xyzq_sorted.xyzq, loc2glo,
                     *directStream);
  // cudaStreamSynchronize(*directStream);

  set_vdwtype(numAtoms, glo_vdwtype, neighborList->get_ind_sorted(),
              *directStream);
  // cudaCheck(cudaStreamSynchronize(*directStream));

  neighborList->build(0, zone_patom, boxDimensions[0], boxDimensions[1],
                      boxDimensions[2], cutoff, xyzq_sorted.xyzq, loc2glo,
                      *directStream);
  cudaStreamSynchronize(*directStream);
  // std::cout << "Prepared nlist.\n";
}

template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setBoxDimensions(std::vector<double> dim) {
  boxDimensions = dim;
  set_box_size(dim[0], dim[1], dim[2]);
}

__global__ void addSortedForce(const int *loc2glo, int size, int stride,
                               const long long int *__restrict__ sortedVal,
                               long long int *__restrict__ val) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    // val[id] += sortedVal[loc2glo[id]];
    int j = loc2glo[i];
    val[j] += sortedVal[i];
    val[j + stride] += sortedVal[i + stride];
    val[j + 2 * stride] += sortedVal[i + 2 * stride];
  }
}

// TODO : testing for now
__global__ void setSortedCoords(int numAtoms, const int *loc2glo,
                                const float4 *__restrict__ xyzq,
                                float4 *__restrict__ xyzq_sorted) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numAtoms) {
    int j = loc2glo[i];
    xyzq_sorted[i].x = xyzq[j].x;
    xyzq_sorted[i].y = xyzq[j].y;
    xyzq_sorted[i].z = xyzq[j].z;
    xyzq_sorted[i].w = xyzq[j].w;
  }
}

template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::calc_force(const float4 *xyzq, bool calcEnergy,
                                            bool calcVirial) {

  int numThreads = 512;
  int numBlocks = (numAtoms - 1) / numThreads + 1;
  setSortedCoords<<<numBlocks, numThreads, 0, *directStream>>>(
      numAtoms, loc2glo, xyzq, xyzq_sorted.xyzq);

  forceSortedVal->clear(*directStream);

  nvtxRangePushA("direct space force");
  calc_force(0, xyzq_sorted.xyzq, neighborList->getBuilder(0), calcEnergy,
             calcVirial, forceSortedVal->stride(), forceSortedVal->xyz(),
             *directStream);

  calc_14_force(xyzq, calcEnergy, calcVirial, forceVal->stride(),
                forceVal->xyz(), *directStream);
  nvtxRangePop();

  // TODO : add the forceSortedVal in the right order to forceVal

  addSortedForce<<<numBlocks, numThreads, 0, *directStream>>>(
      loc2glo, numAtoms, forceVal->stride(), forceSortedVal->xyz(),
      forceVal->xyz());
}

//
// Explicit instances of CudaPMEDirectForce
//
template class CudaPMEDirectForce<long long int, float>;
#endif // NOCUDAC
