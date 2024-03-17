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
#include "Bspline.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cassert>
#include <cuda.h>
#include <iostream>
#include <math.h>

template <typename T>
__global__ void
fill_bspline_4(const float4 *xyzq, const int ncoord, const float *recip,
               const int nfftx, const int nffty, const int nfftz, int *gix,
               int *giy, int *giz, float *charge, float *thetax, float *thetay,
               float *thetaz, float *dthetax, float *dthetay, float *dthetaz) {
  // Position to xyzq and atomgrid
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < ncoord) {
    float4 xyzqi = xyzq[pos];
    float x = xyzqi.x;
    float y = xyzqi.y;
    float z = xyzqi.z;
    float q = xyzqi.w;

    float w;
    // NOTE: I don't think we need the +2.0f here..
    w = x * recip[0] + y * recip[1] + z * recip[2] + 2.0f;
    float frx = (float)(nfftx * (w - (floorf(w + 0.5f) - 0.5f)));

    w = x * recip[3] + y * recip[4] + z * recip[5] + 2.0f;
    float fry = (float)(nffty * (w - (floorf(w + 0.5f) - 0.5f)));

    w = x * recip[6] + y * recip[7] + z * recip[8] + 2.0f;
    float frz = (float)(nfftz * (w - (floorf(w + 0.5f) - 0.5f)));

    int frxi = (int)(frx);
    int fryi = (int)(fry);
    int frzi = (int)(frz);

    float wx = frx - (float)frxi;
    float wy = fry - (float)fryi;
    float wz = frz - (float)frzi;

    gix[pos] = frxi;
    giy[pos] = fryi;
    giz[pos] = frzi;
    charge[pos] = q;

    float3 theta_tmp[4];
    float3 dtheta_tmp[4];

    theta_tmp[3].x = 0.0f;
    theta_tmp[3].y = 0.0f;
    theta_tmp[3].z = 0.0f;
    theta_tmp[1].x = wx;
    theta_tmp[1].y = wy;
    theta_tmp[1].z = wz;
    theta_tmp[0].x = 1.0f - wx;
    theta_tmp[0].y = 1.0f - wy;
    theta_tmp[0].z = 1.0f - wz;

    // compute standard b-spline recursion
    theta_tmp[2].x = 0.5f * wx * theta_tmp[1].x;
    theta_tmp[2].y = 0.5f * wy * theta_tmp[1].y;
    theta_tmp[2].z = 0.5f * wz * theta_tmp[1].z;

    theta_tmp[1].x =
        0.5f * ((wx + 1.0f) * theta_tmp[0].x + (2.0f - wx) * theta_tmp[1].x);
    theta_tmp[1].y =
        0.5f * ((wy + 1.0f) * theta_tmp[0].y + (2.0f - wy) * theta_tmp[1].y);
    theta_tmp[1].z =
        0.5f * ((wz + 1.0f) * theta_tmp[0].z + (2.0f - wz) * theta_tmp[1].z);

    theta_tmp[0].x = 0.5f * (1.0f - wx) * theta_tmp[0].x;
    theta_tmp[0].y = 0.5f * (1.0f - wy) * theta_tmp[0].y;
    theta_tmp[0].z = 0.5f * (1.0f - wz) * theta_tmp[0].z;

    // perform standard b-spline differentiationa
    dtheta_tmp[0].x = -theta_tmp[0].x;
    dtheta_tmp[0].y = -theta_tmp[0].y;
    dtheta_tmp[0].z = -theta_tmp[0].z;

    dtheta_tmp[1].x = theta_tmp[0].x - theta_tmp[1].x;
    dtheta_tmp[1].y = theta_tmp[0].y - theta_tmp[1].y;
    dtheta_tmp[1].z = theta_tmp[0].z - theta_tmp[1].z;

    dtheta_tmp[2].x = theta_tmp[1].x - theta_tmp[2].x;
    dtheta_tmp[2].y = theta_tmp[1].y - theta_tmp[2].y;
    dtheta_tmp[2].z = theta_tmp[1].z - theta_tmp[2].z;

    dtheta_tmp[3].x = theta_tmp[2].x - theta_tmp[3].x;
    dtheta_tmp[3].y = theta_tmp[2].y - theta_tmp[3].y;
    dtheta_tmp[3].z = theta_tmp[2].z - theta_tmp[3].z;

    // one more recursion
    theta_tmp[3].x = (1.0f / 3.0f) * wx * theta_tmp[2].x;
    theta_tmp[3].y = (1.0f / 3.0f) * wy * theta_tmp[2].y;
    theta_tmp[3].z = (1.0f / 3.0f) * wz * theta_tmp[2].z;

    theta_tmp[2].x = (1.0f / 3.0f) * ((wx + 1.0f) * theta_tmp[1].x +
                                      (3.0f - wx) * theta_tmp[2].x);
    theta_tmp[2].y = (1.0f / 3.0f) * ((wy + 1.0f) * theta_tmp[1].y +
                                      (3.0f - wy) * theta_tmp[2].y);
    theta_tmp[2].z = (1.0f / 3.0f) * ((wz + 1.0f) * theta_tmp[1].z +
                                      (3.0f - wz) * theta_tmp[2].z);

    theta_tmp[1].x = (1.0f / 3.0f) * ((wx + 2.0f) * theta_tmp[0].x +
                                      (2.0f - wx) * theta_tmp[1].x);
    theta_tmp[1].y = (1.0f / 3.0f) * ((wy + 2.0f) * theta_tmp[0].y +
                                      (2.0f - wy) * theta_tmp[1].y);
    theta_tmp[1].z = (1.0f / 3.0f) * ((wz + 2.0f) * theta_tmp[0].z +
                                      (2.0f - wz) * theta_tmp[1].z);

    theta_tmp[0].x = (1.0f / 3.0f) * (1.0f - wx) * theta_tmp[0].x;
    theta_tmp[0].y = (1.0f / 3.0f) * (1.0f - wy) * theta_tmp[0].y;
    theta_tmp[0].z = (1.0f / 3.0f) * (1.0f - wz) * theta_tmp[0].z;

    // Store theta_tmp and dtheta_tmp into global memory
    int pos4 = pos * 4;
    thetax[pos4] = theta_tmp[0].x;
    thetax[pos4 + 1] = theta_tmp[1].x;
    thetax[pos4 + 2] = theta_tmp[2].x;
    thetax[pos4 + 3] = theta_tmp[3].x;

    thetay[pos4] = theta_tmp[0].y;
    thetay[pos4 + 1] = theta_tmp[1].y;
    thetay[pos4 + 2] = theta_tmp[2].y;
    thetay[pos4 + 3] = theta_tmp[3].y;

    thetaz[pos4] = theta_tmp[0].z;
    thetaz[pos4 + 1] = theta_tmp[1].z;
    thetaz[pos4 + 2] = theta_tmp[2].z;
    thetaz[pos4 + 3] = theta_tmp[3].z;

    dthetax[pos4] = dtheta_tmp[0].x;
    dthetax[pos4 + 1] = dtheta_tmp[1].x;
    dthetax[pos4 + 2] = dtheta_tmp[2].x;
    dthetax[pos4 + 3] = dtheta_tmp[3].x;

    dthetay[pos4] = dtheta_tmp[0].y;
    dthetay[pos4 + 1] = dtheta_tmp[1].y;
    dthetay[pos4 + 2] = dtheta_tmp[2].y;
    dthetay[pos4 + 3] = dtheta_tmp[3].y;

    dthetaz[pos4] = dtheta_tmp[0].z;
    dthetaz[pos4 + 1] = dtheta_tmp[1].z;
    dthetaz[pos4 + 2] = dtheta_tmp[2].z;
    dthetaz[pos4 + 3] = dtheta_tmp[3].z;

    pos += blockDim.x * gridDim.x;
  }
}

//
// Bspline class method definitions
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//

template <typename T> void Bspline<T>::set_ncoord(const int ncoord) {
  reallocate<T>(&thetax, &thetax_len, ncoord * order, 1.2f);
  reallocate<T>(&thetay, &thetay_len, ncoord * order, 1.2f);
  reallocate<T>(&thetaz, &thetaz_len, ncoord * order, 1.2f);
  reallocate<T>(&dthetax, &dthetax_len, ncoord * order, 1.2f);
  reallocate<T>(&dthetay, &dthetay_len, ncoord * order, 1.2f);
  reallocate<T>(&dthetaz, &dthetaz_len, ncoord * order, 1.2f);
  reallocate<int>(&gix, &gix_len, ncoord, 1.2f);
  reallocate<int>(&giy, &giy_len, ncoord, 1.2f);
  reallocate<int>(&giz, &giz_len, ncoord, 1.2f);
  reallocate<T>(&charge, &charge_len, ncoord, 1.2f);
}

template <typename T>
Bspline<T>::Bspline(const int ncoord, const int order, const int nfftx,
                    const int nffty, const int nfftz)
    : thetax(NULL), thetay(NULL), thetaz(NULL), dthetax(NULL), dthetay(NULL),
      dthetaz(NULL), gix(NULL), giy(NULL), giz(NULL), charge(NULL),
      order(order), nfftx(nfftx), nffty(nffty), nfftz(nfftz) {
  set_ncoord(ncoord);

  allocate<T>(&recip, 9);
}

template <typename T> Bspline<T>::~Bspline() {
  deallocate<T>(&thetax);
  deallocate<T>(&thetay);
  deallocate<T>(&thetaz);
  deallocate<T>(&dthetax);
  deallocate<T>(&dthetay);
  deallocate<T>(&dthetaz);
  deallocate<int>(&gix);
  deallocate<int>(&giy);
  deallocate<int>(&giz);
  deallocate<T>(&charge);
  deallocate<T>(&recip);
}

template <typename T>
template <typename B>
void Bspline<T>::set_recip(const B *h_recip) {
  T h_recip_T[9];
  for (int i = 0; i < 9; i++)
    h_recip_T[i] = (T)h_recip[i];
  copy_HtoD<T>(h_recip_T, recip, 9);
}

template <typename T>
void Bspline<T>::fill_bspline(const float4 *xyzq, const int ncoord) {
  // Re-allocates (theta, dtheta, gridp) if needed
  set_ncoord(ncoord);

  int nthread = 64;
  int nblock = (ncoord - 1) / nthread + 1;

  /*
  bool ortho = (recip[1] == 0.0 && recip[2] == 0.0 && recip[3] == 0.0 &&
                recip[5] == 0.0 && recip[6] == 0.0 && recip[7] == 0.0);
  */

  switch (order) {
  case 4:
    fill_bspline_4<T><<<nblock, nthread>>>(
        xyzq, ncoord, recip, nfftx, nffty, nfftz, gix, giy, giz, charge, thetax,
        thetay, thetaz, dthetax, dthetay, dthetaz);
    break;
  default:
    //std::cerr << "Bspline::fill_bspline: order != 4 not implemented"
    //          << std::endl;
    throw std::invalid_argument("Bspline::fill_bspline: order != 4 not implemented\n");
    exit(1);
  }

  cudaCheck(cudaGetLastError());
}

//
// Prints a part of (dthetax, dthetay, dthetaz)
//
template <typename T> void Bspline<T>::print_dtheta(int start, int end) {
  assert(start <= end);
  assert(start >= 0);
  assert(end < dthetax_len);
  assert(end < dthetay_len);
  assert(end < dthetaz_len);

  T *h_dthetax = new T[dthetax_len * order];
  T *h_dthetay = new T[dthetay_len * order];
  T *h_dthetaz = new T[dthetaz_len * order];

  copy_DtoH<T>(dthetax, h_dthetax, (end + 1) * order);
  copy_DtoH<T>(dthetay, h_dthetay, (end + 1) * order);
  copy_DtoH<T>(dthetaz, h_dthetaz, (end + 1) * order);
  for (int i = start; i <= end; i++) {
    std::cout << h_dthetax[i] << " " << h_dthetay[i] << " " << h_dthetaz[i]
              << std::endl;
  }

  delete[] h_dthetax;
  delete[] h_dthetay;
  delete[] h_dthetaz;
}

//
// Compares (dthetax, dthetay, dthetaz) between two Bsplines
//
template <typename T>
bool Bspline<T>::compare_dtheta(Bspline &a, int ncoord, double tol) {
  assert(ncoord > 0);
  assert(ncoord <= dthetax_len);
  assert(ncoord <= a.dthetax_len);

  T *h_dthetax1 = new T[ncoord * order];
  T *h_dthetay1 = new T[ncoord * order];
  T *h_dthetaz1 = new T[ncoord * order];

  T *h_dthetax2 = new T[ncoord * order];
  T *h_dthetay2 = new T[ncoord * order];
  T *h_dthetaz2 = new T[ncoord * order];

  copy_DtoH<T>(dthetax, h_dthetax1, ncoord * order);
  copy_DtoH<T>(dthetay, h_dthetay1, ncoord * order);
  copy_DtoH<T>(dthetaz, h_dthetaz1, ncoord * order);

  copy_DtoH<T>(a.dthetax, h_dthetax2, ncoord * order);
  copy_DtoH<T>(a.dthetay, h_dthetay2, ncoord * order);
  copy_DtoH<T>(a.dthetaz, h_dthetaz2, ncoord * order);

  bool ok = true;

  try {
    for (int i = 0; i < ncoord; i++) {
      double dx = fabs(h_dthetax1[i] - h_dthetax2[i]);
      double dy = fabs(h_dthetay1[i] - h_dthetay2[i]);
      double dz = fabs(h_dthetaz1[i] - h_dthetaz2[i]);
      if (dx > tol || dy > tol || dz > tol)
        throw i;
    }
  } catch (int i) {
    std::cout << "compare_dtheta, outside tolerance at i=" << i << std::endl;
    std::cout << "this: " << h_dthetax1[i] << " " << h_dthetay1[i] << " "
              << h_dthetaz1[i] << std::endl;
    std::cout << "comp: " << h_dthetax2[i] << " " << h_dthetay2[i] << " "
              << h_dthetaz2[i] << std::endl;
    ok = false;
  }

  delete[] h_dthetax1;
  delete[] h_dthetay1;
  delete[] h_dthetaz1;

  delete[] h_dthetax2;
  delete[] h_dthetay2;
  delete[] h_dthetaz2;

  return ok;
}

//
// Explicit instances of Bspline
//
template class Bspline<float>;
template void Bspline<float>::set_recip<double>(const double *h_recip);
#endif // NOCUDAC
