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
#ifndef BSPLINE_H
#define BSPLINE_H

template <typename T> class Bspline {
private:
  // Length of the B-spline data arrays
  int thetax_len;
  int thetay_len;
  int thetaz_len;
  int dthetax_len;
  int dthetay_len;
  int dthetaz_len;

  // Size of the FFT
  int nfftx;
  int nffty;
  int nfftz;

  // B-spline order
  int order;

  // Length of the data arrays
  int gix_len;
  int giy_len;
  int giz_len;
  int charge_len;

  // Reciprocal vectors
  T *recip;

public:
  // B-spline data
  T *thetax;
  T *thetay;
  T *thetaz;
  T *dthetax;
  T *dthetay;
  T *dthetaz;

  // Grid positions and charge of the atoms
  int *gix;
  int *giy;
  int *giz;
  T *charge;

private:
  void set_ncoord(const int ncoord);

public:
  Bspline(const int ncoord, const int order, const int nfftx, const int nffty,
          const int nfftz);
  ~Bspline();

  template <typename B> void set_recip(const B *recip);

  void fill_bspline(const float4 *xyzq, const int ncoord);

  void print_dtheta(int start, int end);

  bool compare_dtheta(Bspline &a, int ncoord, double tol);
};

#endif // BSPLINE_H
#endif // NOCUDAC
