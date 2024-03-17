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
#ifndef DOMDECRECIP_H
#define DOMDECRECIP_H

class DomdecRecip {
protected:
  // Settings
  int nfftx, nffty, nfftz;
  int order;
  double kappa;

public:
  DomdecRecip() {}
  DomdecRecip(const int nfftx, const int nffty, const int nfftz,
              const int order, const double kappa)
      : nfftx(nfftx), nffty(nffty), nfftz(nfftz), order(order), kappa(kappa) {}
  ~DomdecRecip() {}
  void setParameters(const int nfftx_, const int nffty_, const int nfftz_,
                     const int order_, const double kappa_) {
    nfftx = nfftx_;
    nffty = nffty_;
    nfftz = nfftz_;
    order = order_;
    kappa = kappa_;
  }

  // virtual void clear_energy_virial() = 0;

  // virtual void get_energy_virial(const bool calc_energy, const bool
  // calc_virial,
  //				 double& energy, double& energy_self, double *virial) =
  //0;
};

#endif // DOMDECRECIP_H
#endif // NOCUDAC
