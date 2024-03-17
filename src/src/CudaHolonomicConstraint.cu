// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include <cmath>
#include <iostream>

#include "CharmmContext.h"
#include "CudaHolonomicConstraint.h"

CudaHolonomicConstraint::CudaHolonomicConstraint() {}

void CudaHolonomicConstraint::setSimulationContext(
    std::shared_ptr<CharmmContext> ctx) {
  context = ctx;

  numWaterMolecules = 0;
  mO = 15.99940;
  mH = 1.00800;
  mH2O = mO + 2.0 * mH;

  mO_div_mH2O = mO / mH2O;
  mH_div_mH2O = mH / mH2O;

  rOHsq = 0.91623184;
  rHHsq = 2.29189321;

  ra = mH_div_mH2O * std::sqrt(4.0 * rOHsq - rHHsq);
  ra_inv = 1.0 / ra;
  rb = ra * mO / (2.0 * mH);
  rc = std::sqrt(rHHsq) / 2.0;
  rc2 = 2.0 * rc;
}

void CudaHolonomicConstraint::setup(double ts) {
  timeStep = ts;
  //xyzq_stored.set_ncoord(context->getNumAtoms());
  coords_stored.allocate(context->getNumAtoms());

  settleWaterIndex = context->getWaterMolecules();
  numWaterMolecules = settleWaterIndex.size();
  // std::cout << "[Holo] "
  //           << "Num of waters : " << settleWaterIndex.size() << "\n";
  //  settleWaterIndex = waterMolecules;
  shakeAtoms = context->getShakeAtoms();
  shakeParams = context->getShakeParams();
}

/*
Each thread handles one water molecule
*/
__global__ static void
settle(const double mH_div_mH2O, const double mO_div_mH2O, const double ra_inv,
       const double ra, const double rb, const double rc, const double rc2,
       const int numWaterMolecules, const int4 *__restrict__ settleWaterIndex,
       const double4 *__restrict__ xyzq, double4 *__restrict__ xyzqNew) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  const double one = 1.0;
  if (index < numWaterMolecules) {
    int i = settleWaterIndex[index].x;
    int j = settleWaterIndex[index].y;
    int k = settleWaterIndex[index].z;

    double x1 = xyzq[i].x;
    double y1 = xyzq[i].y;
    double z1 = xyzq[i].z;
    double x2 = xyzq[j].x;
    double y2 = xyzq[j].y;
    double z2 = xyzq[j].z;
    double x3 = xyzq[k].x;
    double y3 = xyzq[k].y;
    double z3 = xyzq[k].z;

    // Convert to primed coordinates
    double xp1 = xyzqNew[i].x;
    double yp1 = xyzqNew[i].y;
    double zp1 = xyzqNew[i].z;
    double xp2 = xyzqNew[j].x;
    double yp2 = xyzqNew[j].y;
    double zp2 = xyzqNew[j].z;
    double xp3 = xyzqNew[k].x;
    double yp3 = xyzqNew[k].y;
    double zp3 = xyzqNew[k].z;

    // Calculate the center of mass for (x1, y1, z1)
    double xcm = xp1 * mO_div_mH2O + (xp2 + xp3) * mH_div_mH2O;
    double ycm = yp1 * mO_div_mH2O + (yp2 + yp3) * mH_div_mH2O;
    double zcm = zp1 * mO_div_mH2O + (zp2 + zp3) * mH_div_mH2O;

    double xa1 = xp1 - xcm;
    double ya1 = yp1 - ycm;
    double za1 = zp1 - zcm;
    double xb1 = xp2 - xcm;
    double yb1 = yp2 - ycm;
    double zb1 = zp2 - zcm;
    double xc1 = xp3 - xcm;
    double yc1 = yp3 - ycm;
    double zc1 = zp3 - zcm;

    double xb0 = x2 - x1;
    double yb0 = y2 - y1;
    double zb0 = z2 - z1;
    double xc0 = x3 - x1;
    double yc0 = y3 - y1;
    double zc0 = z3 - z1;

    double xakszd = yb0 * zc0 - zb0 * yc0;
    double yakszd = zb0 * xc0 - xb0 * zc0;
    double zakszd = xb0 * yc0 - yb0 * xc0;
    double xaksxd = ya1 * zakszd - za1 * yakszd;
    double yaksxd = za1 * xakszd - xa1 * zakszd;
    double zaksxd = xa1 * yakszd - ya1 * xakszd;
    double xaksyd = yakszd * zaksxd - zakszd * yaksxd;
    double yaksyd = zakszd * xaksxd - xakszd * zaksxd;
    double zaksyd = xakszd * yaksxd - yakszd * xaksxd;

    double axlng_inv =
        one / sqrt(xaksxd * xaksxd + yaksxd * yaksxd + zaksxd * zaksxd);
    double aylng_inv =
        one / sqrt(xaksyd * xaksyd + yaksyd * yaksyd + zaksyd * zaksyd);
    double azlng_inv =
        one / sqrt(xakszd * xakszd + yakszd * yakszd + zakszd * zakszd);

    double trans11 = xaksxd * axlng_inv;
    double trans21 = yaksxd * axlng_inv;
    double trans31 = zaksxd * axlng_inv;
    double trans12 = xaksyd * aylng_inv;
    double trans22 = yaksyd * aylng_inv;
    double trans32 = zaksyd * aylng_inv;
    double trans13 = xakszd * azlng_inv;
    double trans23 = yakszd * azlng_inv;
    double trans33 = zakszd * azlng_inv;

    // Calculate necessary primed coordinates
    double xb0p = trans11 * xb0 + trans21 * yb0 + trans31 * zb0;
    double yb0p = trans12 * xb0 + trans22 * yb0 + trans32 * zb0;
    double xc0p = trans11 * xc0 + trans21 * yc0 + trans31 * zc0;
    double yc0p = trans12 * xc0 + trans22 * yc0 + trans32 * zc0;
    double za1p = trans13 * xa1 + trans23 * ya1 + trans33 * za1;
    double xb1p = trans11 * xb1 + trans21 * yb1 + trans31 * zb1;
    double yb1p = trans12 * xb1 + trans22 * yb1 + trans32 * zb1;
    double zb1p = trans13 * xb1 + trans23 * yb1 + trans33 * zb1;
    double xc1p = trans11 * xc1 + trans21 * yc1 + trans31 * zc1;
    double yc1p = trans12 * xc1 + trans22 * yc1 + trans32 * zc1;
    double zc1p = trans13 * xc1 + trans23 * yc1 + trans33 * zc1;

    // Calculate rotation angles
    double sinphi = za1p * ra_inv;
    double cosphi = sqrt(one - sinphi * sinphi);
    double sinpsi = (zb1p - zc1p) / (rc2 * cosphi);
    double cospsi = sqrt(one - sinpsi * sinpsi);

    double ya2p = ra * cosphi;
    double xb2p = -rc * cospsi;
    double yb2p = -rb * cosphi - rc * sinpsi * sinphi;
    double yc2p = -rb * cosphi + rc * sinpsi * sinphi;

    double alpha = (xb2p * (xb0p - xc0p) + yb0p * yb2p + yc0p * yc2p);
    double beta = (xb2p * (yc0p - yb0p) + xb0p * yb2p + xc0p * yc2p);
    double gamma = xb0p * yb1p - xb1p * yb0p + xc0p * yc1p - xc1p * yc0p;

    double alpha_beta = alpha * alpha + beta * beta;
    double sintheta =
        (alpha * gamma - beta * sqrt(alpha_beta - gamma * gamma)) / alpha_beta;

    double costheta = sqrt(one - sintheta * sintheta);

    double xa3p = -ya2p * sintheta;
    double ya3p = ya2p * costheta;
    double za3p = za1p;
    double xb3p = xb2p * costheta - yb2p * sintheta;
    double yb3p = xb2p * sintheta + yb2p * costheta;
    double zb3p = zb1p;
    double xc3p = -xb2p * costheta - yc2p * sintheta;
    double yc3p = -xb2p * sintheta + yc2p * costheta;
    double zc3p = zc1p;

    xyzqNew[i].x = xcm + trans11 * xa3p + trans12 * ya3p + trans13 * za3p;
    xyzqNew[i].y = ycm + trans21 * xa3p + trans22 * ya3p + trans23 * za3p;
    xyzqNew[i].z = zcm + trans31 * xa3p + trans32 * ya3p + trans33 * za3p;
    xyzqNew[j].x = xcm + trans11 * xb3p + trans12 * yb3p + trans13 * zb3p;
    xyzqNew[j].y = ycm + trans21 * xb3p + trans22 * yb3p + trans23 * zb3p;
    xyzqNew[j].z = zcm + trans31 * xb3p + trans32 * yb3p + trans33 * zb3p;
    xyzqNew[k].x = xcm + trans11 * xc3p + trans12 * yc3p + trans13 * zc3p;
    xyzqNew[k].y = ycm + trans21 * xc3p + trans22 * yc3p + trans23 * zc3p;
    xyzqNew[k].z = zcm + trans31 * xc3p + trans32 * yc3p + trans33 * zc3p;
  }
}
/*
__global__ static void settle_test(const double rOHsq, const double rHHsq,
                                   const int numWaterMolecules,
                                   const int4 *__restrict__ settleWaterIndex,
                                   const double4 *__restrict__ xyzqNew) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index < numWaterMolecules) {
    int i = settleWaterIndex[index].x;
    int j = settleWaterIndex[index].y;
    int k = settleWaterIndex[index].z;

    double tol = 1e-5;
    double rsq = (xyzqNew[i].x - xyzqNew[j].x) * (xyzqNew[i].x - xyzqNew[j].x) +
                 (xyzqNew[i].y - xyzqNew[j].y) * (xyzqNew[i].y - xyzqNew[j].y) +
                 (xyzqNew[i].z - xyzqNew[j].z) * (xyzqNew[i].z - xyzqNew[j].z);
    if (abs(rsq - rOHsq) > tol * 0.1) {
      printf("OH bond length error: %f\n", rsq - rOHsq);
    }
    assert(abs(rsq - rOHsq) < tol);

    rsq = (xyzqNew[i].x - xyzqNew[k].x) * (xyzqNew[i].x - xyzqNew[k].x) +
          (xyzqNew[i].y - xyzqNew[k].y) * (xyzqNew[i].y - xyzqNew[k].y) +
          (xyzqNew[i].z - xyzqNew[k].z) * (xyzqNew[i].z - xyzqNew[k].z);
    if (abs(rsq - rOHsq) > tol * 0.1) {
      printf("OH bond length error: %f\n", rsq - rOHsq);
    }
    assert(abs(rsq - rOHsq) < tol);

    rsq = (xyzqNew[j].x - xyzqNew[k].x) * (xyzqNew[j].x - xyzqNew[k].x) +
          (xyzqNew[j].y - xyzqNew[k].y) * (xyzqNew[j].y - xyzqNew[k].y) +
          (xyzqNew[j].z - xyzqNew[k].z) * (xyzqNew[j].z - xyzqNew[k].z);
    if (abs(rsq - rHHsq) > tol * 0.1) {
      printf("HH bond length error: %f\n", rsq - rHHsq);
    }
    assert(abs(rsq - rHHsq) < tol);
  }
}
*/
void CudaHolonomicConstraint::constrainWaterMolecules(const double4 *ref) {

  auto current = context->getCoordinatesCharges().getDeviceArray().data();

  if (numWaterMolecules) {
    int numThreads = 128;
    int numBlocks = (numWaterMolecules - 1) / numThreads + 1;

    settle<<<numBlocks, numThreads, 0, *stream>>>(
        mH_div_mH2O, mO_div_mH2O, ra_inv, ra, rb, rc, rc2, numWaterMolecules,
        settleWaterIndex.getDeviceArray().data(), ref, current);

    // cudaCheck(cudaStreamSynchronize(*stream));
  }
  // std::cout << "[Holo]Constraining water molecules done.\n";
}

__global__ static void updateVelocitiesKernel(
    int numAtoms, double inv_timeStep, const double4 *__restrict__ stored,
    const double4 *__restrict__ current, double4 *__restrict__ velMass) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index < numAtoms) {
    double cx = current[index].x;
    double cy = current[index].y;
    double cz = current[index].z;

    double sx = stored[index].x;
    double sy = stored[index].y;
    double sz = stored[index].z;

    velMass[index].x += inv_timeStep * (cx - sx);
    velMass[index].y += inv_timeStep * (cy - sy);
    velMass[index].z += inv_timeStep * (cz - sz);
    /*
    double deltax = current[index].x - stored[index].x;
    double deltay = current[index].y - stored[index].y;
    double deltaz = current[index].z - stored[index].z;

    velMass[index].x += inv_timeStep * deltax;
    velMass[index].y += inv_timeStep * deltay;
    velMass[index].z += inv_timeStep * deltaz;


    double cxt = current[index].x * inv_timeStep;
    double sxt = stored[index].x * inv_timeStep;
    double cvt = cxt - sxt;
    double v2 = velMass[index].x + cvt;

    velMass[index].x += (current[index].x - stored[index].x) * inv_timeStep;
    velMass[index].y += (current[index].y - stored[index].y) * inv_timeStep;
    velMass[index].z += (current[index].z - stored[index].z) * inv_timeStep;
    */
  }
}

void CudaHolonomicConstraint::updateVelocities() {
  auto velMass = context->getVelocityMass().getDeviceArray().data();
  // auto xyzq = context->getXYZQ()->getDeviceXYZQ();
  auto coordsCharge = context->getCoordinatesCharges().getDeviceArray().data();

  int numThreads = 128;
  int numBlocks = (context->getNumAtoms() - 1) / numThreads + 1;

  updateVelocitiesKernel<<<numBlocks, numThreads, 0, *stream>>>(
      context->getNumAtoms(), 1.0 / timeStep,
      coords_stored.getDeviceArray().data(), coordsCharge, velMass);

  // cudaCheck(cudaDeviceSynchronize());

  // cudaCheck(cudaStreamSynchronize(*stream));
}

/*
__global__ static void updateVelocitiesKernel(
    int numAtoms, double inv_timeStep, const float4 *__restrict__ stored,
    const float4 *__restrict__ current, double4 *__restrict__ velMass) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index < numAtoms) {

    velMass[index].x = (current[index].x - stored[index].x) * inv_timeStep;
    velMass[index].y = (current[index].y - stored[index].y) * inv_timeStep;
    velMass[index].z = (current[index].z - stored[index].z) * inv_timeStep;
  }
}

void CudaHolonomicConstraint::updateVelocities(XYZQ *ref, XYZQ *current) {
  auto velMass = context->getVelocityMass().getDeviceArray().data();
  auto xyzq = context->getXYZQ()->getDeviceXYZQ();
  auto xyzqRef = ref->getDeviceXYZQ();

  int numThreads = 128;
  int numBlocks = (context->getNumAtoms() - 1) / numThreads + 1;

  updateVelocitiesKernel<<<numBlocks, numThreads>>>(
      context->getNumAtoms(), 1.0 / timeStep, xyzqRef, xyzq, velMass);

  cudaCheck(cudaDeviceSynchronize());
}
*/

__global__ void
constrainShakeTwoAtomsKernel(int numShakes, const double4 *__restrict__ ref,
                             double4 *__restrict__ current,
                             const float4 *__restrict__ shakeParams,
                             const int4 *__restrict__ shakeAtomsIndex) {

  const int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < numShakes) {
    int4 shakeAtoms = shakeAtomsIndex[index];
    if (shakeAtoms.z == -1 && shakeAtoms.w == -1) {
      float4 params = shakeParams[index];
      double xpij = current[shakeAtoms.x].x - current[shakeAtoms.y].x;
      double ypij = current[shakeAtoms.x].y - current[shakeAtoms.y].y;
      double zpij = current[shakeAtoms.x].z - current[shakeAtoms.y].z;

      double rijsq = xpij * xpij + ypij * ypij + zpij * zpij;

      double diff = params.z - rijsq;

      double xrij = ref[shakeAtoms.x].x - ref[shakeAtoms.y].x;
      double yrij = ref[shakeAtoms.x].y - ref[shakeAtoms.y].y;
      double zrij = ref[shakeAtoms.x].z - ref[shakeAtoms.y].z;

      double rrijsq = xrij * xrij + yrij * yrij + zrij * zrij;
      double rijrijp = xrij * xpij + yrij * ypij + zrij * zpij;
      // double lambda =
      //     2.0 * (-rijrijp + sqrt(rijrijp * rijrijp + rrijsq * diff)) /
      //     (rrijsq);

      double massi = params.x;
      double massj = params.w;
      double mm = massi + massj;
      double lambda =
          (-rijrijp + sqrt(rijrijp * rijrijp + rrijsq * diff)) / (rrijsq * mm);

      current[shakeAtoms.x].x += massi * lambda * xrij;
      current[shakeAtoms.x].y += massi * lambda * yrij;
      current[shakeAtoms.x].z += massi * lambda * zrij;
      current[shakeAtoms.y].x -= massj * lambda * xrij;
      current[shakeAtoms.y].y -= massj * lambda * yrij;
      current[shakeAtoms.y].z -= massj * lambda * zrij;
    }
  }
}

__global__ void
constrainShakeThreeAtomsKernel(int numShakes, const double4 *__restrict__ ref,
                               double4 *__restrict__ current,
                               const float4 *__restrict__ shakeParams,
                               const int4 *__restrict__ shakeAtomsIndex) {

  const int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < numShakes) {
    int4 shakeAtoms = shakeAtomsIndex[index];
    if (shakeAtoms.z != -1 && shakeAtoms.w == -1) {
      double tol = 1e-5;
      int max_niter = 25;
      float4 params = shakeParams[index];
      double a120 = 0.0;
      double a130 = 0.0;

      double xrij = ref[shakeAtoms.x].x - ref[shakeAtoms.y].x;
      double yrij = ref[shakeAtoms.x].y - ref[shakeAtoms.y].y;
      double zrij = ref[shakeAtoms.x].z - ref[shakeAtoms.y].z;
      double xrik = ref[shakeAtoms.x].x - ref[shakeAtoms.z].x;
      double yrik = ref[shakeAtoms.x].y - ref[shakeAtoms.z].y;
      double zrik = ref[shakeAtoms.x].z - ref[shakeAtoms.z].z;

      double rrijsq = xrij * xrij + yrij * yrij + zrij * zrij;
      double rriksq = xrik * xrik + yrik * yrik + zrik * zrik;
      double rijrik = xrij * xrik + yrij * yrik + zrij * zrik;

      double mmi = params.x;
      double mmj = params.w;
      double mmk = mmj;
      double mij = params.x + params.w;
      double mik = mij;

      double acorr1 = mij * mij * rrijsq;
      double acorr2 = mij * mmi * 2.0 * rijrik;
      double acorr3 = mmi * mmi * rriksq;
      double acorr4 = mmi * mmi * rrijsq;
      double acorr5 = mik * mmi * 2.0 * rijrik;
      double acorr6 = mik * mik * rriksq;

      double xpij = current[shakeAtoms.x].x - current[shakeAtoms.y].x;
      double ypij = current[shakeAtoms.x].y - current[shakeAtoms.y].y;
      double zpij = current[shakeAtoms.x].z - current[shakeAtoms.y].z;
      double xpik = current[shakeAtoms.x].x - current[shakeAtoms.z].x;
      double ypik = current[shakeAtoms.x].y - current[shakeAtoms.z].y;
      double zpik = current[shakeAtoms.x].z - current[shakeAtoms.z].z;

      double rijsq = xpij * xpij + ypij * ypij + zpij * zpij;
      double riksq = xpik * xpik + ypik * ypik + zpik * zpik;

      double dij = params.z - rijsq;
      double dik = params.z - riksq;

      double rijrijp = xrij * xpij + yrij * ypij + zrij * zpij;
      double rijrikp = xrij * xpik + yrij * ypik + zrij * zpik;
      double rikrijp = xpij * xrik + ypij * yrik + zpij * zrik;
      double rikrikp = xrik * xpik + yrik * ypik + zrik * zpik;

      double dinv =
          0.5 / (rijrijp * rikrikp * mij * mik - rijrikp * rikrijp * mmi * mmi);
      double a12 = dinv * (rikrikp * mik * (dij)-rikrijp * mmi * (dik));
      double a13 = dinv * (-mmi * rijrikp * (dij) + rijrijp * mij * (dik));

      int aniter = 0;
      do {
        aniter = aniter + 1;
        a120 = a12;
        a130 = a13;
        double a12corr =
            acorr1 * a12 * a12 + acorr2 * a12 * a13 + acorr3 * a13 * a13;
        double a13corr =
            acorr4 * a12 * a12 + acorr5 * a12 * a13 + acorr6 * a13 * a13;
        a12 = dinv * (rikrikp * mik * (dij - a12corr) -
                      rikrijp * mmi * (dik - a13corr));
        a13 = dinv * (-mmi * rijrikp * (dij - a12corr) +
                      rijrijp * mij * (dik - a13corr));

      } while ((abs(a120 - a12) > tol || (abs(a130 - a13) > tol)) &&
               aniter < max_niter);
      current[shakeAtoms.x].x += mmi * (a12 * xrij + a13 * xrik);
      current[shakeAtoms.y].x -= mmj * a12 * xrij;
      current[shakeAtoms.z].x -= mmk * a13 * xrik;

      current[shakeAtoms.x].y += mmi * (a12 * yrij + a13 * yrik);
      current[shakeAtoms.y].y -= mmj * a12 * yrij;
      current[shakeAtoms.z].y -= mmk * a13 * yrik;

      current[shakeAtoms.x].z += mmi * (a12 * zrij + a13 * zrik);
      current[shakeAtoms.y].z -= mmj * a12 * zrij;
      current[shakeAtoms.z].z -= mmk * a13 * zrik;
    }
  }
}

__global__ void
constrainShakeFourAtomsKernel(int numShakes, const double4 *__restrict__ ref,
                              double4 *__restrict__ current,
                              const float4 *__restrict__ shakeParams,
                              const int4 *__restrict__ shakeAtomsIndex) {
  const int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < numShakes) {
    int4 shakeAtoms = shakeAtomsIndex[index];
    // if (shakeAtoms.z != -1 && shakeAtoms.w != -1) {
    if (shakeAtoms.w != -1) {
      double tol = 1e-5;
      int max_niter = 25;
      float4 params = shakeParams[index];
      double a120 = 0.0;
      double a130 = 0.0;
      double a140 = 0.0;

      double xrij = ref[shakeAtoms.x].x - ref[shakeAtoms.y].x;
      double yrij = ref[shakeAtoms.x].y - ref[shakeAtoms.y].y;
      double zrij = ref[shakeAtoms.x].z - ref[shakeAtoms.y].z;
      double xrik = ref[shakeAtoms.x].x - ref[shakeAtoms.z].x;
      double yrik = ref[shakeAtoms.x].y - ref[shakeAtoms.z].y;
      double zrik = ref[shakeAtoms.x].z - ref[shakeAtoms.z].z;
      double xril = ref[shakeAtoms.x].x - ref[shakeAtoms.w].x;
      double yril = ref[shakeAtoms.x].y - ref[shakeAtoms.w].y;
      double zril = ref[shakeAtoms.x].z - ref[shakeAtoms.w].z;

      double rrijsq = xrij * xrij + yrij * yrij + zrij * zrij;
      double rriksq = xrik * xrik + yrik * yrik + zrik * zrik;
      double rrilsq = xril * xril + yril * yril + zril * zril;
      double rijrik = xrij * xrik + yrij * yrik + zrij * zrik;
      double rijril = xrij * xril + yrij * yril + zrij * zril;
      double rikril = xrik * xril + yrik * yril + zrik * zril;

      double mmi = params.x;
      double mmj = params.w;
      double mmk = mmj;
      double mml = mmj;
      double mij = params.x + params.w;
      double mik = mij;
      double mil = mij;

      double acorr1 = mij * mij * rrijsq;
      double acorr2 = 2.0 * mij * mmi * rijrik;
      double acorr3 = mmi * mmi * rriksq;
      double acorr4 = mmi * mmi * rrijsq;
      double acorr5 = 2.0 * mik * mmi * rijrik;
      double acorr6 = mik * mik * rriksq;
      double acorr7 = 2.0 * mij * mmi * rijril;
      double acorr8 = 2.0 * mmi * mmi * rikril;
      double acorr9 = mmi * mmi * rrilsq;
      double acorr10 = 2.0 * mmi * mmi * rijril;
      double acorr11 = 2.0 * mmi * mik * rikril;
      double acorr12 = 2.0 * mmi * mmi * rijrik;
      double acorr13 = 2.0 * mmi * mil * rijril;
      double acorr14 = 2.0 * mmi * mil * rikril;
      double acorr15 = mil * mil * rrilsq;

      double xpij = current[shakeAtoms.x].x - current[shakeAtoms.y].x;
      double ypij = current[shakeAtoms.x].y - current[shakeAtoms.y].y;
      double zpij = current[shakeAtoms.x].z - current[shakeAtoms.y].z;
      double xpik = current[shakeAtoms.x].x - current[shakeAtoms.z].x;
      double ypik = current[shakeAtoms.x].y - current[shakeAtoms.z].y;
      double zpik = current[shakeAtoms.x].z - current[shakeAtoms.z].z;
      double xpil = current[shakeAtoms.x].x - current[shakeAtoms.w].x;
      double ypil = current[shakeAtoms.x].y - current[shakeAtoms.w].y;
      double zpil = current[shakeAtoms.x].z - current[shakeAtoms.w].z;

      double rijsq = xpij * xpij + ypij * ypij + zpij * zpij;
      double riksq = xpik * xpik + ypik * ypik + zpik * zpik;
      double rilsq = xpil * xpil + ypil * ypil + zpil * zpil;

      double dij = params.z - rijsq;
      double dik = params.z - riksq;
      double dil = params.z - rilsq;
      double rijrijp = xrij * xpij + yrij * ypij + zrij * zpij;
      double rijrikp = xrij * xpik + yrij * ypik + zrij * zpik;
      double rijrilp = xrij * xpil + yrij * ypil + zrij * zpil;
      double rikrijp = xrik * xpij + yrik * ypij + zrik * zpij;
      double rikrikp = xrik * xpik + yrik * ypik + zrik * zpik;
      double rikrilp = xrik * xpil + yrik * ypil + zrik * zpil;
      double rilrijp = xril * xpij + yril * ypij + zril * zpij;
      double rilrikp = xril * xpik + yril * ypik + zril * zpik;
      double rilrilp = xril * xpil + yril * ypil + zril * zpil;

      double d1 = mik * mil * rikrikp * rilrilp - mmi * mmi * rikrilp * rilrikp;
      double d2 = mmi * mil * rikrijp * rilrilp - mmi * mmi * rikrilp * rilrijp;
      double d3 = mmi * mmi * rikrijp * rilrikp - mik * mmi * rikrikp * rilrijp;
      double d4 = mmi * mil * rijrikp * rilrilp - mmi * mmi * rijrilp * rilrikp;
      double d5 = mij * mil * rijrijp * rilrilp - mmi * mmi * rijrilp * rilrijp;
      double d6 = mij * mmi * rijrijp * rilrikp - mmi * mmi * rijrikp * rilrijp;
      double d7 = mmi * mmi * rijrikp * rikrilp - mmi * mik * rijrilp * rikrikp;
      double d8 = mij * mmi * rijrijp * rikrilp - mmi * mmi * rijrilp * rikrijp;
      double d9 = mij * mik * rijrijp * rikrikp - mmi * mmi * rijrikp * rikrijp;

      double dinv =
          0.5 / (rijrijp * mij * d1 - mmi * rijrikp * d2 + mmi * rijrilp * d3);
      double a12 = dinv * (d1 * dij - d2 * dik + d3 * dil);
      double a13 = dinv * (-d4 * dij + d5 * dik - d6 * dil);
      double a14 = dinv * (d7 * dij - d8 * dik + d9 * dil);
      int aniter = 0;

      do {
        aniter = aniter + 1;
        a120 = a12;
        a130 = a13;
        a140 = a14;
        double a12corr = acorr1 * a12 * a12 + acorr2 * a12 * a13 +
                         acorr3 * a13 * a13 + acorr7 * a12 * a14 +
                         acorr8 * a13 * a14 + acorr9 * a14 * a14;
        double a13corr = acorr4 * a12 * a12 + acorr5 * a12 * a13 +
                         acorr6 * a13 * a13 + acorr10 * a12 * a14 +
                         acorr11 * a13 * a14 + acorr9 * a14 * a14;
        double a14corr = acorr4 * a12 * a12 + acorr12 * a12 * a13 +
                         acorr3 * a13 * a13 + acorr13 * a12 * a14 +
                         acorr14 * a13 * a14 + acorr15 * a14 * a14;

        a12 = dinv * (d1 * (dij - a12corr) - d2 * (dik - a13corr) +
                      d3 * (dil - a14corr));
        a13 = dinv * (-d4 * (dij - a12corr) + d5 * (dik - a13corr) -
                      d6 * (dil - a14corr));
        a14 = dinv * (d7 * (dij - a12corr) - d8 * (dik - a13corr) +
                      d9 * (dil - a14corr));
      } while ((abs(a120 - a12) > tol || (abs(a130 - a13) > tol) ||
                (abs(a140 - a14) > tol)) &&
               aniter < max_niter);

      current[shakeAtoms.x].x += mmi * (a12 * xrij + a13 * xrik + a14 * xril);
      current[shakeAtoms.x].y += mmi * (a12 * yrij + a13 * yrik + a14 * yril);
      current[shakeAtoms.x].z += mmi * (a12 * zrij + a13 * zrik + a14 * zril);
      current[shakeAtoms.y].x -= mmj * a12 * xrij;
      current[shakeAtoms.y].y -= mmj * a12 * yrij;
      current[shakeAtoms.y].z -= mmj * a12 * zrij;
      current[shakeAtoms.z].x -= mmk * a13 * xrik;
      current[shakeAtoms.z].y -= mmk * a13 * yrik;
      current[shakeAtoms.z].z -= mmk * a13 * zrik;
      current[shakeAtoms.w].x -= mml * a14 * xril;
      current[shakeAtoms.w].y -= mml * a14 * yril;
      current[shakeAtoms.w].z -= mml * a14 * zril;
    }
  }
}

void CudaHolonomicConstraint::constrainShakeAtoms(const double4 *ref) {

  if (shakeAtoms.size()) {
    auto current = context->getCoordinatesCharges().getDeviceArray().data();

    int numThreads = 128;
    int numBlocks = (shakeAtoms.size() - 1) / numThreads + 1;

    constrainShakeTwoAtomsKernel<<<numBlocks, numThreads, 0, *stream>>>(
        shakeAtoms.size(), ref, current, shakeParams.getDeviceArray().data(),
        shakeAtoms.getDeviceArray().data());
    // cudaCheck(cudaStreamSynchronize(*stream));

    constrainShakeThreeAtomsKernel<<<numBlocks, numThreads, 0, *stream>>>(
        shakeAtoms.size(), ref, current, shakeParams.getDeviceArray().data(),
        shakeAtoms.getDeviceArray().data());
    // cudaCheck(cudaDeviceSynchronize());

    constrainShakeFourAtomsKernel<<<numBlocks, numThreads, 0, *stream>>>(
        shakeAtoms.size(), ref, current, shakeParams.getDeviceArray().data(),
        shakeAtoms.getDeviceArray().data());
    // cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaStreamSynchronize(*stream));
  }
}

void CudaHolonomicConstraint::handleHolonomicConstraints(const double4 *ref) {

  /*
  if (charmmContext->getConstraintType() == ConstraintType::HOLONOMIC) {
    constrainWaterMolecules();
  }
  */

  auto current = context->getCoordinatesCharges().getDeviceArray().data();

  constrainWaterMolecules(ref);
  copy_DtoD_async<double4>(current, coords_stored.getDeviceArray().data(),
                           context->getNumAtoms(), *memcpyStream);

  constrainShakeAtoms(ref);

  cudaStreamSynchronize(*memcpyStream);
  updateVelocities();

  // xx updateVelocities(ref, current);
}
