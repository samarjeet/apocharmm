// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CudaHolonomicConstraint.h"

#include "CharmmContext.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

CudaHolonomicConstraint::CudaHolonomicConstraint(void)
    : m_Context(nullptr), m_SettleAtoms(), m_ShakeAtoms(),
      m_AllConstrainedAtomPairs(), m_ShakeParams(), m_CoordsStored(),
      m_TimeStep(-9999.9999), m_Stream(nullptr), mO(15.99940), mH(1.00800),
      mH2O(18.0154), mO_div_mH2O(-9999.9999), mH_div_mH2O(-9999.9999),
      rOHsq(0.91623184), rHHsq(2.29189321), ra(-9999.9999), ra_inv(-9999.9999),
      rb(-9999.9999), rc(-9999.9999), rc2(-9999.9999) {
  this->mO_div_mH2O = this->mO / this->mH2O;
  this->mH_div_mH2O = this->mH / this->mH2O;

  this->ra = this->mH_div_mH2O * std::sqrt(4.0 * this->rOHsq - this->rHHsq);
  this->ra_inv = 1.0 / this->ra;
  this->rb = this->ra * this->mO / (2.0 * this->mH);
  this->rc = std::sqrt(this->rHHsq) / 2.0;
  this->rc2 = 2.0 * this->rc;
}

void CudaHolonomicConstraint::setCharmmContext(
    std::shared_ptr<CharmmContext> ctx) {
  m_Context = ctx;
  return;
}

void CudaHolonomicConstraint::setStream(std::shared_ptr<cudaStream_t> stream) {
  m_Stream = stream;
  return;
}

void CudaHolonomicConstraint::setup(const double timeStep) {
  if (m_Context == nullptr) {
    throw std::runtime_error("CudaHolonomicConstraint::setup(const double): No "
                             "CharmmContext was set");
  }

  const int numAtoms = m_Context->getNumAtoms();

  m_TimeStep = timeStep;
  m_CoordsStored.resize(numAtoms);

  m_SettleAtoms = m_Context->getWaterMolecules();
  m_ShakeAtoms = m_Context->getShakeAtoms();
  m_ShakeParams = m_Context->getShakeParams();

  std::vector<int2> allConstrainedAtomPairs;
  for (std::size_t i = 0; i < m_ShakeAtoms.size(); i++) {
    allConstrainedAtomPairs.push_back(
        make_int2(m_ShakeAtoms[i].x, m_ShakeAtoms[i].y));
    if (m_ShakeAtoms[i].z != -1) {
      allConstrainedAtomPairs.push_back(
          make_int2(m_ShakeAtoms[i].x, m_ShakeAtoms[i].z));
      allConstrainedAtomPairs.push_back(
          make_int2(m_ShakeAtoms[i].y, m_ShakeAtoms[i].z));
    } else if (m_ShakeAtoms[i].w == -1) {
      allConstrainedAtomPairs.push_back(
          make_int2(m_ShakeAtoms[i].x, m_ShakeAtoms[i].w));
      allConstrainedAtomPairs.push_back(
          make_int2(m_ShakeAtoms[i].y, m_ShakeAtoms[i].w));
      allConstrainedAtomPairs.push_back(
          make_int2(m_ShakeAtoms[i].z, m_ShakeAtoms[i].w));
    }
  }

  for (std::size_t i = 0; i < m_SettleAtoms.size(); i++) {
    allConstrainedAtomPairs.push_back(
        make_int2(m_SettleAtoms[i].x, m_SettleAtoms[i].y));
    allConstrainedAtomPairs.push_back(
        make_int2(m_SettleAtoms[i].x, m_SettleAtoms[i].z));
    allConstrainedAtomPairs.push_back(
        make_int2(m_SettleAtoms[i].y, m_SettleAtoms[i].z));
  }

  m_AllConstrainedAtomPairs = allConstrainedAtomPairs;

  return;
}

void CudaHolonomicConstraint::handleHolonomicConstraints(
    const double4 *coordsRef) {
  copy_DtoD_async<double4>(
      m_Context->getCoordinatesCharges().getDeviceArray().data(),
      m_CoordsStored.getDeviceArray().data(), m_Context->getNumAtoms(),
      *m_Stream);

  this->constrainWaterMolecules(coordsRef);
  this->constrainShakeAtoms(coordsRef);
  this->updateVelocities();

  cudaCheck(cudaStreamSynchronize(*m_Stream));

  return;
}

__global__ static void RemoveForceAlongHolonomicConstraintsKernel(
    double *__restrict__ forces, const int forceStride,
    const int2 *__restrict__ constrainedPairs, const int numConstraints,
    const double4 *__restrict__ coords) {
  constexpr int maxIter = 20;
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int iter = 0; iter < maxIter; iter++) {
    for (int i = index; i < numConstraints; i += stride) {
      const int2 pair = constrainedPairs[i];
      const double xpij =
          forces[0 * forceStride + pair.x] - forces[0 * forceStride + pair.y];
      const double ypij =
          forces[1 * forceStride + pair.x] - forces[1 * forceStride + pair.y];
      const double zpij =
          forces[2 * forceStride + pair.x] - forces[2 * forceStride + pair.y];

      const double xrij = coords[pair.x].x - coords[pair.y].x;
      const double yrij = coords[pair.x].y - coords[pair.y].y;
      const double zrij = coords[pair.x].z - coords[pair.y].z;

      const double rrijsq = xrij * xrij + yrij * yrij + zrij * zrij;
      const double rijrijp = xrij * xpij + yrij * ypij + zrij * zpij;

      const double acor = -0.5 * rijrijp / rrijsq; // 0.5 as per CHARMM

      atomicAdd(forces + 0 * forceStride + pair.x, acor * xrij);
      atomicAdd(forces + 1 * forceStride + pair.x, acor * yrij);
      atomicAdd(forces + 2 * forceStride + pair.x, acor * zrij);
      atomicAdd(forces + 0 * forceStride + pair.y, -acor * xrij);
      atomicAdd(forces + 1 * forceStride + pair.y, -acor * yrij);
      atomicAdd(forces + 2 * forceStride + pair.y, -acor * zrij);
    }
  }

  return;
}

void CudaHolonomicConstraint::removeForceAlongHolonomicConstraints(void) {
  if (m_AllConstrainedAtomPairs.size() > 0) {
    RemoveForceAlongHolonomicConstraintsKernel<<<64, 1024, 0, *m_Stream>>>(
        m_Context->getForces()->xyz(), m_Context->getForceStride(),
        m_AllConstrainedAtomPairs.getDeviceArray().data(),
        m_AllConstrainedAtomPairs.size(),
        m_Context->getCoordinatesCharges().getDeviceArray().data());
    cudaStreamSynchronize(*m_Stream);
  }
  return;
}

//
// Each thread handles one water molecule
//
__global__ static void
SettleKernel(double4 *__restrict__ xyzq, const double4 *__restrict__ xyzq0,
             const int4 *__restrict__ settleAtoms, const int numSettles,
             const double mO_div_mH2O, const double mH_div_mH2O,
             const double ra, const double ra_inv, const double rb,
             const double rc, const double rc2) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = index; i < numSettles; i += stride) {
    const int iatom = settleAtoms[i].x;
    const int jatom = settleAtoms[i].y;
    const int katom = settleAtoms[i].z;

    const double x1 = xyzq0[iatom].x;
    const double y1 = xyzq0[iatom].y;
    const double z1 = xyzq0[iatom].z;
    const double x2 = xyzq0[jatom].x;
    const double y2 = xyzq0[jatom].y;
    const double z2 = xyzq0[jatom].z;
    const double x3 = xyzq0[katom].x;
    const double y3 = xyzq0[katom].y;
    const double z3 = xyzq0[katom].z;

    // Convert to primed coordinates
    const double xp1 = xyzq[iatom].x;
    const double yp1 = xyzq[iatom].y;
    const double zp1 = xyzq[iatom].z;
    const double xp2 = xyzq[jatom].x;
    const double yp2 = xyzq[jatom].y;
    const double zp2 = xyzq[jatom].z;
    const double xp3 = xyzq[katom].x;
    const double yp3 = xyzq[katom].y;
    const double zp3 = xyzq[katom].z;

    // Calculate the center of mass for (x1, y1, z1)
    const double xcm = xp1 * mO_div_mH2O + (xp2 + xp3) * mH_div_mH2O;
    const double ycm = yp1 * mO_div_mH2O + (yp2 + yp3) * mH_div_mH2O;
    const double zcm = zp1 * mO_div_mH2O + (zp2 + zp3) * mH_div_mH2O;

    const double xa1 = xp1 - xcm;
    const double ya1 = yp1 - ycm;
    const double za1 = zp1 - zcm;
    const double xb1 = xp2 - xcm;
    const double yb1 = yp2 - ycm;
    const double zb1 = zp2 - zcm;
    const double xc1 = xp3 - xcm;
    const double yc1 = yp3 - ycm;
    const double zc1 = zp3 - zcm;

    const double xb0 = x2 - x1;
    const double yb0 = y2 - y1;
    const double zb0 = z2 - z1;
    const double xc0 = x3 - x1;
    const double yc0 = y3 - y1;
    const double zc0 = z3 - z1;

    const double xakszd = yb0 * zc0 - zb0 * yc0;
    const double yakszd = zb0 * xc0 - xb0 * zc0;
    const double zakszd = xb0 * yc0 - yb0 * xc0;
    const double xaksxd = ya1 * zakszd - za1 * yakszd;
    const double yaksxd = za1 * xakszd - xa1 * zakszd;
    const double zaksxd = xa1 * yakszd - ya1 * xakszd;
    const double xaksyd = yakszd * zaksxd - zakszd * yaksxd;
    const double yaksyd = zakszd * xaksxd - xakszd * zaksxd;
    const double zaksyd = xakszd * yaksxd - yakszd * xaksxd;

    const double axlng_inv =
        1.0 / sqrt(xaksxd * xaksxd + yaksxd * yaksxd + zaksxd * zaksxd);
    const double aylng_inv =
        1.0 / sqrt(xaksyd * xaksyd + yaksyd * yaksyd + zaksyd * zaksyd);
    const double azlng_inv =
        1.0 / sqrt(xakszd * xakszd + yakszd * yakszd + zakszd * zakszd);

    const double trans11 = xaksxd * axlng_inv;
    const double trans21 = yaksxd * axlng_inv;
    const double trans31 = zaksxd * axlng_inv;
    const double trans12 = xaksyd * aylng_inv;
    const double trans22 = yaksyd * aylng_inv;
    const double trans32 = zaksyd * aylng_inv;
    const double trans13 = xakszd * azlng_inv;
    const double trans23 = yakszd * azlng_inv;
    const double trans33 = zakszd * azlng_inv;

    // Calculate necessary primed coordinates
    const double xb0p = trans11 * xb0 + trans21 * yb0 + trans31 * zb0;
    const double yb0p = trans12 * xb0 + trans22 * yb0 + trans32 * zb0;
    const double xc0p = trans11 * xc0 + trans21 * yc0 + trans31 * zc0;
    const double yc0p = trans12 * xc0 + trans22 * yc0 + trans32 * zc0;
    const double za1p = trans13 * xa1 + trans23 * ya1 + trans33 * za1;
    const double xb1p = trans11 * xb1 + trans21 * yb1 + trans31 * zb1;
    const double yb1p = trans12 * xb1 + trans22 * yb1 + trans32 * zb1;
    const double zb1p = trans13 * xb1 + trans23 * yb1 + trans33 * zb1;
    const double xc1p = trans11 * xc1 + trans21 * yc1 + trans31 * zc1;
    const double yc1p = trans12 * xc1 + trans22 * yc1 + trans32 * zc1;
    const double zc1p = trans13 * xc1 + trans23 * yc1 + trans33 * zc1;

    // Calculate rotation angles
    const double sinphi = za1p * ra_inv;
    const double cosphi = sqrt(1.0 - sinphi * sinphi);
    const double sinpsi = (zb1p - zc1p) / (rc2 * cosphi);
    const double cospsi = sqrt(1.0 - sinpsi * sinpsi);

    const double ya2p = ra * cosphi;
    const double xb2p = -rc * cospsi;
    const double yb2p = -rb * cosphi - rc * sinpsi * sinphi;
    const double yc2p = -rb * cosphi + rc * sinpsi * sinphi;

    const double alpha = (xb2p * (xb0p - xc0p) + yb0p * yb2p + yc0p * yc2p);
    const double beta = (xb2p * (yc0p - yb0p) + xb0p * yb2p + xc0p * yc2p);
    const double gamma = xb0p * yb1p - xb1p * yb0p + xc0p * yc1p - xc1p * yc0p;

    const double alpha_beta = alpha * alpha + beta * beta;
    const double sintheta =
        (alpha * gamma - beta * sqrt(alpha_beta - gamma * gamma)) / alpha_beta;
    const double costheta = sqrt(1.0 - sintheta * sintheta);

    const double xa3p = -ya2p * sintheta;
    const double ya3p = ya2p * costheta;
    const double za3p = za1p;
    const double xb3p = xb2p * costheta - yb2p * sintheta;
    const double yb3p = xb2p * sintheta + yb2p * costheta;
    const double zb3p = zb1p;
    const double xc3p = -xb2p * costheta - yc2p * sintheta;
    const double yc3p = -xb2p * sintheta + yc2p * costheta;
    const double zc3p = zc1p;

    xyzq[iatom].x = xcm + trans11 * xa3p + trans12 * ya3p + trans13 * za3p;
    xyzq[iatom].y = ycm + trans21 * xa3p + trans22 * ya3p + trans23 * za3p;
    xyzq[iatom].z = zcm + trans31 * xa3p + trans32 * ya3p + trans33 * za3p;
    xyzq[jatom].x = xcm + trans11 * xb3p + trans12 * yb3p + trans13 * zb3p;
    xyzq[jatom].y = ycm + trans21 * xb3p + trans22 * yb3p + trans23 * zb3p;
    xyzq[jatom].z = zcm + trans31 * xb3p + trans32 * yb3p + trans33 * zb3p;
    xyzq[katom].x = xcm + trans11 * xc3p + trans12 * yc3p + trans13 * zc3p;
    xyzq[katom].y = ycm + trans21 * xc3p + trans22 * yc3p + trans23 * zc3p;
    xyzq[katom].z = zcm + trans31 * xc3p + trans32 * yc3p + trans33 * zc3p;
  }

  return;
}

void CudaHolonomicConstraint::constrainWaterMolecules(
    const double4 *coordsRef) {
  double4 *coords = m_Context->getCoordinatesCharges().getDeviceArray().data();

  if (m_SettleAtoms.size() > 0) {
    constexpr int numThreads = 128;
    const int numBlocks =
        (static_cast<int>(m_SettleAtoms.size()) + numThreads - 1) / numThreads;

    SettleKernel<<<numBlocks, numThreads, 0, *m_Stream>>>(
        coords, coordsRef, m_SettleAtoms.getDeviceArray().data(),
        static_cast<int>(m_SettleAtoms.size()), this->mO_div_mH2O,
        this->mH_div_mH2O, this->ra, this->ra_inv, this->rb, this->rc,
        this->rc2);
  }

  return;
}

__global__ static void Shake2Kernel(double4 *__restrict__ coords,
                                    const double4 *__restrict__ coordsRef,
                                    const int4 *__restrict__ shakeAtoms,
                                    const float4 *__restrict__ shakeParams,
                                    const int numShakes) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = index; i < numShakes; i += stride) {
    const int4 atoms = shakeAtoms[i];
    if ((atoms.z == -1) && (atoms.w == -1)) {
      const float4 params = shakeParams[i];
      const double imass = static_cast<double>(params.x);
      const double jmass = static_cast<double>(params.w);
      const double bijsq = static_cast<double>(params.z);

      const double xpij = coords[atoms.x].x - coords[atoms.y].x;
      const double ypij = coords[atoms.x].y - coords[atoms.y].y;
      const double zpij = coords[atoms.x].z - coords[atoms.y].z;

      const double xrij = coordsRef[atoms.x].x - coordsRef[atoms.y].x;
      const double yrij = coordsRef[atoms.x].y - coordsRef[atoms.y].y;
      const double zrij = coordsRef[atoms.x].z - coordsRef[atoms.y].z;

      const double pijpijsq = xpij * xpij + ypij * ypij + zpij * zpij;
      const double rijrijsq = xrij * xrij + yrij * yrij + zrij * zrij;
      const double rijpijsq = xrij * xpij + yrij * ypij + zrij * zpij;
      const double dijsq = bijsq - pijpijsq;
      const double lambda =
          (-rijpijsq + sqrt(rijpijsq * rijpijsq + rijrijsq * dijsq)) /
          (rijrijsq * (imass + jmass));

      coords[atoms.x].x += imass * lambda * xrij;
      coords[atoms.x].y += imass * lambda * yrij;
      coords[atoms.x].z += imass * lambda * zrij;
      coords[atoms.y].x -= jmass * lambda * xrij;
      coords[atoms.y].y -= jmass * lambda * yrij;
      coords[atoms.y].z -= jmass * lambda * zrij;
    }
  }

  return;
}

__global__ static void Shake3Kernel(double4 *__restrict__ coords,
                                    const double4 *__restrict__ coordsRef,
                                    const int4 *__restrict__ shakeAtoms,
                                    const float4 *__restrict__ shakeParams,
                                    const int numShakes) {
  constexpr int MAX_ITER = 25;
  constexpr double TOL = 1e-5;

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = index; i < numShakes; i += stride) {
    const int4 atoms = shakeAtoms[i];
    if ((atoms.z != -1) && (atoms.w == -1)) {
      const float4 params = shakeParams[i];
      const double imass = static_cast<double>(params.x);
      const double jmass = static_cast<double>(params.w);
      const double kmass = jmass;
      const double ijmass = imass + jmass;
      const double ikmass = imass + kmass;
      const double bijsq = static_cast<double>(params.z);
      const double biksq = bijsq;

      const double xpij = coords[atoms.x].x - coords[atoms.y].x;
      const double ypij = coords[atoms.x].y - coords[atoms.y].y;
      const double zpij = coords[atoms.x].z - coords[atoms.y].z;
      const double xpik = coords[atoms.x].x - coords[atoms.z].x;
      const double ypik = coords[atoms.x].y - coords[atoms.z].y;
      const double zpik = coords[atoms.x].z - coords[atoms.z].z;

      const double xrij = coordsRef[atoms.x].x - coordsRef[atoms.y].x;
      const double yrij = coordsRef[atoms.x].y - coordsRef[atoms.y].y;
      const double zrij = coordsRef[atoms.x].z - coordsRef[atoms.y].z;
      const double xrik = coordsRef[atoms.x].x - coordsRef[atoms.z].x;
      const double yrik = coordsRef[atoms.x].y - coordsRef[atoms.z].y;
      const double zrik = coordsRef[atoms.x].z - coordsRef[atoms.z].z;

      const double pijpijsq = xpij * xpij + ypij * ypij + zpij * zpij;
      const double pikpiksq = xpik * xpik + ypik * ypik + zpik * zpik;
      const double rijrijsq = xrij * xrij + yrij * yrij + zrij * zrij;
      const double rikriksq = xrik * xrik + yrik * yrik + zrik * zrik;
      const double rijriksq = xrij * xrik + yrij * yrik + zrij * zrik;

      const double rijpijsq = xrij * xpij + yrij * ypij + zrij * zpij;
      const double rijpiksq = xrij * xpik + yrij * ypik + zrij * zpik;
      const double rikpijsq = xrik * xpij + yrik * ypij + zrik * zpij;
      const double rikpiksq = xrik * xpik + yrik * ypik + zrik * zpik;

      const double dijsq = bijsq - pijpijsq;
      const double diksq = biksq - pikpiksq;

      const double dinv = 0.5 / (rijpijsq * rikpiksq * ijmass * ikmass -
                                 rijpiksq * rikpijsq * imass * imass);
      const double acorr1 = ijmass * ijmass * rijrijsq;
      const double acorr2 = 2.0 * ijmass * imass * rijriksq;
      const double acorr3 = imass * imass * rikriksq;
      const double acorr4 = imass * imass * rijrijsq;
      const double acorr5 = 2.0 * ikmass * imass * rijriksq;
      const double acorr6 = ikmass * ikmass * rikriksq;

      double a120 = 0.0, a130 = 0.0;
      double a12 =
          dinv * (rikpiksq * ikmass * dijsq - rikpijsq * imass * diksq);
      double a13 =
          dinv * (rijpijsq * ijmass * diksq - rijpiksq * imass * dijsq);
      for (int ITER = 0; ITER < MAX_ITER; ITER++) {
        a120 = a12;
        a130 = a13;

        const double a12corr =
            acorr1 * a12 * a12 + acorr2 * a12 * a13 + acorr3 * a13 * a13;
        const double a13corr =
            acorr4 * a12 * a12 + acorr5 * a12 * a13 + acorr6 * a13 * a13;

        a12 = dinv * (rikpiksq * ikmass * (dijsq - a12corr) -
                      rikpijsq * imass * (diksq - a13corr));
        a13 = dinv * (rijpijsq * ijmass * (diksq - a13corr) -
                      rijpiksq * imass * (dijsq - a12corr));

        if ((abs(a120 - a12) < TOL) && (abs(a130 - a13) < TOL))
          break;
      }

      coords[atoms.x].x += imass * (a12 * xrij + a13 * xrik);
      coords[atoms.x].y += imass * (a12 * yrij + a13 * yrik);
      coords[atoms.x].z += imass * (a12 * zrij + a13 * zrik);

      coords[atoms.y].x -= jmass * a12 * xrij;
      coords[atoms.y].y -= jmass * a12 * yrij;
      coords[atoms.y].z -= jmass * a12 * zrij;

      coords[atoms.z].x -= kmass * a13 * xrik;
      coords[atoms.z].y -= kmass * a13 * yrik;
      coords[atoms.z].z -= kmass * a13 * zrik;
    }
  }

  return;
}

__global__ static void Shake4Kernel(double4 *__restrict__ coords,
                                    const double4 *__restrict__ coordsRef,
                                    const int4 *__restrict__ shakeAtoms,
                                    const float4 *__restrict__ shakeParams,
                                    const int numShakes) {
  constexpr int MAX_ITER = 25;
  constexpr double TOL = 1e-5;

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = index; i < numShakes; i += stride) {
    const int4 atoms = shakeAtoms[i];
    if (atoms.w != -1) {
      const float4 params = shakeParams[i];
      const double imass = static_cast<double>(params.x);
      const double jmass = static_cast<double>(params.w);
      const double kmass = jmass;
      const double lmass = jmass;
      const double ijmass = imass + jmass;
      const double ikmass = imass + kmass;
      const double ilmass = imass + lmass;
      const double bijsq = static_cast<double>(params.z);
      const double biksq = bijsq;
      const double bilsq = bijsq;

      const double xpij = coords[atoms.x].x - coords[atoms.y].x;
      const double ypij = coords[atoms.x].y - coords[atoms.y].y;
      const double zpij = coords[atoms.x].z - coords[atoms.y].z;
      const double xpik = coords[atoms.x].x - coords[atoms.z].x;
      const double ypik = coords[atoms.x].y - coords[atoms.z].y;
      const double zpik = coords[atoms.x].z - coords[atoms.z].z;
      const double xpil = coords[atoms.x].x - coords[atoms.w].x;
      const double ypil = coords[atoms.x].y - coords[atoms.w].y;
      const double zpil = coords[atoms.x].z - coords[atoms.w].z;

      const double xrij = coordsRef[atoms.x].x - coordsRef[atoms.y].x;
      const double yrij = coordsRef[atoms.x].y - coordsRef[atoms.y].y;
      const double zrij = coordsRef[atoms.x].z - coordsRef[atoms.y].z;
      const double xrik = coordsRef[atoms.x].x - coordsRef[atoms.z].x;
      const double yrik = coordsRef[atoms.x].y - coordsRef[atoms.z].y;
      const double zrik = coordsRef[atoms.x].z - coordsRef[atoms.z].z;
      const double xril = coordsRef[atoms.x].x - coordsRef[atoms.w].x;
      const double yril = coordsRef[atoms.x].y - coordsRef[atoms.w].y;
      const double zril = coordsRef[atoms.x].z - coordsRef[atoms.w].z;

      const double pijpijsq = xpij * xpij + ypij * ypij + zpij * zpij;
      const double pikpiksq = xpik * xpik + ypik * ypik + zpik * zpik;
      const double pilpilsq = xpil * xpil + ypil * ypil + zpil * zpil;
      const double rijrijsq = xrij * xrij + yrij * yrij + zrij * zrij;
      const double rijriksq = xrij * xrik + yrij * yrik + zrij * zrik;
      const double rijrilsq = xrij * xril + yrij * yril + zrij * zril;
      const double rikriksq = xrik * xrik + yrik * yrik + zrik * zrik;
      const double rikrilsq = xrik * xril + yrik * yril + zrik * zril;
      const double rilrilsq = xril * xril + yril * yril + zril * zril;

      const double rijpijsq = xrij * xpij + yrij * ypij + zrij * zpij;
      const double rijpiksq = xrij * xpik + yrij * ypik + zrij * zpik;
      const double rijpilsq = xrij * xpil + yrij * ypil + zrij * zpil;
      const double rikpijsq = xrik * xpij + yrik * ypij + zrik * zpij;
      const double rikpiksq = xrik * xpik + yrik * ypik + zrik * zpik;
      const double rikpilsq = xrik * xpil + yrik * ypil + zrik * zpil;
      const double rilpijsq = xril * xpij + yril * ypij + zril * zpij;
      const double rilpiksq = xril * xpik + yril * ypik + zril * zpik;
      const double rilpilsq = xril * xpil + yril * ypil + zril * zpil;

      const double dijsq = bijsq - pijpijsq;
      const double diksq = biksq - pikpiksq;
      const double dilsq = bilsq - pilpilsq;

      const double d1 = ikmass * ilmass * rikpiksq * rilpilsq -
                        imass * imass * rikpilsq * rilpiksq;
      const double d2 = imass * ilmass * rikpijsq * rilpilsq -
                        imass * imass * rikpilsq * rilpijsq;
      const double d3 = imass * imass * rikpijsq * rilpiksq -
                        ikmass * imass * rikpiksq * rilpijsq;
      const double d4 = imass * ilmass * rijpiksq * rilpilsq -
                        imass * imass * rijpilsq * rilpiksq;
      const double d5 = ijmass * ilmass * rijpijsq * rilpilsq -
                        imass * imass * rijpilsq * rilpijsq;
      const double d6 = ijmass * imass * rijpijsq * rilpiksq -
                        imass * imass * rijpiksq * rilpijsq;
      const double d7 = imass * imass * rijpiksq * rikpilsq -
                        imass * ikmass * rijpilsq * rikpiksq;
      const double d8 = ijmass * imass * rijpijsq * rikpilsq -
                        imass * imass * rijpilsq * rikpijsq;
      const double d9 = ijmass * ikmass * rijpijsq * rikpiksq -
                        imass * imass * rijpiksq * rikpijsq;
      const double dinv = 0.5 / (rijpijsq * ijmass * d1 -
                                 rijpiksq * imass * d2 + rijpilsq * imass * d3);
      const double acorr1 = ijmass * ijmass * rijrijsq;
      const double acorr2 = 2.0 * ijmass * imass * rijriksq;
      const double acorr3 = imass * imass * rikriksq;
      const double acorr4 = imass * imass * rijrijsq;
      const double acorr5 = 2.0 * ikmass * imass * rijriksq;
      const double acorr6 = ikmass * ikmass * rikriksq;
      const double acorr7 = 2.0 * ijmass * imass * rijrilsq;
      const double acorr8 = 2.0 * imass * imass * rikrilsq;
      const double acorr9 = imass * imass * rilrilsq;
      const double acorr10 = 2.0 * imass * imass * rijrilsq;
      const double acorr11 = 2.0 * imass * ikmass * rikrilsq;
      const double acorr12 = 2.0 * imass * imass * rijriksq;
      const double acorr13 = 2.0 * imass * ilmass * rijrilsq;
      const double acorr14 = 2.0 * imass * ilmass * rikrilsq;
      const double acorr15 = ilmass * ilmass * rilrilsq;

      double a12 = dinv * (d1 * dijsq - d2 * diksq + d3 * dilsq);
      double a13 = dinv * (-d4 * dijsq + d5 * diksq - d6 * dilsq);
      double a14 = dinv * (d7 * dijsq - d8 * diksq + d9 * dilsq);
      double a120 = 0.0, a130 = 0.0, a140 = 0.0;
      for (int ITER = 0; ITER < MAX_ITER; ITER++) {
        a120 = a12;
        a130 = a13;
        a140 = a14;

        const double a12corr = acorr1 * a12 * a12 + acorr2 * a12 * a13 +
                               acorr3 * a13 * a13 + acorr7 * a12 * a14 +
                               acorr8 * a13 * a14 + acorr9 * a14 * a14;
        const double a13corr = acorr4 * a12 * a12 + acorr5 * a12 * a13 +
                               acorr6 * a13 * a13 + acorr10 * a12 * a14 +
                               acorr11 * a13 * a14 + acorr9 * a14 * a14;
        const double a14corr = acorr4 * a12 * a12 + acorr12 * a12 * a13 +
                               acorr3 * a13 * a13 + acorr13 * a12 * a14 +
                               acorr14 * a13 * a14 + acorr15 * a14 * a14;

        a12 = dinv * (d1 * (dijsq - a12corr) - d2 * (diksq - a13corr) +
                      d3 * (dilsq - a14corr));
        a13 = dinv * (-d4 * (dijsq - a12corr) + d5 * (diksq - a13corr) -
                      d6 * (dilsq - a14corr));
        a14 = dinv * (d7 * (dijsq - a12corr) - d8 * (diksq - a13corr) +
                      d9 * (dilsq - a14corr));

        if ((abs(a120 - a12) < TOL) && (abs(a130 - a13) < TOL) &&
            (abs(a140 - a14) < TOL))
          break;
      }
    }
  }

  return;
}

void CudaHolonomicConstraint::constrainShakeAtoms(const double4 *coordsRef) {
  if (m_ShakeAtoms.size() > 0) {
    double4 *coords =
        m_Context->getCoordinatesCharges().getDeviceArray().data();

    constexpr int numThreads = 128;
    const int numBlocks =
        (static_cast<int>(m_ShakeAtoms.size()) + numThreads - 1) / numThreads;

    Shake2Kernel<<<numBlocks, numThreads, 0, *m_Stream>>>(
        coords, coordsRef, m_ShakeAtoms.getDeviceArray().data(),
        m_ShakeParams.getDeviceArray().data(),
        static_cast<int>(m_ShakeAtoms.size()));

    Shake3Kernel<<<numBlocks, numThreads, 0, *m_Stream>>>(
        coords, coordsRef, m_ShakeAtoms.getDeviceArray().data(),
        m_ShakeParams.getDeviceArray().data(),
        static_cast<int>(m_ShakeAtoms.size()));

    Shake4Kernel<<<numBlocks, numThreads, 0, *m_Stream>>>(
        coords, coordsRef, m_ShakeAtoms.getDeviceArray().data(),
        m_ShakeParams.getDeviceArray().data(),
        static_cast<int>(m_ShakeAtoms.size()));
  }

  return;
}

__global__ static void
UpdateVelocitiesKernel(double4 *__restrict__ velMass,
                       const double4 *__restrict__ coords,
                       const double4 *__restrict__ coordsStored,
                       const int numAtoms, const double invTimeStep) {

  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = index; i < numAtoms; i += stride) {
    const double x = coords[i].x;
    const double y = coords[i].y;
    const double z = coords[i].z;

    const double x0 = coordsStored[i].x;
    const double y0 = coordsStored[i].y;
    const double z0 = coordsStored[i].z;

    velMass[i].x += invTimeStep * (x - x0);
    velMass[i].y += invTimeStep * (y - y0);
    velMass[i].z += invTimeStep * (z - z0);
  }

  return;
}

void CudaHolonomicConstraint::updateVelocities(void) {
  constexpr int numThreads = 128;
  const int numBlocks =
      (m_Context->getNumAtoms() + numThreads - 1) / numThreads;

  UpdateVelocitiesKernel<<<numBlocks, numThreads, 0, *m_Stream>>>(
      m_Context->getVelocityMass().getDeviceArray().data(),
      m_Context->getCoordinatesCharges().getDeviceArray().data(),
      m_CoordsStored.getDeviceArray().data(), m_Context->getNumAtoms(),
      1.0 / m_TimeStep);

  return;
}
