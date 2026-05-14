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

#include "CudaEnergyVirial.h"

#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cassert>
#include <cstring>
#include <iostream>

//
// Class creator
//
CudaEnergyVirial::CudaEnergyVirial(void) : EnergyVirial() {
  m_HostBufferLength = 0;
  m_HostBuffer = NULL;
  m_DeviceBufferLength = 0;
  m_DeviceBuffer = NULL;
}

//
// Class destructor
//
CudaEnergyVirial::~CudaEnergyVirial(void) { this->deallocateBuffer(); }

//
// Clears (sets to zero) energies and virials
//
void CudaEnergyVirial::clear(cudaStream_t stream) {
  this->reallocateBuffer();
  clear_gpu_array<char>(m_DeviceBuffer, m_DeviceBufferLength, stream);
  std::memset(static_cast<void *>(m_HostBuffer), 0,
              static_cast<std::size_t>(m_HostBufferLength));
  return;
}

//
// Clears (sets to zero) energies
//
void CudaEnergyVirial::clearEnergy(cudaStream_t stream) {
  this->reallocateBuffer();
  const int pos = 27 * 3 * sizeof(double) + 27 * 3 * sizeof(long long int) +
                  9 * sizeof(double);
  const int len = this->getN() * sizeof(double);
  clear_gpu_array<char>(m_DeviceBuffer + pos, len, stream);
  std::memset(static_cast<void *>(m_HostBuffer + pos), 0,
              static_cast<std::size_t>(len));
  return;
}

//
// Clears (sets to zero) a specified energy
//
void CudaEnergyVirial::clearEtermDevice(const std::string &name,
                                        cudaStream_t stream) {
  this->reallocateBuffer();
  const int pos = 27 * 3 * sizeof(double) + 27 * 3 * sizeof(long long int) +
                  9 * sizeof(double) +
                  this->getEnergyIndex(name) * sizeof(double);
  const int len = sizeof(double);
  clear_gpu_array<char>(m_DeviceBuffer + pos, len, stream);
  return;
}

//
// Clears (sets to zero) virials
//
void CudaEnergyVirial::clearVirial(cudaStream_t stream) {
  this->reallocateBuffer();
  int len = 27 * 3 * sizeof(double) + 27 * 3 * sizeof(long long int) +
            9 * sizeof(double);
  clear_gpu_array<char>(m_DeviceBuffer, len, stream);
  std::memset(static_cast<void *>(m_HostBuffer), 0,
              static_cast<std::size_t>(len));
  return;
}

//
// Direct-space virial calculation
//
__global__ static void calcVirialKernel(const int ncoord,
                                        const float4 *__restrict__ xyzq,
                                        const double boxx, const double boxy,
                                        const double boxz, const int stride,
                                        const double *__restrict__ force,
                                        Virial_t *__restrict__ virial) {
  // Shared memory:
  // blockDim.x*3*sizeof(double)
  extern __shared__ volatile double sh_vir[];

  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  int ish = i - ncoord;

  double vir[9];
  if (i < ncoord) {
    const float4 xyzqi = xyzq[i];
    const double x = static_cast<double>(xyzqi.x);
    const double y = static_cast<double>(xyzqi.y);
    const double z = static_cast<double>(xyzqi.z);
    const double fx = static_cast<double>(force[0 * stride + i]);
    const double fy = static_cast<double>(force[1 * stride + i]);
    const double fz = static_cast<double>(force[2 * stride + i]);
    vir[0] = x * fx;
    vir[1] = x * fy;
    vir[2] = x * fz;
    vir[3] = y * fx;
    vir[4] = y * fy;
    vir[5] = y * fz;
    vir[6] = z * fx;
    vir[7] = z * fy;
    vir[8] = z * fz;
  } else if ((ish >= 0) && (ish <= 26)) {
    double sforcex = virial->sforce_dp[ish][0] +
                     ((double)virial->sforce_fp[ish][0]) * INV_FORCE_SCALE_VIR;
    double sforcey = virial->sforce_dp[ish][1] +
                     ((double)virial->sforce_fp[ish][1]) * INV_FORCE_SCALE_VIR;
    double sforcez = virial->sforce_dp[ish][2] +
                     ((double)virial->sforce_fp[ish][2]) * INV_FORCE_SCALE_VIR;
    double shx, shy, shz;
    calc_box_shift<double>(ish, boxx, boxy, boxz, shx, shy, shz);
    vir[0] = shx * sforcex;
    vir[1] = shx * sforcey;
    vir[2] = shx * sforcez;
    vir[3] = shy * sforcex;
    vir[4] = shy * sforcey;
    vir[5] = shy * sforcez;
    vir[6] = shz * sforcex;
    vir[7] = shz * sforcey;
    vir[8] = shz * sforcez;
  } else {
#pragma unroll
    for (int k = 0; k < 9; k++)
      vir[k] = 0.0;
  }

// Reduce
// 0-2
#pragma unroll
  for (int k = 0; k < 3; k++)
    sh_vir[threadIdx.x + k * blockDim.x] = vir[k];
  __syncthreads();
  for (int i = 1; i < blockDim.x; i *= 2) {
    int pos = threadIdx.x + i;
    double vir_val[3];
#pragma unroll
    for (int k = 0; k < 3; k++)
      vir_val[k] = (pos < blockDim.x) ? sh_vir[pos + k * blockDim.x] : 0.0;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 3; k++) {
      sh_vir[threadIdx.x + k * blockDim.x] =
          sh_vir[threadIdx.x + k * blockDim.x] + vir_val[k];
      // sh_vir[threadIdx.x + k * blockDim.x] += vir_val[k];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k = 0; k < 3; k++)
      atomicAdd(&virial->virmat[k], -sh_vir[k * blockDim.x]);
  }

// 3-5
#pragma unroll
  for (int k = 0; k < 3; k++)
    sh_vir[threadIdx.x + k * blockDim.x] = vir[k + 3];
  __syncthreads();
  for (int i = 1; i < blockDim.x; i *= 2) {
    int pos = threadIdx.x + i;
    double vir_val[3];
#pragma unroll
    for (int k = 0; k < 3; k++)
      vir_val[k] = (pos < blockDim.x) ? sh_vir[pos + k * blockDim.x] : 0.0;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 3; k++) {
      sh_vir[threadIdx.x + k * blockDim.x] =
          sh_vir[threadIdx.x + k * blockDim.x] + vir_val[k];
      // sh_vir[threadIdx.x + k * blockDim.x] += vir_val[k];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k = 0; k < 3; k++)
      atomicAdd(&virial->virmat[k + 3], -sh_vir[k * blockDim.x]);
  }

// 6-8
#pragma unroll
  for (int k = 0; k < 3; k++)
    sh_vir[threadIdx.x + k * blockDim.x] = vir[k + 6];
  __syncthreads();
  for (int i = 1; i < blockDim.x; i *= 2) {
    int pos = threadIdx.x + i;
    double vir_val[3];
#pragma unroll
    for (int k = 0; k < 3; k++)
      vir_val[k] = (pos < blockDim.x) ? sh_vir[pos + k * blockDim.x] : 0.0;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 3; k++) {
      sh_vir[threadIdx.x + k * blockDim.x] =
          sh_vir[threadIdx.x + k * blockDim.x] + vir_val[k];
      // sh_vir[threadIdx.x + k * blockDim.x] += vir_val[k];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k = 0; k < 3; k++)
      atomicAdd(&virial->virmat[k + 6], -sh_vir[k * blockDim.x]);
  }

  return;
}

//
// Calculates virial
//
void CudaEnergyVirial::calcVirial(const int ncoord, const float4 *xyzq,
                                  const double boxx, const double boxy,
                                  const double boxz, const int stride,
                                  const double *force, cudaStream_t stream) {
  this->reallocateBuffer();

  constexpr int nthread = 256;
  const int nblock = (ncoord + 27 + nthread - 1) / nthread;
  constexpr int shmem_size = nthread * 3 * sizeof(double);

  calcVirialKernel<<<nblock, nthread, shmem_size, stream>>>(
      ncoord, xyzq, boxx, boxy, boxz, stride, force, this->getVirialPointer());

  cudaCheck(cudaGetLastError());

  return;
}

//
// Copies energy and virial values to host
//
void CudaEnergyVirial::copyToHost(cudaStream_t stream) {
  this->reallocateBuffer();
  copy_DtoH<char>(m_DeviceBuffer, m_HostBuffer, m_DeviceBufferLength, stream);
  return;
}

//
// Return device pointer to the Virial_t -structure
//
Virial_t *CudaEnergyVirial::getVirialPointer(void) {
  this->reallocateBuffer();
  return reinterpret_cast<Virial_t *>(m_DeviceBuffer);
}

//
// Returns device pointer to energy term "name"
//
double *CudaEnergyVirial::getEnergyPointer(const int idx) {
  this->reallocateBuffer();
  const int pos = 27 * 3 * sizeof(double) + 27 * 3 * sizeof(long long int) +
                  9 * sizeof(double) + idx * sizeof(double);
  return reinterpret_cast<double *>(m_DeviceBuffer + pos);
}

double *CudaEnergyVirial::getEnergyPointer(const std::string &name) {
  return this->getEnergyPointer(this->getEnergyIndex(name));
}

double *CudaEnergyVirial::getEnergyPointer(const char *name) {
  return this->getEnergyPointer(std::string(name));
}

//
// Return the virial 3x3 matrix
//
void CudaEnergyVirial::getVirial(double *virmat) {
  this->reallocateBuffer();
  const int pos = 27 * 3 * sizeof(double) + 27 * 3 * sizeof(long long int);
  const double *p = reinterpret_cast<double *>(m_HostBuffer + pos);
  for (int i = 0; i < 9; i++)
    virmat[i] = p[i];
  return;
}

void CudaEnergyVirial::getVirial(CudaContainer<double> &virial) {
  copy_DtoD_sync<double>(this->getVirialPointer()->virmat,
                         virial.getDeviceArray().data(), 9);
  return;
}

//
// Return value of energy called "name"
//
double CudaEnergyVirial::getEnergy(const int idx) {
  this->reallocateBuffer();
  const int pos = 27 * 3 * sizeof(double) + 27 * 3 * sizeof(long long int) +
                  9 * sizeof(double) + idx * sizeof(double);
  const double *ptr = reinterpret_cast<double *>(m_HostBuffer + pos);
  return *ptr;
}

double CudaEnergyVirial::getEnergy(const std::string &name) {
  return this->getEnergy(this->getEnergyIndex(name));
}

double CudaEnergyVirial::getEnergy(const char *name) {
  return this->getEnergy(std::string(name));
}

//
// Return sforce 27*3 array
//
void CudaEnergyVirial::getSforce(double *sforce) {
  this->reallocateBuffer();
  const Virial_t *virial = reinterpret_cast<Virial_t *>(m_HostBuffer);
  for (int i = 0; i < 27; i++) {
    for (int d = 0; d < 3; d++) {
      sforce[i * 3 + d] = virial->sforce_dp[i][d] +
                          static_cast<double>(virial->sforce_fp[i][d]) *
                              INV_FORCE_SCALE_VIR_CPU;
    }
  }
  return;
}

/* *
__global__ static void
AddPotentialEnergiesKernel(double *__restrict__ dst,
                           const double *__restrict__ src, const int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  double tmp = 0.0;
  for (int i = index; i < n; i += stride) {
    // add the ith enegy from other to self
    tmp += src[i];
    // selfBuffer[0] = *(otherBuffer + i);
  }

  sum = BlockReduceSum<double>(sum);

  if (threadIdx.x == 0)
    atomicAdd(dst, sum);

  return;
}

void CudaEnergyVirial::addPotentialEnergies(const CudaEnergyVirial &other) {
  assert(getN() == 1);
  int otherN = other.getN();
  // addPotentialEnergiesKernel<<<1, 1>>>(otherN,
  // other.getEnergyPointer("ewex"),
  //                                      getEnergyPointer("total"));
  // cudaCheck(cudaDeviceSynchronize());
}
* */

//
// Make sure d_buffer & h_buffer is allocated and has enough space
//
void CudaEnergyVirial::reallocateBuffer(void) {
  // Buffer consists of:
  // 27*3 sforce_dp    (double)
  // 27*3 sforce_fp    (long long int)
  // 9    virial       (double)
  // n    energy terms (double)
  const int len = 27 * 3 * sizeof(double) + 27 * 3 * sizeof(long long int) +
                  9 * sizeof(double) + this->getN() * sizeof(double);
  reallocate<char>(&m_DeviceBuffer, &m_DeviceBufferLength, len, 1.0f);
  reallocate_host<char>(&m_HostBuffer, &m_HostBufferLength, len, 1.0f);
  return;
}

//
// Safely deallocate memory
//
void CudaEnergyVirial::deallocateBuffer(void) {
  if (m_HostBuffer != NULL) {
    m_HostBufferLength = 0;
    deallocate_host<char>(&m_HostBuffer);
  }

  if (m_DeviceBuffer != NULL) {
    m_DeviceBufferLength = 0;
    deallocate<char>(&m_DeviceBuffer);
  }

  return;
}

#endif // NOCUDAC
