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
#include <iostream>

//
// Direct-space virial calculation
//
__global__ void calcVirialKernel(const int ncoord,
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
    float4 xyzqi = xyzq[i];
    double x = (double)xyzqi.x;
    double y = (double)xyzqi.y;
    double z = (double)xyzqi.z;
    double fx = (double)force[i];
    double fy = (double)force[i + stride];
    double fz = (double)force[i + stride * 2];
    vir[0] = x * fx;
    vir[1] = x * fy;
    vir[2] = x * fz;
    vir[3] = y * fx;
    vir[4] = y * fy;
    vir[5] = y * fz;
    vir[6] = z * fx;
    vir[7] = z * fy;
    vir[8] = z * fz;
  } else if (ish >= 0 && ish <= 26) {
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
    for (int k = 0; k < 3; k++)
      sh_vir[threadIdx.x + k * blockDim.x] += vir_val[k];
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
    for (int k = 0; k < 3; k++)
      sh_vir[threadIdx.x + k * blockDim.x] += vir_val[k];
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
    for (int k = 0; k < 3; k++)
      sh_vir[threadIdx.x + k * blockDim.x] += vir_val[k];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k = 0; k < 3; k++)
      atomicAdd(&virial->virmat[k + 6], -sh_vir[k * blockDim.x]);
  }
}

// ###############################################################################################
// ###############################################################################################
// ###############################################################################################

//
// Class creator
//
CudaEnergyVirial::CudaEnergyVirial() {
  h_buffer = NULL;
  d_buffer = NULL;
  h_buffer_len = 0;
  d_buffer_len = 0;
}

//
// Class destructor
//
CudaEnergyVirial::~CudaEnergyVirial() {
  if (d_buffer != NULL)
    deallocate<char>(&d_buffer);
  if (h_buffer != NULL)
    deallocate_host<char>(&h_buffer);
}

//
// Make sure d_buffer & h_buffer is allocated and has enough space
//
void CudaEnergyVirial::reallocateBuffer() {
  // Buffer consists of:
  // 27*3 sforce_dp    (double)
  // 27*3 sforce_fp    (long long int)
  // 9    virial       (double)
  // n    energy terms (double)
  int buffer_len_req = 27 * 3 * sizeof(double) +
                       27 * 3 * sizeof(long long int) + 9 * sizeof(double) +
                       this->getN() * sizeof(double);
  reallocate<char>(&d_buffer, &d_buffer_len, buffer_len_req, 1.0f);
  reallocate_host<char>(&h_buffer, &h_buffer_len, buffer_len_req, 1.0f);
}

//
// Clears (sets to zero) energies and virials
//
void CudaEnergyVirial::clear(cudaStream_t stream) {
  this->reallocateBuffer();
  clear_gpu_array<char>(d_buffer, d_buffer_len, stream);
  memset(h_buffer, 0, h_buffer_len);
}

//
// Clears (sets to zero) energies
//
void CudaEnergyVirial::clearEnergy(cudaStream_t stream) {
  this->reallocateBuffer();
  int clear_pos = 27 * 3 * sizeof(double) + 27 * 3 * sizeof(long long int) +
                  9 * sizeof(double);
  int clear_len = this->getN() * sizeof(double);
  clear_gpu_array<char>(&d_buffer[clear_pos], clear_len, stream);
  memset(&h_buffer[clear_pos], 0, clear_len);
}

//
// Clears (sets to zero) a specified energy
//
void CudaEnergyVirial::clearEtermDevice(std::string &nameEterm,
                                        cudaStream_t stream) {
  this->reallocateBuffer();
  int clear_pos = 27 * 3 * sizeof(double) + 27 * 3 * sizeof(long long int) +
                  9 * sizeof(double) +
                  this->getEnergyIndex(nameEterm) * sizeof(double);
  int clear_len = sizeof(double);
  clear_gpu_array<char>(&d_buffer[clear_pos], clear_len, stream);
}

//
// Clears (sets to zero) virials
//
void CudaEnergyVirial::clearVirial(cudaStream_t stream) {
  this->reallocateBuffer();
  int clear_len = 27 * 3 * sizeof(double) + 27 * 3 * sizeof(long long int) +
                  9 * sizeof(double);
  clear_gpu_array<char>(d_buffer, clear_len, stream);
  memset(h_buffer, 0, clear_len);
}

//
// Calculates virial
//
void CudaEnergyVirial::calcVirial(const int ncoord, const float4 *xyzq,
                                  const double boxx, const double boxy,
                                  const double boxz, const int stride,
                                  const double *force, cudaStream_t stream) {
  this->reallocateBuffer();

  int nthread, nblock, shmem_size;
  nthread = 256;
  nblock = (ncoord + 27 - 1) / nthread + 1;
  shmem_size = nthread * 3 * sizeof(double);

  calcVirialKernel<<<nblock, nthread, shmem_size, stream>>>(
      ncoord, xyzq, boxx, boxy, boxz, stride, force, this->getVirialPointer());

  cudaCheck(cudaGetLastError());
}

//
// Copies energy and virial values to host
//
void CudaEnergyVirial::copyToHost(cudaStream_t stream) {
  this->reallocateBuffer();
  copy_DtoH<char>(d_buffer, h_buffer, d_buffer_len, stream);
}

//
// Returns device pointer to energy term "name"
//
double *CudaEnergyVirial::getEnergyPointer(std::string &name) {
  this->reallocateBuffer();
  return (
      double *)(&d_buffer[27 * 3 * sizeof(double) +
                          27 * 3 * sizeof(long long int) + 9 * sizeof(double) +
                          this->getEnergyIndex(name) * sizeof(double)]);
}

double *CudaEnergyVirial::getEnergyPointer(const char *nameIn) {
  std::string name{nameIn};
  return getEnergyPointer(name);
}
//
// Return device pointer to the Virial_t -structure
//
Virial_t *CudaEnergyVirial::getVirialPointer() {
  this->reallocateBuffer();
  return (Virial_t *)d_buffer;
}

//
// Return value of energy called "name"
//
double CudaEnergyVirial::getEnergy(std::string &name) {
  this->reallocateBuffer();
  double *p =
      (double *)(&h_buffer[27 * 3 * sizeof(double) +
                           27 * 3 * sizeof(long long int) + 9 * sizeof(double) +
                           this->getEnergyIndex(name) * sizeof(double)]);
  return *p;
}

double CudaEnergyVirial::getEnergy(const char *name) {
  std::string str(name);
  return this->getEnergy(str);
}

//
// Return the virial 3x3 matrix
//
void CudaEnergyVirial::getVirial(double *virmat) {
  this->reallocateBuffer();
  double *p = (double *)(&h_buffer[27 * 3 * sizeof(double) +
                                   27 * 3 * sizeof(long long int)]);
  for (int i = 0; i < 9; i++)
    virmat[i] = p[i];
}

void CudaEnergyVirial::getVirial(CudaContainer<double> &virial) {
  copy_DtoD_sync<double>(this->getVirialPointer()->virmat,
                         virial.getDeviceArray().data(), 9);
}
/*
std::vector<double> CudaEnergyVirial::calculateVirial() {
  copyToHost();
  std::vector<double> virmat(9);
  this->reallocateBuffer();
  double *p = (double *)(&h_buffer[27 * 3 * sizeof(double) +
                                   27 * 3 * sizeof(long long int)]);
  for (int i = 0; i < 9; i++)
    virmat[i] = p[i];
  return virmat;
}
*/
//
// Return sforce 27*3 array
//
void CudaEnergyVirial::getSforce(double *sforce) {
  this->reallocateBuffer();
  Virial_t *virial = (Virial_t *)(&h_buffer[0]);
  for (int i = 0; i < 27; i++) {
    for (int d = 0; d < 3; d++) {
      sforce[i * 3 + d] =
          virial->sforce_dp[i][d] +
          ((double)virial->sforce_fp[i][d]) * INV_FORCE_SCALE_VIR_CPU;
    }
  }
}

__global__ void addPotentialEnergiesKernel(int otherN,
                                           double *__restrict__ otherBuffer,
                                           double *__restrict__ selfBuffer) {
  for (int i = 0; i < otherN; ++i) {
    // add the ith enegy from other to self
    selfBuffer[0] = *(otherBuffer + i);
  }
}

void CudaEnergyVirial::addPotentialEnergies(const CudaEnergyVirial &other) {
  assert(getN() == 1);
  int otherN = other.getN();
  // addPotentialEnergiesKernel<<<1, 1>>>(otherN,
  // other.getEnergyPointer("ewex"),
  //                                      getEnergyPointer("total"));
  // cudaCheck(cudaDeviceSynchronize());
}
#endif // NOCUDAC
