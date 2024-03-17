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
#include "XYZQ.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

//
// XYZQ class method definitions
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
//

//
// Copies x, y, z coordinates into xyzq -array
//
__global__ void set_xyz_kernel(const int ncoord, const double *__restrict__ x,
                               const double *__restrict__ y,
                               const double *__restrict__ z,
                               float4 *__restrict__ xyzq) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < ncoord) {
    xyzq[tid].x = x[tid];
    xyzq[tid].y = y[tid];
    xyzq[tid].z = z[tid];
  }
}

//
// Copies (x, y, z, q) into xyzq -array
//
__global__ void set_xyzq_kernel(const int ncoord, const double *__restrict__ x,
                                const double *__restrict__ y,
                                const double *__restrict__ z,
                                const float *__restrict__ q,
                                float4 *__restrict__ xyzq) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < ncoord) {
    float4 xyzq_val;
    xyzq_val.x = x[tid];
    xyzq_val.y = y[tid];
    xyzq_val.z = z[tid];
    xyzq_val.w = q[tid];
    xyzq[tid] = xyzq_val;
  }
}

//
// Copies (x, y, z, q) into xyzq -array and also shifts (x, y, z)
//
__global__ void set_xyzq_shift_kernel(
    const int ncoord, const double *__restrict__ x,
    const double *__restrict__ y, const double *__restrict__ z,
    const float *__restrict__ q, const int *__restrict__ loc2glo,
    const float3 *__restrict__ xyz_shift, const double boxx, const double boxy,
    const double boxz, float4 *__restrict__ xyzq) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < ncoord) {
    float4 xyzq_val;
    float3 shift = xyz_shift[tid];
    xyzq_val.x = (float)(x[tid] + ((double)shift.x) * boxx);
    xyzq_val.y = (float)(y[tid] + ((double)shift.y) * boxy);
    xyzq_val.z = (float)(z[tid] + ((double)shift.z) * boxz);
    xyzq_val.w = q[loc2glo[tid]];
    xyzq[tid] = xyzq_val;
  }
}

//
// Copies (x, y, z) into xyzq -array and also shifts (x, y, z)
//
__global__ void
set_xyz_shift_kernel(const int ncoord, const double *__restrict__ x,
                     const double *__restrict__ y, const double *__restrict__ z,
                     const float3 *__restrict__ xyz_shift, const double boxx,
                     const double boxy, const double boxz,
                     float4 *__restrict__ xyzq) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < ncoord) {
    float4 xyzq_val;
    float3 shift = xyz_shift[tid];
    xyzq_val.x = (float)(x[tid] + ((double)shift.x) * boxx);
    xyzq_val.y = (float)(y[tid] + ((double)shift.y) * boxy);
    xyzq_val.z = (float)(z[tid] + ((double)shift.z) * boxz);
    xyzq[tid].x = xyzq_val.x;
    xyzq[tid].y = xyzq_val.y;
    xyzq[tid].z = xyzq_val.z;
  }
}

//##########################################################################################
//##########################################################################################
//##########################################################################################

//
// Return xyzq length that has extra align:
// ncoord-1 = last possible index
//
int XYZQ::get_xyzq_len(const int ncoord_in) {
  return ((ncoord_in - 1) / align + 1) * align;
}

__global__ void clear(int size, float4 *xyzq) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < size) {
    xyzq[index].x = 0.0f;
    xyzq[index].y = 0.0f;
    xyzq[index].z = 0.0f;
    xyzq[index].w = 0.0f;
  }
}

void XYZQ::set_ncoord(const int ncrd) {
  align = warpsize;
  ncoord = ncrd;
  xyzq_len = get_xyzq_len(ncoord);
  allocate<float4>(&xyzq, xyzq_len);

  int numThreads = 128;
  int numBlocks = (ncoord - 1) / numThreads + 1;
  clear<<<numBlocks, numThreads>>>(ncoord, xyzq);
  cudaDeviceSynchronize();

  host_xyzq =
      std::make_shared<std::vector<float4>>(std::vector<float4>(ncoord));
  // printf("Size of host array is : %d ", host_xyzq->size());
  // host_xyzq->resize(xyzq_len);
}

//
// Class creator
//
XYZQ::XYZQ() {
  ncoord = 0;
  xyzq_len = 0;
  align = warpsize;
  xyzq = NULL;
}

//
// Class creator
//
XYZQ::XYZQ(int ncoord, int align) : ncoord(ncoord), align(align) {
  xyzq_len = get_xyzq_len(ncoord);
  allocate<float4>(&xyzq, xyzq_len);
  host_xyzq->resize(xyzq_len);
}

//
// Class creator
//
XYZQ::XYZQ(const char *filename, int align) : align(align) {
  std::ifstream file(filename);
  if (file.is_open()) {
    float x, y, z, q;

    // Count number of coordinates
    ncoord = 0;
    while (file >> x >> y >> z >> q)
      ncoord++;

    // Rewind
    file.clear();
    file.seekg(0, std::ios::beg);

    // Allocate CPU memory
    float4 *xyzq_cpu = new float4[ncoord];

    // Read coordinates
    int i = 0;
    while (file >> xyzq_cpu[i].x >> xyzq_cpu[i].y >> xyzq_cpu[i].z >>
           xyzq_cpu[i].w)
      i++;

    // Allocate GPU memory
    xyzq_len = get_xyzq_len(ncoord);
    allocate<float4>(&xyzq, xyzq_len);

    // Copy coordinates from CPU to GPU
    copy_HtoD<float4>(xyzq_cpu, xyzq, ncoord);

    // Deallocate CPU memory
    delete[] xyzq_cpu;

  } else {
     std::stringstream tmpexc; 
    tmpexc << "Error opening file " << filename << std::endl;
    throw std::invalid_argument(tmpexc.str());
    exit(1);
  }
}

//
// Class destructor
//
XYZQ::~XYZQ() {
  if (xyzq != NULL)
    deallocate<float4>(&xyzq);
}

//
// Re-allocates array, does not preserve content
//
void XYZQ::realloc(int ncoord_new, float fac) {
  reallocate<float4>(&xyzq, &xyzq_len, get_xyzq_len(ncoord_new), fac);
  this->ncoord = ncoord_new;
}

//
// Re-sizes array, preserves content
//
void XYZQ::resize(int ncoord_new, float fac) {
  ::resize<float4>(&xyzq, &xyzq_len, ncoord, get_xyzq_len(ncoord_new), fac);
  this->ncoord = ncoord_new;
}

//
// Copies xyzq from host
// NOTE: Does not reallocate xyzq
//
void XYZQ::set_xyzq(int ncopy, const float4 *h_xyzq, size_t offset,
                    cudaStream_t stream) {
  copy_HtoD<float4>(&h_xyzq[offset], &xyzq[offset], ncopy, stream);
}

//
// Copies x,y,z,q (on device) into the coordinate slots
//
void XYZQ::set_xyzq(const cudaXYZ<double> &coord, const float *q,
                    cudaStream_t stream) {
  assert(ncoord == coord.size());

  int nthread = 512;
  int nblock = (ncoord - 1) / nthread + 1;

  set_xyzq_kernel<<<nblock, nthread, 0, stream>>>(
      coord.size(), coord.x(), coord.y(), coord.z(), q, xyzq);

  cudaCheck(cudaGetLastError());
}

//
// Copies x,y,z,q (on device) into the coordinate slots
//
void XYZQ::set_xyzq(const cudaXYZ<double> &coord, const float *q,
                    const int *loc2glo, const float3 *xyz_shift,
                    const double boxx, const double boxy, const double boxz,
                    cudaStream_t stream) {
  assert(ncoord == coord.size());

  int nthread = 512;
  int nblock = (ncoord - 1) / nthread + 1;

  set_xyzq_shift_kernel<<<nblock, nthread, 0, stream>>>(
      coord.size(), coord.x(), coord.y(), coord.z(), q, loc2glo, xyz_shift,
      boxx, boxy, boxz, xyzq);

  cudaCheck(cudaGetLastError());
}

//
// Copies x,y,z (on device) into the coordinate slots
//
void XYZQ::set_xyz(const cudaXYZ<double> &coord, cudaStream_t stream) {
  assert(ncoord == coord.size());

  int nthread = 512;
  int nblock = (ncoord - 1) / nthread + 1;

  set_xyz_kernel<<<nblock, nthread, 0, stream>>>(coord.size(), coord.x(),
                                                 coord.y(), coord.z(), xyzq);

  cudaCheck(cudaGetLastError());
}

//
// Copies x,y,z,q (on device) into the coordinate slots
//
void XYZQ::set_xyz(const cudaXYZ<double> &coord, const int start, const int end,
                   const float3 *xyz_shift, const double boxx,
                   const double boxy, const double boxz, cudaStream_t stream) {
  assert(ncoord == coord.size());
  assert(start >= 0);
  assert(end < ncoord);
  assert(end < coord.size());
  int nset = end - start + 1;
  assert(nset >= 0);

  if (nset == 0)
    return;

  int nthread = 512;
  int nblock = (nset - 1) / nthread + 1;

  set_xyz_shift_kernel<<<nblock, nthread, 0, stream>>>(
      nset, coord.x() + start, coord.y() + start, coord.z() + start,
      &xyz_shift[start], boxx, boxy, boxz, &xyzq[start]);

  cudaCheck(cudaGetLastError());
}

//
// Compares two XYZQ arrays
//
bool XYZQ::compare(XYZQ &xyzq_in, const double tol, double &max_diff) {
  assert(xyzq_in.ncoord == ncoord);

  float4 *h_xyzq = new float4[ncoord];
  float4 *h_xyzq_in = new float4[ncoord];
  copy_DtoH<float4>(xyzq, h_xyzq, ncoord);
  copy_DtoH<float4>(xyzq_in.xyzq, h_xyzq_in, ncoord);

  bool ok = true;

  max_diff = 0.0;
  int i;
  double dx, dy, dz, dq;
  double diff;
  try {
    for (i = 0; i < ncoord; i++) {
      dx = fabs(h_xyzq[i].x - h_xyzq_in[i].x);
      dy = fabs(h_xyzq[i].y - h_xyzq_in[i].y);
      dz = fabs(h_xyzq[i].z - h_xyzq_in[i].z);
      dq = fabs(h_xyzq[i].w - h_xyzq_in[i].w);
      diff = max(dx, max(dy, dz));
      max_diff = max(max_diff, diff);
      if (diff > tol || dq > 0.0)
        throw 1;
    }
  } catch (int a) {
    std::cout << "i = " << i << std::endl;
    std::cout << "this: x,y,z,q = " << h_xyzq[i].x << " " << h_xyzq[i].y << " "
              << h_xyzq[i].z << " " << h_xyzq[i].w << std::endl;
    std::cout << "in  : x,y,z,q = " << h_xyzq_in[i].x << " " << h_xyzq_in[i].y
              << " " << h_xyzq_in[i].z << " " << h_xyzq_in[i].w << std::endl;
    ok = false;
  }

  delete[] h_xyzq;
  delete[] h_xyzq_in;

  return ok;
}

//
// Print to ostream
//
void XYZQ::print(const int start, const int end, std::ostream &out) {
  float4 *h_xyzq = new float4[ncoord];
  copy_DtoH_sync<float4>(xyzq, h_xyzq, ncoord);

  for (int i = start; i <= end; i++) {
    out << i << " " << h_xyzq[i].x << " " << h_xyzq[i].y << " " << h_xyzq[i].z
        << " " << h_xyzq[i].w << std::endl;
  }

  delete[] h_xyzq;
}

//
// Save to file
//
void XYZQ::save(const char *filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    float4 *h_xyzq = new float4[ncoord];
    copy_DtoH_sync<float4>(xyzq, h_xyzq, ncoord);
    for (int i = 0; i < ncoord; i++) {
      file << h_xyzq[i].x << " " << h_xyzq[i].y << " " << h_xyzq[i].z << " "
           << h_xyzq[i].w << std::endl;
    }
    delete[] h_xyzq;
  } else {
     std::stringstream tmpexc; 
    tmpexc << "Error opening file " << filename << std::endl;
    throw std::invalid_argument(tmpexc.str());
    exit(1);
  }
}

float4 *XYZQ::getDeviceXYZQ() { return xyzq; }

void XYZQ::transferFromDevice() {
  cudaMemcpy(host_xyzq->data(), xyzq, sizeof(float4) * host_xyzq->size(),
             cudaMemcpyDeviceToHost);
}

void XYZQ::transferToDevice() {
  cudaMemcpy(xyzq, host_xyzq->data(), sizeof(float4) * host_xyzq->size(),
             cudaMemcpyHostToDevice);
}

std::shared_ptr<std::vector<float4>> XYZQ::getHostXYZQ() {
  transferFromDevice();
  return host_xyzq;
}

void XYZQ::setDeviceXYZQ(float4 *in) { xyzq = in; }

void XYZQ::set_xyz(const std::vector<float> &coords) {
  transferFromDevice();
  for (int i = 0; i < ncoord; i++) {
    host_xyzq->at(i).x = coords[i * 3];
    host_xyzq->at(i).y = coords[i * 3 + 1];
    host_xyzq->at(i).z = coords[i * 3 + 2];
  }
  transferToDevice();
}

std::vector<float> XYZQ::get_xyz() {
  transferFromDevice();
  std::vector<float> coords(ncoord * 3);
  for (int i = 0; i < ncoord; i++) {
    coords[i * 3] = host_xyzq->at(i).x;
    coords[i * 3 + 1] = host_xyzq->at(i).y;
    coords[i * 3 + 2] = host_xyzq->at(i).z;
  }
  return coords;
}

std::vector<float> XYZQ::get_q() {
   transferFromDevice();
   std::vector<float> qs(ncoord);
   for (int i = 0; i < ncoord; i++) {
      qs[i] = host_xyzq->at(i).w;
   }
   return qs;
}

__global__ static void imageCenterKernel(float3 boxSize, int numGroups,
                                         const int2 *__restrict__ groups,
                                         float4 *__restrict__ xyzq) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index < numGroups) {
    int2 group = groups[index];

    float gx = 0.0;
    float gy = 0.0;
    float gz = 0.0;
    for (int i = group.x; i <= group.y; ++i) {
      gx += xyzq[i].x;
      gy += xyzq[i].y;
      gz += xyzq[i].z;
    }
    gx /= (group.y - group.x + 1);
    gy /= (group.y - group.x + 1);
    gz /= (group.y - group.x + 1);

    if (gx < -0.5 * boxSize.x) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].x += boxSize.x;
      }
    }

    if (gx > 0.5 * boxSize.x) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].x -= boxSize.x;
      }
    }

    if (gy < -0.5 * boxSize.y) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].y += boxSize.y;
      }
    }

    if (gy > 0.5 * boxSize.y) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].y -= boxSize.y;
      }
    }

    if (gz < -0.5 * boxSize.z) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].z += boxSize.z;
      }
    }

    if (gz > 0.5 * boxSize.z) {
      for (int i = group.x; i <= group.y; ++i) {
        xyzq[i].z -= boxSize.z;
      }
    }
  }
}

void XYZQ::imageCenter(const std::vector<float> &boxDimensions,
                       // CudaContainer<int4> &groups) {
                       CudaContainer<int2> &groups) {

  int numGroups = groups.size();
  int numThreads = 128;
  int numBlocks = (numGroups - 1) / numThreads + 1;

  float3 boxSize = {boxDimensions[0], boxDimensions[1], boxDimensions[2]};
  // auto dv = groups.getDeviceArray().data();
  imageCenterKernel<<<numBlocks, numThreads>>>(
      boxSize, numGroups, groups.getDeviceArray().data(), xyzq);
  cudaCheck(cudaDeviceSynchronize());
}

void XYZQ::orient(bool massWeighting, bool rotation) {}

#endif // NOCUDAC
