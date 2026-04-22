// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#ifndef NOCUDAC
#include "Force.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "reduce.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
Force<T>::Force(void) : m_Size(0), m_Stride(0), m_Capacity(0), m_XYZ(nullptr) {}

template <typename T> Force<T>::Force(const int size) : Force<T>() {
  this->realloc(size);
}

template <typename T> Force<T>::Force(const char *filename) : Force<T>() {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::invalid_argument("Error opening file " + std::string(filename));

  T fx, fy, fz;

  // Count number of coordinates
  int nforce = 0;
  while (file >> fx >> fy >> fz)
    nforce++;

  // Rewind
  file.clear();
  file.seekg(0, std::ios::beg);

  // Allocate CPU memory
  std::vector<T> x(nforce), y(nforce), z(nforce);

  // Read coordinates
  int i = 0;
  while (file >> x[i] >> y[i] >> z[i])
    i++;

  // Allocate GPU memory
  this->realloc(nforce);

  // Copy coordinates from CPU to GPU
  copy_HtoD_sync<T>(x.data(), this->x(), nforce);
  copy_HtoD_sync<T>(y.data(), this->y(), nforce);
  copy_HtoD_sync<T>(z.data(), this->z(), nforce);
}

template <typename T> Force<T>::~Force(void) {
  m_Size = 0;
  m_Stride = 0;
  m_Capacity = 0;
  if (m_XYZ != nullptr) {
    deallocate<T>(&m_XYZ);
    m_XYZ = nullptr;
  }
}

template <typename T> void Force<T>::clear(cudaStream_t stream) {
  clear_gpu_array<T>(m_XYZ, 3 * m_Stride, stream);
  return;
}

//
// Compares two force arrays, returns true if the difference is within tolerance
// NOTE: Comparison is done in double precision
//
template <typename T>
bool Force<T>::compare(Force<T> &force, const double tol, double &max_diff) {
  assert(force.size() == this->size());

  std::vector<T> xyz1(this->size()), xyz2(force.size());
  copy_DtoH_sync<T>(this->xyz(), xyz1.data(), this->size());
  copy_DtoH_sync<T>(force.xyz(), xyz2.data(), force.size());

  max_diff = 0.0;

  for (int i = 0; i < this->size(); i++) {
    const double fx1 = static_cast<double>(xyz1[0 * this->stride() + i]);
    const double fy1 = static_cast<double>(xyz1[1 * this->stride() + i]);
    const double fz1 = static_cast<double>(xyz1[2 * this->stride() + i]);
    const double fx2 = static_cast<double>(xyz2[0 * force.stride() + i]);
    const double fy2 = static_cast<double>(xyz2[1 * force.stride() + i]);
    const double fz2 = static_cast<double>(xyz2[2 * force.stride() + i]);
    if (std::isnan(fx1) || std::isnan(fy1) || std::isnan(fz1) ||
        std::isnan(fx2) || std::isnan(fy2) || std::isnan(fz2)) {
      std::cout << "i = " << i << std::endl;
      std::cout << "this: fx1 fy1 fz1 = " << fx1 << " " << fy1 << " " << fz1
                << std::endl;
      std::cout << "force:fx2 fy2 fz2 = " << fx2 << " " << fy2 << " " << fz2
                << std::endl;
      return false;
    }
    const double diff =
        std::max(std::abs(fx1 - fx2),
                 std::max(std::abs(fy1 - fy2), std::abs(fz1 - fz2)));
    max_diff = std::max(diff, max_diff);
    if (diff > tol) {
      std::cout << "i = " << i << std::endl;
      std::cout << "this: fx1 fy1 fz1 = " << fx1 << " " << fy1 << " " << fz1
                << std::endl;
      std::cout << "force:fx2 fy2 fz2 = " << fx2 << " " << fy2 << " " << fz2
                << std::endl;
      std::cout << "difference: " << diff << std::endl;
      return false;
    }
  }

  return true;
}

template <typename T> int Force<T>::stride(void) { return m_Stride; }

template <typename T> int Force<T>::stride(void) const { return m_Stride; }

template <typename T> int Force<T>::size(void) { return m_Size; }

template <typename T> int Force<T>::size(void) const { return m_Size; }

template <typename T> T *Force<T>::xyz(void) { return m_XYZ; }

template <typename T> const T *Force<T>::xyz(void) const { return m_XYZ; }

template <typename T> T *Force<T>::x(void) { return m_XYZ + 0 * m_Stride; }

template <typename T> const T *Force<T>::x(void) const {
  return m_XYZ + 0 * m_Stride;
}

template <typename T> T *Force<T>::y(void) { return m_XYZ + 1 * m_Stride; }

template <typename T> const T *Force<T>::y(void) const {
  return m_XYZ + 1 * m_Stride;
}

template <typename T> T *Force<T>::z(void) { return m_XYZ + 2 * m_Stride; }

template <typename T> const T *Force<T>::z(void) const {
  return m_XYZ + 2 * m_Stride;
}

//
// Gets forces to host
//
template <typename T> void Force<T>::getXYZ(T *h_x, T *h_y, T *h_z) {
  std::vector<T> h_xyz(3 * m_Stride);
  copy_DtoH_sync<T>(m_XYZ, h_xyz.data(), 3 * m_Stride);
  for (int i = 0; i < m_Size; i++) {
    h_x[i] = h_xyz[0 * m_Stride + i];
    h_y[i] = h_xyz[1 * m_Stride + i];
    h_z[i] = h_xyz[2 * m_Stride + i];
  }
  return;
}

//
// Converts one type of force array to another. Result is in "force"
//
template <typename T>
template <typename T2>
void Force<T>::convert(Force<T2> &force, cudaStream_t stream) {
  assert(force.size() == this->size());

  if (force.stride() == this->stride()) {
    constexpr int nthread = 512;
    const int nblock = (3 * this->stride() - 1) / nthread + 1;
    reduce_force<T, T2><<<nblock, nthread, 0, stream>>>(
        3 * this->stride(), this->xyz(), force.xyz());
    cudaCheck(cudaGetLastError());
  } else {
    constexpr int nthread = 512;
    const int nblock = (this->size() - 1) / nthread + 1;
    reduce_force<T, T2><<<nblock, nthread, 0, stream>>>(
        this->size(), this->stride(), this->xyz(), force.stride(), force.xyz());
    cudaCheck(cudaGetLastError());
  }

  return;
}

//
// Converts one type of force array to another. Result is in "this"
// NOTE: Only works when the size of the types T and T2 match
//
template <typename T>
template <typename T2>
void Force<T>::convert(cudaStream_t stream) {
  assert(sizeof(T) == sizeof(T2));

  constexpr int nthread = 512;
  const int nblock = (3 * this->stride() - 1) / nthread + 1;

  reduce_force<T, T2>
      <<<nblock, nthread, 0, stream>>>(3 * this->stride(), this->xyz());
  cudaCheck(cudaGetLastError());

  return;
}

//
// Converts one type of force array to another. Result is in "force"
//
template <typename T>
template <typename T2, typename T3>
void Force<T>::convert_to(Force<T3> &force, cudaStream_t stream) {
  assert(force.size() == this->size());
  assert(force.stride() == this->stride());
  assert(sizeof(T2) == sizeof(T3));

  constexpr int nthread = 512;
  const int nblock = (3 * this->stride() - 1) / nthread + 1;

  reduce_force<T, T2><<<nblock, nthread, 0, stream>>>(
      3 * this->stride(), this->xyz(), (T2 *)force.xyz());
  cudaCheck(cudaGetLastError());

  return;
}

//
// Converts one type of force array to another and adds force to the result.
// Result is in "this"
// NOTE: Only works when the size of the types T and T2 match
//
template <typename T>
template <typename T2, typename T3>
void Force<T>::convert_add(Force<T3> &force, cudaStream_t stream) {
  assert(force.stride() == this->stride());
  assert(sizeof(T) == sizeof(T2));

  constexpr int nthread = 512;
  const int nblock = (3 * this->stride() - 1) / nthread + 1;

  reduce_add_force<T, T2, T3><<<nblock, nthread, 0, stream>>>(
      3 * this->stride(), force.xyz(), this->xyz());
  cudaCheck(cudaGetLastError());

  return;
}

//
// Adds force to the converted result
// Result is in "this"
// NOTE: Only works when the size of the types T and T2 match
//
template <typename T>
template <typename T2, typename T3>
void Force<T>::add(Force<T3> &force, cudaStream_t stream) {
  assert(force.stride() == this->stride());
  assert(sizeof(T) == sizeof(T2));

  constexpr int nthread = 512;
  const int nblock = (3 * this->stride() - 1) / nthread + 1;

  add_force<T2, T3><<<nblock, nthread, 0, stream>>>(
      3 * this->stride(), force.xyz(), (T2 *)this->xyz());
  cudaCheck(cudaGetLastError());

  return;
}

//
// Adds non-strided force_data
//
template <typename T>
template <typename T2>
void Force<T>::add(float3 *force_data, int force_n, cudaStream_t stream) {
  assert(force_n <= this->size());
  assert(sizeof(T) == sizeof(T2));

  constexpr int nthread = 512;
  const int nblock = (force_n - 1) / nthread + 1;

  add_nonstrided_force<<<nblock, nthread, 0, stream>>>(
      force_n, force_data, this->stride(), (double *)this->xyz());
  cudaCheck(cudaGetLastError());

  return;
}

template <typename T>
__global__ static void addForceSameTypeKernel(T *__restrict__ this_xyz,
                                              const T *__restrict__ other_xyz,
                                              const int size,
                                              const int stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    this_xyz[0 * stride + idx] += other_xyz[0 * stride + idx];
    this_xyz[1 * stride + idx] += other_xyz[1 * stride + idx];
    this_xyz[2 * stride + idx] += other_xyz[2 * stride + idx];
  }
  return;
}

// Add another force of same type as this
template <typename T>
void Force<T>::add(const Force<T> &force, cudaStream_t stream) {
  constexpr int numThreads = 512;
  const int numBlocks = (this->size() - 1) / numThreads + 1;

  addForceSameTypeKernel<<<numBlocks, numThreads, 0, stream>>>(
      this->xyz(), force.xyz(), this->size(), this->stride());
  cudaDeviceSynchronize();

  return;
}

//
// Save to file
//
template <typename T>
template <typename T2>
void Force<T>::save(const char *filename) {
  assert(sizeof(T) == sizeof(T2));

  std::ofstream file(filename);
  if (!file.is_open())
    throw std::invalid_argument("Error opening file " + std::string(filename));

  std::vector<T2> h_xyz(3 * m_Stride);
  copy_DtoH_sync<T2>((T2 *)m_XYZ, h_xyz.data(), 3 * m_Stride);
  for (int i = 0; i < m_Size; i++) {
    file << h_xyz[0 * m_Stride + i] << ' ' << h_xyz[1 * m_Stride + i] << ' '
         << h_xyz[2 * m_Stride + i] << '\n';
  }

  return;
}

template <typename T> void Force<T>::realloc(int size, float fac) {
  m_Size = size;
  // Returns stride that aligns with 256 byte boundaries
  m_Stride = (((size - 1 + 32) * sizeof(T) - 1) / 256 + 1) * 256 / sizeof(T);
  const int new_capacity = static_cast<int>(static_cast<double>(3 * m_Stride) *
                                            static_cast<double>(fac));
  reallocate<T>(&m_XYZ, &m_Capacity, new_capacity, fac);
  return;
}

//
// Explicit instances of Force class
//
template class Force<long long int>;
template class Force<double>;
template class Force<float>;
template void Force<long long int>::convert<double>(cudaStream_t stream);
template void Force<long long int>::convert_add<double>(Force<float> &force,
                                                        cudaStream_t stream);
template void Force<long long int>::add<double>(Force<long long int> &force,
                                                cudaStream_t stream);
template void Force<long long int>::convert<float>(Force<float> &force,
                                                   cudaStream_t stream);
template void Force<long long int>::convert<double>(Force<double> &force,
                                                    cudaStream_t stream);
template void Force<float>::convert_to<double>(Force<long long int> &force,
                                               cudaStream_t stream);
template void Force<long long int>::add<double>(float3 *force_data, int force_n,
                                                cudaStream_t stream);
template void Force<long long int>::save<double>(const char *filename);
template void Force<long long int>::save<float>(const char *filename);
template void Force<double>::save<double>(const char *filename);

template void Force<double>::add<double>(Force<long long> &force,
                                         cudaStream_t stream);
// template void Force<float>::save<float>(const char* filename);

// template void Force<long long int>::saveFloat(const char* fileName);

#endif // NOCUDAC
