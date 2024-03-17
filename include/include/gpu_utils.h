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

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <stdio.h>

static const int warpsize = 32;
// static __constant__ const int warpsize = 32;

static __constant__ const float FORCE_SCALE = (float)(1ll << 40);
static __constant__ const double INV_FORCE_SCALE =
    (double)1.0 / (double)(1ll << 40);

static __constant__ const float FORCE_SCALE_I = (float)(1 << 31);
static __constant__ const double INV_FORCE_SCALE_I =
    (double)1.0 / (double)(1 << 31);

static __constant__ const float FORCE_SCALE_VIR = (float)(1ll << 30);
static __constant__ const double INV_FORCE_SCALE_VIR =
    (double)1.0 / (double)(1ll << 30);
static __constant__ const long long int CONVERT_TO_VIR = (1ll << 10);

// CPU code version
static const double INV_FORCE_SCALE_CPU = (double)1.0 / (double)(1ll << 40);
static const double INV_FORCE_SCALE_VIR_CPU = (double)1.0 / (double)(1ll << 30);

#define cufftCheck(stmt)                                                       \
  do {                                                                         \
    cufftResult err = stmt;                                                    \
    if (err != CUFFT_SUCCESS) {                                                \
      printf("Error running %s in file %s, function %s\n", #stmt, __FILE__,    \
             __FUNCTION__);                                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 9
#define ALL(...) __all_sync(0xFFFFFFFF, __VA_ARGS__)
#define ANY(...) __any_sync(0xFFFFFFFF, __VA_ARGS__)
#define BALLOT(...) __ballot_sync(0xFFFFFFFF, __VA_ARGS__)
#define SHFL(...) __shfl_sync(0xFFFFFFFF, __VA_ARGS__)
#define SHFL_UP(...) __shfl_up_sync(0xFFFFFFFF, __VA_ARGS__)
#define SHFL_XOR(...) __shfl_xor_sync(0xFFFFFFFF, __VA_ARGS__)
#else
#define ALL(...) __all(__VA_ARGS__)
#define ANY(...) __any(__VA_ARGS__)
#define BALLOT(...) __ballot(__VA_ARGS__)
#define SHFL(...) __shfl(__VA_ARGS__)
#define SHFL_UP(...) __shfl_up(__VA_ARGS__)
#define SHFL_XOR(...) __shfl_xor(__VA_ARGS__)
#endif

// Double precision atomicAdd from CUDA_C_Programming_Guide.pdf (ver 5.0)
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600) ||                         \
    (__CUDACC_VER_MAJOR__ < 8)
static __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

/*
//
// Float atomicMin
//
static __device__ float atomicMin(float* addr, float value) {
  float old = *addr, assumed;

  if (old <= value) return old;
  do {
    assumed = old;
    old = atomicCAS((unsigned int*)addr, __float_as_int(assumed),
                    __float_as_int((old <= value) ? old : value));
  } while (old!=assumed);

  return old;
}

//
// Float atomicMax
//
static __device__ float atomicMaxold(float* addr, float value) {
  float old = *addr, assumed;

  if (old >= value) return old;
  do {
    assumed = old;
    old = atomicCAS((unsigned int*)addr, __float_as_int(assumed),
                    __float_as_int((assumed >= value) ? assumed : value));
  } while (old!=assumed);

  return old;
}
*/

static __device__ float atomicMin(float *address, float val) {
  int *address_as_i = (int *)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

static __device__ float atomicMax(float *address, float val) {
  int *address_as_i = (int *)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

//----------------------------------------------------------------------------------------

//
// Calculates exclusive plus-scan across warp for binary (0 or 1) values
//
// wid = warp ID = threadIdx.x % warpsize
//
__forceinline__ __device__ int binary_excl_scan(int val, int wid) {
  return __popc(BALLOT(val) & ((1 << wid) - 1));
}

//
// Calculates reduction across warp for binary (0 or 1) values. Result is in all
// threads within the warp
//
__forceinline__ __device__ int binary_reduce(int val) {
  return __popc(BALLOT(val));
}

__forceinline__ __device__ int min_shfl(int val) {
  for (int i = 16; i >= 1; i /= 2)
    val = min(val, SHFL_XOR(val, i));
  return val;
}

__forceinline__ __device__ int max_shfl(int val) {
  for (int i = 16; i >= 1; i /= 2)
    val = max(val, SHFL_XOR(val, i));
  return val;
}

//
// Broadcasts value from a single lane to all lanes
//
__forceinline__ __device__ int bcast_shfl(int val, const int srclane) {
  return SHFL(val, srclane);
}
//
// Calculates inclusive plus scan across warp
//
__forceinline__ __device__ int incl_scan_shfl(int val, const int wid,
                                              const int scansize = warpsize) {
  for (int i = 1; i < scansize; i *= 2) {
    int n = SHFL_UP(val, i, scansize);
    if (wid >= i)
      val += n;
  }
  return val;
}

//----------------------------------------------------------------------------------------

static __forceinline__ __device__ float __internal_fmad(float a, float b,
                                                        float c) {
#if __CUDA_ARCH__ >= 200
  return __fmaf_rn(a, b, c);
#else  // __CUDA_ARCH__ >= 200
  return a * b + c;
#endif // __CUDA_ARCH__ >= 200
}

// Following inline functions are copied from PMEMD CUDA implementation.
// Credit goes to:
/*             Scott Le Grand (NVIDIA)             */
/*               Duncan Poole (NVIDIA)             */
/*                Ross Walker (SDSC)               */
//
// Faster ERFC approximation courtesy of Norbert Juffa. NVIDIA Corporation
static __forceinline__ __device__ float fasterfc(float a) {
  /* approximate log(erfc(a)) with rel. error < 7e-9 */
  float t, x = a;
  t = (float)-1.6488499458192755E-006;
  t = __internal_fmad(t, x, (float)2.9524665006554534E-005);
  t = __internal_fmad(t, x, (float)-2.3341951153749626E-004);
  t = __internal_fmad(t, x, (float)1.0424943374047289E-003);
  t = __internal_fmad(t, x, (float)-2.5501426008983853E-003);
  t = __internal_fmad(t, x, (float)3.1979939710877236E-004);
  t = __internal_fmad(t, x, (float)2.7605379075746249E-002);
  t = __internal_fmad(t, x, (float)-1.4827402067461906E-001);
  t = __internal_fmad(t, x, (float)-9.1844764013203406E-001);
  t = __internal_fmad(t, x, (float)-1.6279070384382459E+000);
  t = t * x;
  return exp2f(t);
}

__device__ inline unsigned long long int llitoulli(long long int l) {
  unsigned long long int u;
  asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
  return u;
}

__device__ inline long long int lliroundf(float f) {
  long long int l;
  asm("cvt.rni.s64.f32 	%0, %1;" : "=l"(l) : "f"(f));
  return l;
}

__device__ inline long long int lliroundd(double d) {
  long long int l;
  asm("cvt.rni.s64.f64 	%0, %1;" : "=l"(l) : "d"(d));
  return l;
}

// End of copied code.

__device__ inline unsigned int itoui(int l) {
  unsigned int u;
  asm("mov.b32    %0, %1;" : "=r"(u) : "r"(l));
  return u;
}

__device__ inline int iroundf(float f) {
  int l;
  asm("cvt.rni.s32.f32 	%0, %1;" : "=r"(l) : "f"(f));
  return l;
}

// ----------------------------------------------------------------------------------------------
template <typename AT, typename CT>
__forceinline__ __device__ AT roundCTtoAT(CT a) {
  return (AT)a;
}

template <>
__forceinline__ __device__ long long int
roundCTtoAT<long long int, float>(float a) {
  return lliroundf(a * FORCE_SCALE);
}

template <>
__forceinline__ __device__ long long int
roundCTtoAT<long long int, double>(double a) {
  return lliroundd(a * FORCE_SCALE);
}

// ----------------------------------------------------------------------------------------------
template <typename AT, typename CT>
__forceinline__ __device__ void
calc_component_force(CT fij, const CT dx, const CT dy, const CT dz, AT &fxij,
                     AT &fyij, AT &fzij) {
  fxij = (AT)(fij * dx);
  fyij = (AT)(fij * dy);
  fzij = (AT)(fij * dz);
}

template <>
__forceinline__ __device__ void calc_component_force<long long int, float>(
    float fij, const float dx, const float dy, const float dz,
    long long int &fxij, long long int &fyij, long long int &fzij) {
  fij *= FORCE_SCALE;
  fxij = lliroundf(fij * dx);
  fyij = lliroundf(fij * dy);
  fzij = lliroundf(fij * dz);
}

template <>
__forceinline__ __device__ void calc_component_force<long long int, double>(
    double fij, const double dx, const double dy, const double dz,
    long long int &fxij, long long int &fyij, long long int &fzij) {
  fij *= FORCE_SCALE;
  fxij = lliroundd(fij * dx);
  fyij = lliroundd(fij * dy);
  fzij = lliroundd(fij * dz);
}

// ----------------------------------------------------------------------------------------------
template <typename AT, typename CT>
__forceinline__ __device__ void
calc_component_force(CT fij1, const CT dx1, const CT dy1, const CT dz1, CT fij2,
                     const CT dx2, const CT dy2, const CT dz2, AT &fxij,
                     AT &fyij, AT &fzij) {
  fxij = (AT)(fij1 * dx1 + fij2 * dx2);
  fyij = (AT)(fij1 * dy1 + fij2 * dy2);
  fzij = (AT)(fij1 * dz1 + fij2 * dz2);
}

template <>
__forceinline__ __device__ void calc_component_force<long long int, float>(
    float fij1, const float dx1, const float dy1, const float dz1, float fij2,
    const float dx2, const float dy2, const float dz2, long long int &fxij,
    long long int &fyij, long long int &fzij) {
  fij1 *= FORCE_SCALE;
  fij2 *= FORCE_SCALE;
  fxij = lliroundf(fij1 * dx1 + fij2 * dx2);
  fyij = lliroundf(fij1 * dy1 + fij2 * dy2);
  fzij = lliroundf(fij1 * dz1 + fij2 * dz2);
}

template <>
__forceinline__ __device__ void calc_component_force<long long int, double>(
    double fij1, const double dx1, const double dy1, const double dz1,
    double fij2, const double dx2, const double dy2, const double dz2,
    long long int &fxij, long long int &fyij, long long int &fzij) {
  fij1 *= FORCE_SCALE;
  fij2 *= FORCE_SCALE;
  fxij = lliroundd(fij1 * dx1 + fij2 * dx2);
  fyij = lliroundd(fij1 * dy1 + fij2 * dy2);
  fzij = lliroundd(fij1 * dz1 + fij2 * dz2);
}

// ----------------------------------------------------------------------------------------------

template <typename AT>
__forceinline__ __device__ void write_force(const AT fx, const AT fy,
                                            const AT fz, const int ind,
                                            const int stride, AT *force) {
  // The generic version can not be used
}

// Template specialization for 64bit integer = "long long int"
template <>
__forceinline__ __device__ void
write_force<long long int>(const long long int fx, const long long int fy,
                           const long long int fz, const int ind,
                           const int stride, long long int *force) {
  atomicAdd((unsigned long long int *)&force[ind], llitoulli(fx));
  atomicAdd((unsigned long long int *)&force[ind + stride], llitoulli(fy));
  atomicAdd((unsigned long long int *)&force[ind + stride * 2], llitoulli(fz));
}
// ----------------------------------------------------------------------------------------------

//
// Calculates box shift
//
// ish = (imx+1) + 3*(imy+1) + 9*(imz+1) = 0...26
//
template <typename T>
__forceinline__ __device__ __host__ void
calc_box_shift(int ish, const T boxx, const T boxy, const T boxz, T &shx,
               T &shy, T &shz) {
  shz = (ish / 9 - 1) * boxz;
  ish -= (ish / 9) * 9;
  shy = (ish / 3 - 1) * boxy;
  ish -= (ish / 3) * 3;
  shx = (ish - 1) * boxx;
}

// ----------------------------------------------------------------------------------------------

__forceinline__ __device__ int
calc_ishift(const float4 xyzq_i, const float4 xyzq_j, const float3 half_box) {
  float3 dxyz;
  dxyz.x = xyzq_i.x - xyzq_j.x;
  dxyz.y = xyzq_i.y - xyzq_j.y;
  dxyz.z = xyzq_i.z - xyzq_j.z;

  int3 is;
  is.x = 0;
  is.y = 0;
  is.z = 0;

  // is = -1, 0, 1
  if (dxyz.x >= half_box.x) {
    is.x = -1;
  } else if (dxyz.x < -half_box.x) {
    is.x = 1;
  }

  if (dxyz.y >= half_box.y) {
    is.y = -1;
  } else if (dxyz.y < -half_box.y) {
    is.y = 1;
  }

  if (dxyz.z >= half_box.z) {
    is.z = -1;
  } else if (dxyz.z < -half_box.z) {
    is.z = 1;
  }

  return is.x + 1 + (is.y + 1) * 3 + (is.z + 1) * 9;
}

//----------------------------------------------------------------------------------------
//
// Reads value val from warp wid
//
#if __CUDA_ARCH__ >= 300
template <typename AT>
__forceinline__ __device__ void get_val(AT &val, const int wid) {
  // This generic version does nothing!
}

template <>
__forceinline__ __device__ void get_val(long long int &val, const int wid) {
  int lo = __double2loint(__longlong_as_double(val));
  int hi = __double2hiint(__longlong_as_double(val));

  lo = SHFL(lo, wid);
  hi = SHFL(hi, wid);

  val = __double_as_longlong(__hiloint2double(hi, lo));
}
#endif
//----------------------------------------------------------------------------------------

#endif // GPU_UTILS_H
#endif // NOCUDAC
