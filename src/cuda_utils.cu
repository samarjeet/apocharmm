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
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <algorithm>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <nvtx3/nvToolsExtCuda.h>
// #include <nvtx3/nvtx3.hpp>
#include <utility>

//----------------------------------------------------------------------------------------
//
// Deallocate page-locked host memory
// pp = memory pointer
//
void deallocate_host_T(void **pp) {
  if (*pp != NULL) {
    cudaCheck(cudaFreeHost((void *)(*pp)));
    *pp = NULL;
  }
}
//----------------------------------------------------------------------------------------
//
// Allocate page-locked host memory
// pp = memory pointer
// len = length of the array
//
void allocate_host_T(void **pp, const int len, const size_t sizeofT) {
  cudaCheck(cudaMallocHost(pp, sizeofT * len));
}

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate page-locked host memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will
// be fac*newlen
//
void reallocate_host_T(void **pp, int *curlen, const int newlen,
                       const float fac, const size_t sizeofT) {
  if (*pp != NULL && *curlen < newlen) {
    cudaCheck(cudaFreeHost((void *)(*pp)));
    *pp = NULL;
  }

  if (*pp == NULL) {
    if (fac > 1.0f) {
      *curlen = (int)(((double)(newlen)) * (double)fac);
    } else {
      *curlen = newlen;
    }
    allocate_host_T(pp, *curlen, sizeofT);
  }
}

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate page-locked host memory, preserves content
//
void resize_host_T(void **pp, int *curlen, const int cur_size,
                   const int new_size, const float fac, const size_t sizeofT) {
  char *old = NULL;

  if (*pp != NULL && *curlen < new_size) {
    old = new char[cur_size * sizeofT];
    memcpy(old, *pp, cur_size * sizeofT);
    cudaCheck(cudaFreeHost((void *)(*pp)));
    *pp = NULL;
  }

  if (*pp == NULL) {
    if (fac > 1.0f) {
      *curlen = (int)(((double)(new_size)) * (double)fac);
    } else {
      *curlen = new_size;
    }
    allocate_host_T(pp, *curlen, sizeofT);
    if (old != NULL) {
      memcpy(*pp, old, cur_size * sizeofT);
      delete[] old;
    }
  }
}

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
void deallocate_T(void **pp) {
  if (*pp != NULL) {
    cudaCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }
}
//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
void allocate_T(void **pp, const int len, const size_t sizeofT) {
  cudaCheck(cudaMalloc(pp, sizeofT * len));
}

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate gpu memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will
// be fac*newlen
//
void reallocate_T(void **pp, int *curlen, const int newlen, const float fac,
                  const size_t sizeofT) {
  if (*pp != NULL && *curlen < newlen) {
    cudaCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }

  if (*pp == NULL) {
    if (fac > 1.0f) {
      *curlen = (int)(((double)(newlen)) * (double)fac);
    } else {
      *curlen = newlen;
    }
    allocate_T(pp, *curlen, sizeofT);
  }
}

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate page-locked host memory, preserves content
//
void resize_T(void **pp, int *curlen, const int cur_size, const int new_size,
              const float fac, const size_t sizeofT) {
  void *old = NULL;

  if (*pp != NULL && *curlen < new_size) {
    allocate_T(&old, cur_size, sizeofT);
    copy_DtoD_T(*pp, old, cur_size, sizeofT);
    cudaCheck(cudaDeviceSynchronize()); // Make sure D-D copy is done
    cudaCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }

  if (*pp == NULL) {
    if (fac > 1.0f) {
      *curlen = (int)(((double)(new_size)) * (double)fac);
    } else {
      *curlen = new_size;
    }
    allocate_T(pp, *curlen, sizeofT);
    if (old != NULL) {
      copy_DtoD_T(old, *pp, cur_size, sizeofT);
      cudaCheck(cudaDeviceSynchronize()); // Make sure D-D copy is done
      deallocate_T(&old);
    }
  }
}

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
void copy_HtoD_async_T(const void *h_array, void *d_array, int array_len,
                       cudaStream_t stream, const size_t sizeofT) {
  cudaCheck(cudaMemcpyAsync(d_array, h_array, sizeofT * array_len,
                            cudaMemcpyHostToDevice, stream));
}

void copy_HtoD_T(const void *h_array, void *d_array, int array_len,
                 const size_t sizeofT) {
  cudaCheck(cudaMemcpy(d_array, h_array, sizeofT * array_len,
                       cudaMemcpyHostToDevice));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
void copy_DtoH_async_T(const void *d_array, void *h_array, const int array_len,
                       cudaStream_t stream, const size_t sizeofT) {
  cudaCheck(cudaMemcpyAsync(h_array, d_array, sizeofT * array_len,
                            cudaMemcpyDeviceToHost, stream));
}

void copy_DtoH_T(const void *d_array, void *h_array, const int array_len,
                 const size_t sizeofT) {
  cudaCheck(cudaMemcpy(h_array, d_array, sizeofT * array_len,
                       cudaMemcpyDeviceToHost));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Device
//
void copy_DtoD_async_T(const void *d_src, void *d_dst, const int array_len,
                       cudaStream_t stream, const size_t sizeofT) {
  cudaCheck(cudaMemcpyAsync(d_dst, d_src, sizeofT * array_len,
                            cudaMemcpyDeviceToDevice, stream));
}

void copy_DtoD_T(const void *d_src, void *d_dst, const int array_len,
                 const size_t sizeofT) {
  cudaCheck(
      cudaMemcpy(d_dst, d_src, sizeofT * array_len, cudaMemcpyDeviceToDevice));
}

//----------------------------------------------------------------------------------------

void clear_gpu_array_async_T(void *data, const int ndata, cudaStream_t stream,
                             const size_t sizeofT) {
  cudaCheck(cudaMemsetAsync(data, 0, sizeofT * ndata, stream));
}

void clear_gpu_array_T(void *data, const int ndata, const size_t sizeofT) {
  cudaCheck(cudaMemset(data, 0, sizeofT * ndata));
}

//----------------------------------------------------------------------------------------

void set_gpu_array_async_T(void *data, const int ndata, const int value,
                           cudaStream_t stream, const size_t sizeofT) {
  cudaCheck(cudaMemsetAsync(data, value, sizeofT * ndata, stream));
}

void set_gpu_array_T(void *data, const int ndata, const int value,
                     const size_t sizeofT) {
  cudaCheck(cudaMemset(data, value, sizeofT * ndata));
}

//----------------------------------------------------------------------------------------

void copy3D_HtoD_T(void *src_data, void *dst_data, int src_x0, int src_y0,
                   int src_z0, size_t src_xsize, size_t src_ysize, int dst_x0,
                   int dst_y0, int dst_z0, size_t width, size_t height,
                   size_t depth, size_t dst_xsize, size_t dst_ysize,
                   size_t sizeofT) {
  cudaMemcpy3DParms parms = {0};

  parms.srcPos = make_cudaPos(sizeofT * src_x0, src_y0, src_z0);
  parms.srcPtr =
      make_cudaPitchedPtr(src_data, sizeofT * src_xsize, src_xsize, src_ysize);

  parms.dstPos = make_cudaPos(sizeofT * dst_x0, dst_y0, dst_z0);
  parms.dstPtr =
      make_cudaPitchedPtr(dst_data, sizeofT * dst_xsize, dst_xsize, dst_ysize);

  parms.extent = make_cudaExtent(sizeofT * width, height, depth);
  parms.kind = cudaMemcpyHostToDevice;

  //  cudaCheck(cudaMemcpy3D(&parms));

  if (cudaMemcpy3D(&parms) != cudaSuccess) {
    std::cerr << "copy3D_HtoD_T" << std::endl;
    std::cerr << "source: " << std::endl;
    std::cerr << parms.srcPos.x << " " << parms.srcPos.y << " "
              << parms.srcPos.z << std::endl;
    std::cerr << parms.srcPtr.pitch << " " << parms.srcPtr.xsize << " "
              << parms.srcPtr.ysize << std::endl;
    std::cerr << "destination: " << std::endl;
    std::cerr << parms.dstPos.x << " " << parms.dstPos.y << " "
              << parms.dstPos.z << std::endl;
    std::cerr << parms.dstPtr.pitch << " " << parms.dstPtr.xsize << " "
              << parms.dstPtr.ysize << std::endl;
    std::cerr << "extent: " << std::endl;
    std::cerr << parms.extent.width << " " << parms.extent.height << " "
              << parms.extent.depth << std::endl;
    exit(1);
  }
}

//----------------------------------------------------------------------------------------

void copy3D_DtoH_T(void *src_data, void *dst_data, int src_x0, int src_y0,
                   int src_z0, size_t src_xsize, size_t src_ysize, int dst_x0,
                   int dst_y0, int dst_z0, size_t width, size_t height,
                   size_t depth, size_t dst_xsize, size_t dst_ysize,
                   size_t sizeofT) {
  cudaMemcpy3DParms parms = {0};

  parms.srcPos = make_cudaPos(sizeofT * src_x0, src_y0, src_z0);
  parms.srcPtr =
      make_cudaPitchedPtr(src_data, sizeofT * src_xsize, src_xsize, src_ysize);

  parms.dstPos = make_cudaPos(sizeofT * dst_x0, dst_y0, dst_z0);
  parms.dstPtr =
      make_cudaPitchedPtr(dst_data, sizeofT * dst_xsize, dst_xsize, dst_ysize);

  parms.extent = make_cudaExtent(sizeofT * width, height, depth);
  parms.kind = cudaMemcpyDeviceToHost;

  //  cudaCheck(cudaMemcpy3D(&parms));
  if (cudaMemcpy3D(&parms) != cudaSuccess) {
    std::cerr << "copy3D_DtoH_T" << std::endl;
    std::cerr << "source: " << std::endl;
    std::cerr << parms.srcPos.x << " " << parms.srcPos.y << " "
              << parms.srcPos.z << std::endl;
    std::cerr << parms.srcPtr.pitch << " " << parms.srcPtr.xsize << " "
              << parms.srcPtr.ysize << std::endl;
    std::cerr << "destination: " << std::endl;
    std::cerr << parms.dstPos.x << " " << parms.dstPos.y << " "
              << parms.dstPos.z << std::endl;
    std::cerr << parms.dstPtr.pitch << " " << parms.dstPtr.xsize << " "
              << parms.dstPtr.ysize << std::endl;
    std::cerr << "extent: " << std::endl;
    std::cerr << parms.extent.width << " " << parms.extent.height << " "
              << parms.extent.depth << std::endl;
    exit(1);
  }
}

//----------------------------------------------------------------------------------------

void gpu_range_start(const char *range_name) {
  static int color_id = 0;
  nvtxEventAttributes_t att = {0};
  att.version = NVTX_VERSION;
  att.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  att.colorType = NVTX_COLOR_ARGB;
  if (color_id == 0) {
    att.color = 0xFFFF0000;
  } else if (color_id == 1) {
    att.color = 0xFF00FF00;
  } else if (color_id == 2) {
    att.color = 0xFF0000FF;
  } else if (color_id == 3) {
    att.color = 0xFFFF00FF;
  }
  color_id++;
  if (color_id > 3)
    color_id = 0;
  att.messageType = NVTX_MESSAGE_TYPE_ASCII;
  att.message.ascii = range_name;
  nvtxRangePushEx(&att);
}

void gpu_range_stop() { nvtxRangePop(); }

//----------------------------------------------------------------------------------------

__global__ void read_CUDA_ARCH_kernel(int *cuda_arch) {
  if (threadIdx.x == 0) {
#if __CUDA_ARCH__ == 100
    *cuda_arch = 100;
#elif __CUDA_ARCH__ == 110
    *cuda_arch = 110;
#elif __CUDA_ARCH__ == 120
    *cuda_arch = 120;
#elif __CUDA_ARCH__ == 130
    *cuda_arch = 130;
#elif __CUDA_ARCH__ == 200
    *cuda_arch = 200;
#elif __CUDA_ARCH__ == 210
    *cuda_arch = 210;
#elif __CUDA_ARCH__ == 300
    *cuda_arch = 300;
#elif __CUDA_ARCH__ == 350
    *cuda_arch = 350;
#elif __CUDA_ARCH__ == 500
    *cuda_arch = 500;
#elif __CUDA_ARCH__ == 600
    *cuda_arch = 600;
#elif __CUDA_ARCH__ == 700
    *cuda_arch = 700;
// Later architectures have not yet arrived as of July 2019
#elif __CUDA_ARCH__ == 800
    *cuda_arch = 800;
#elif __CUDA_ARCH__ == 900
    *cuda_arch = 900;
#else
    *cuda_arch = 10000;
#endif
  }
}

//
// Reads the value of __CUDA_ARCH__ from device code
//
int read_CUDA_ARCH() {
  int *d_cuda_arch;
  int h_cuda_arch;
  allocate<int>(&d_cuda_arch, 1);

  read_CUDA_ARCH_kernel<<<1, 1>>>(d_cuda_arch);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "Error executing CUDA kernel read_CUDA_ARCH_kernel in file "
              << __FILE__ << std::endl;
    std::cout << "Error string: " << cudaGetErrorString(err) << std::endl;
    std::cout << "Possible cause: Device compute capability is less than the "
                 "compute capability the code was compiled for."
              << std::endl;
    exit(1);
  }
  cudaCheck(cudaDeviceSynchronize());

  copy_DtoH_sync<int>(d_cuda_arch, &h_cuda_arch, 1);

  deallocate<int>(&d_cuda_arch);

  return h_cuda_arch;
}
//----------------------------------------------------------------------------------------

static int gpu_ind = -1;
static cudaDeviceProp gpu_prop;
static int cuda_arch;
static int high_priority;
static int low_priority;

bool gpuDevCompare(std::pair<int, cudaDeviceProp> dev1,
                   std::pair<int, cudaDeviceProp> dev2) {
  int dev1cc = dev1.second.major * 100 + dev1.second.minor;
  int dev2cc = dev2.second.major * 100 + dev2.second.minor;
  if (dev1cc == dev2cc) {
    // Same compute capacity, choose by the number of SMs
    int dev1sm = dev1.second.multiProcessorCount;
    int dev2sm = dev2.second.multiProcessorCount;
    return (dev1sm >= dev2sm);
  } else {
    return (dev1cc >= dev2cc);
  }
}

void start_gpu(int prnlev, int numnode, int mynode, std::vector<int> &devices) {
  int deviceNum;
  // cudaCheck(cudaGetDeviceCount(&deviceNum));

  cudaError_t error_id = cudaGetDeviceCount(&deviceNum);
  if (error_id != cudaSuccess) {
    std::cout << cudaGetErrorString(error_id) << std::endl;
    exit(1);
  }

  if (deviceNum == 0) {
    std::cout << "No CUDA device found" << std::endl;
    exit(1);
  }

  if (devices.size() != numnode) {
    // if (mynode == 0) std::cout << "Selecting GPUs most powerful first" <<
    // std::endl;
    // Get all device properties and sort "most powerful first"
    std::vector<int> deviceID(deviceNum);
    std::vector<std::pair<int, cudaDeviceProp>> gpuList(deviceNum);
    for (int i = 0; i < deviceNum; i++) {
      gpuList.at(i).first = i;
      cudaCheck(cudaGetDeviceProperties(&gpuList.at(i).second, i));
    }
    std::sort(gpuList.begin(), gpuList.end(), gpuDevCompare);
    gpu_ind = gpuList.at(mynode % deviceNum).first;
  } else {
    // if (mynode == 0) std::cout << "Selecting GPUs using device list" <<
    // std::endl;
    // Use devices in order determined by "devices"
    gpu_ind = devices.at(mynode % deviceNum);
  }
  cudaCheck(cudaSetDevice(gpu_ind));

  cudaCheck(cudaDeviceSynchronize());

  cudaCheck(cudaGetDeviceProperties(&gpu_prop, gpu_ind));

  // std::cout << "CUDA device " << gpu_ind << " has compute mode "
  //          << gpu_prop.computeMode << "\n";
  if (has_stream_priorities()) {
    cudaCheck(cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority));
  }

  if (gpu_prop.major < 2) {
    std::cout << "CUDA device(s) must have compute capability 2.0 or higher"
              << std::endl;
    exit(1);
  }

  int cuda_driver_version;
  cudaCheck(cudaDriverGetVersion(&cuda_driver_version));

  int cuda_rt_version;
  cudaCheck(cudaRuntimeGetVersion(&cuda_rt_version));

  cuda_arch = read_CUDA_ARCH();

  if (cuda_arch < 200) {
    std::cout << "Code must be compiled with compute capability 2.0 or higher"
              << std::endl;
    exit(1);
  }

#ifdef USE_TEXTURE_OBJECTS
  if (cuda_arch < 300 || cuda_rt_version < 5000 || cuda_driver_version < 5000) {
    std::cout << "When compiled with USE_TEXTURE_OBJECTS, CUDA 5.0 and compute "
                 "capability 3.0 (Kepler) or greater required"
              << std::endl;
    exit(1);
  }
#endif

  if (mynode == 0) {
    // std::cout << "Number of CUDA devices found " << deviceNum <<
    // std::endl; std::cout << "Using CUDA driver version " <<
    // cuda_driver_version << std::endl; std::cout << "Using CUDA runtime
    // version " << cuda_rt_version << std::endl; std::cout << "Compiled
    // using CUDA_ARCH " << cuda_arch << std::endl;
  }

  if (prnlev >= 2) {
    std::cout << "Node " << mynode << " uses CUDA device " << gpu_ind << " "
              << gpu_prop.name << " with CUDA_ARCH " << cuda_arch << std::endl;
  }
}

void stop_gpu() {
  cudaCheck(cudaDeviceReset());
  gpu_ind = -1;
}

int3 get_max_nblock() {
  int3 max_nblock;
  max_nblock.x = gpu_prop.maxGridSize[0];
  max_nblock.y = gpu_prop.maxGridSize[1];
  max_nblock.z = gpu_prop.maxGridSize[2];
  if (cuda_arch <= 200) {
    max_nblock.x = min(65535, max_nblock.x);
    max_nblock.y = min(65535, max_nblock.y);
    max_nblock.z = min(65535, max_nblock.z);
  }
  return max_nblock;
}

bool has_stream_priorities() {
  return (bool)gpu_prop.streamPrioritiesSupported;
}

int low_stream_priority() { return low_priority; }

int high_stream_priority() { return high_priority; }

int get_max_nthread() { return gpu_prop.maxThreadsPerBlock; }

int get_max_shmem_size() { return gpu_prop.sharedMemPerBlock; }

int get_major() { return gpu_prop.major; }

int get_gpu_ind() { return gpu_ind; }

int get_cuda_arch() { return cuda_arch; }

//
// Save float3 device buffer on disk
//
void save_float3(const int n, const float3 *buf, const char *filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    float3 *h_buf = new float3[n];
    copy_DtoH_sync<float3>(buf, h_buf, n);
    for (int i = 0; i < n; i++) {
      file << h_buf[i].x << " " << h_buf[i].y << " " << h_buf[i].z << std::endl;
    }
    delete[] h_buf;
  } else {
    std::cerr << "Error opening file " << filename << std::endl;
    exit(1);
  }
}
#endif // NOCUDAC
