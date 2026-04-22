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
#ifndef CUDAXYZ_H
#define CUDAXYZ_H
#include "XYZ.h"
#include "cuda_utils.h"
#include <cassert>
#include <iostream>
#include <vector>

// Forward declaration of hostXYZ
template <typename T> class hostXYZ;

//
// CUDA XYZ array class
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//

template <typename T> class cudaXYZ : public XYZ<T> {
public:
  cudaXYZ() {}

  cudaXYZ(int size) { this->realloc(size); }

  template <typename P> cudaXYZ(hostXYZ<P> &xyz) {
    this->realloc(xyz.size());
    this->set_data(xyz);
  }

  ~cudaXYZ() {
    this->_size = 0;
    this->_capacity = 0;
    if (this->_x != NULL)
      deallocate<T>(&this->_x);
    if (this->_y != NULL)
      deallocate<T>(&this->_y);
    if (this->_z != NULL)
      deallocate<T>(&this->_z);
  }

  void realloc_array(T **array, int *capacity, int size, float fac) {
    reallocate<T>(array, capacity, size, fac);
  }

  void resize_array(T **array, int *capacity, int size, int new_size,
                    float fac) {
    resize<T>(array, capacity, size, new_size, fac);
  }

  // Clears the data array
  void clear(cudaStream_t stream = 0) {
    clear_gpu_array<T>(this->_x, this->_size, stream);
    clear_gpu_array<T>(this->_y, this->_size, stream);
    clear_gpu_array<T>(this->_z, this->_size, stream);
  }

  //--------------------------------------------------------------------------

  // Sets data from hostXYZ
  template <typename P>
  void set_data(hostXYZ<P> &xyz, cudaStream_t stream = 0) {
    assert(this->match(xyz));
    copy_HtoD<T>((T *)xyz.x(), this->_x, this->_size, stream);
    copy_HtoD<T>((T *)xyz.y(), this->_y, this->_size, stream);
    copy_HtoD<T>((T *)xyz.z(), this->_z, this->_size, stream);
  }

  // Sets data from hostXYZ synchroniously
  template <typename P> void set_data_sync(hostXYZ<P> &xyz) {
    assert(this->match(xyz));
    copy_HtoD_sync<T>((T *)xyz.x(), this->_x, this->_size);
    copy_HtoD_sync<T>((T *)xyz.y(), this->_y, this->_size);
    copy_HtoD_sync<T>((T *)xyz.z(), this->_z, this->_size);
  }

  // Sets data from cudaXYZ
  template <typename P>
  void set_data(cudaXYZ<P> &xyz, cudaStream_t stream = 0) {
    assert(this->match(xyz));
    copy_DtoD<T>((T *)xyz._x, this->_x, this->_size, stream);
    copy_DtoD<T>((T *)xyz._y, this->_y, this->_size, stream);
    copy_DtoD<T>((T *)xyz._z, this->_z, this->_size, stream);
  }

  // Sets n first entries of data from cudaXYZ
  template <typename P>
  void set_data(const int n, cudaXYZ<P> &xyz, cudaStream_t stream = 0) {
    assert(n <= this->_size);
    assert(n <= xyz._size);
    copy_DtoD<T>((T *)xyz._x, this->_x, n, stream);
    copy_DtoD<T>((T *)xyz._y, this->_y, n, stream);
    copy_DtoD<T>((T *)xyz._z, this->_z, n, stream);
  }

  // Sets data from cudaXYZ synchroniously
  template <typename P> void set_data_sync(cudaXYZ<P> &xyz) {
    assert(this->match(xyz));
    copy_DtoD_sync<T>((T *)xyz._x, this->_x, this->_size);
    copy_DtoD_sync<T>((T *)xyz._y, this->_y, this->_size);
    copy_DtoD_sync<T>((T *)xyz._z, this->_z, this->_size);
  }

  /*
  // Sets data from cudaXYZ pointer
  template <typename P>
  void set_data(cudaXYZ<P> *xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_DtoD<T>((T *)xyz->data, this->data, 3*this->stride, stream);
  }
  */

  /*
  // Sets data from cudaXYZ
  template <typename P>
  void set_data_sync(cudaXYZ<P> *xyz) {
    assert(this->match(xyz));
    copy_DtoD_sync<T>((T *)xyz->data, this->data, 3*this->stride);
  }
  */

  // Sets data from host arrays
  void set_data_sync(const int size, const T *h_x, const T *h_y, const T *h_z) {
    assert(this->_size == size);
    copy_HtoD_sync<T>(h_x, this->_x, this->_size);
    copy_HtoD_sync<T>(h_y, this->_y, this->_size);
    copy_HtoD_sync<T>(h_z, this->_z, this->_size);
  }

  // Sets data from host arrays with indexing
  void set_data_sync(const std::vector<int> &h_loc2glo, const T *h_x,
                     const T *h_y, const T *h_z) {
    assert(this->_size == h_loc2glo.size());

    T *h_xt = new T[this->_size];
    T *h_yt = new T[this->_size];
    T *h_zt = new T[this->_size];

    for (int i = 0; i < h_loc2glo.size(); i++) {
      int j = h_loc2glo[i];
      h_xt[i] = h_x[j];
      h_yt[i] = h_y[j];
      h_zt[i] = h_z[j];
    }

    this->set_data_sync(this->_size, h_xt, h_yt, h_zt);

    delete[] h_xt;
    delete[] h_yt;
    delete[] h_zt;
  }

  //--------------------------------------------------------------------------

  // Copies data to host buffers (x, y, z)
  void get_data_sync(const int size, double *h_x, double *h_y, double *h_z) {
    assert(size == this->_size);
    copy_DtoH_sync<T>(this->_x, h_x, this->_size);
    copy_DtoH_sync<T>(this->_y, h_y, this->_size);
    copy_DtoH_sync<T>(this->_z, h_z, this->_size);
  }

  //--------------------------------------------------------------------------

  void get_host_xyz(T *&hx, T *&hy, T *&hz) {
    hx = new T[this->_size];
    hy = new T[this->_size];
    hz = new T[this->_size];
    get_data_sync(this->_size, hx, hy, hz);
  }

  void release_host_xyz(T *&hx, T *&hy, T *&hz) {
    delete[] hx;
    delete[] hy;
    delete[] hz;
  }

  /*
  void print(const int start, const int end, std::ostream& out) {
    assert((start >= 0) && (end >= start) && (end < this->n));
    T *h_data = new T[this->stride*3];
    copy_DtoH<T>(this->data, h_data, this->stride*3);

    for (int i=start;i <= end;i++) {
      out << i << " " << h_data[i] << " " << h_data[i+this->stride] << " "
          << h_data[i+this->stride*2] << std::endl;
    }

    delete [] h_data;
  }
  */
};

#endif // CUDAXYZ_H
#endif // NOCUDAC
