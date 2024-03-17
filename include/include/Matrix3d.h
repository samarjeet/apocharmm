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
#ifndef MATRIX3D_H
#define MATRIX3D_H

template <typename T> class Matrix3d {
private:
  // True if we are using an external storage for the data
  bool external_storage;

  // Initializes (allocates) data
  void init(const int size, T *ext_data = NULL);

  double norm(T a, T b);
  bool is_nan(T a);

protected:
  // Size of the matrix
  int nx, ny, nz;

  // Size of the matrix in storage. Allows for padding.
  int xsize, ysize, zsize;

public:
  // Matrix data
  T *data;

  //  Matrix3d();
  Matrix3d(const int nx, const int ny, const int nz, T *ext_data = NULL);
  Matrix3d(const int nx, const int ny, const int nz, const int xsize,
           const int ysize, const int zsize, T *ext_data = NULL);
  Matrix3d(const int nx, const int ny, const int nz, const char *filename,
           T *ext_data = NULL);
  ~Matrix3d();

  void print_info();

  bool compare(Matrix3d<T> *mat, const double tol, double &max_diff);

  void transpose_xyz_yzx_host(int src_x0, int src_y0, int src_z0, int dst_x0,
                              int dst_y0, int dst_z0, int xlen, int ylen,
                              int zlen, Matrix3d<T> *mat);
  void transpose_xyz_yzx_host(Matrix3d<T> *mat);

  void transpose_xyz_zxy_host(Matrix3d<T> *mat);

  void transpose_xyz_yzx(Matrix3d<T> *mat);
  void transpose_xyz_yzx(int src_x0, int src_y0, int src_z0, int dst_x0,
                         int dst_y0, int dst_z0, int xlen, int ylen, int zlen,
                         Matrix3d<T> *mat);

  void transpose_xyz_zxy(Matrix3d<T> *mat);

  void copy_host(int src_x0, int src_y0, int src_z0, int dst_x0, int dst_y0,
                 int dst_z0, int xlen, int ylen, int zlen, Matrix3d<T> *mat);

  void copy(int src_x0, int src_y0, int src_z0, int dst_x0, int dst_y0,
            int dst_z0, int xlen, int ylen, int zlen, Matrix3d<T> *mat);
  void copy(Matrix3d<T> *mat);

  void print(const int x0, const int x1, const int y0, const int y1,
             const int z0, const int z1);

  void load(const int x0, const int x1, const int nx, const int y0,
            const int y1, const int ny, const int z0, const int z1,
            const int nz, const char *filename);

  void load(const int nx, const int ny, const int nz, const char *filename);

  void scale(const T fac);

  int get_nx();
  int get_ny();
  int get_nz();

  int get_xsize();
  int get_ysize();
  int get_zsize();
};

#endif // MATRIX3D_H
#endif // NOCUDAC
