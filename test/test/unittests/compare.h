#pragma once

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

template <typename T>
bool CompareVectors(const std::vector<T> &a, const std::vector<T> &b) {
  if (a.size() != b.size())
    return false;

  for (std::size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i])
      return false;
  }

  return true;
}

template <>
bool CompareVectors(const std::vector<double4> &a,
                    const std::vector<double4> &b) {
  if (a.size() != b.size())
    return false;

  for (std::size_t i = 0; i < a.size(); i++) {
    if (a[i].x != b[i].x)
      return false;
    if (a[i].y != b[i].y)
      return false;
    if (a[i].z != b[i].z)
      return false;
    if (a[i].w != b[i].w)
      return false;
  }

  return true;
}

template <>
bool CompareVectors(const std::vector<int4> &a, const std::vector<int4> &b) {
  if (a.size() != b.size())
    return false;

  for (std::size_t i = 0; i < a.size(); i++) {
    if (a[i].x != b[i].x)
      return false;
    if (a[i].y != b[i].y)
      return false;
    if (a[i].z != b[i].z)
      return false;
    if (a[i].w != b[i].w)
      return false;
  }

  return true;
}

template <>
bool CompareVectors(const std::vector<int3> &a, const std::vector<int3> &b) {
  if (a.size() != b.size())
    return false;

  for (std::size_t i = 0; i < a.size(); i++) {
    if (a[i].x != b[i].x)
      return false;
    if (a[i].y != b[i].y)
      return false;
    if (a[i].z != b[i].z)
      return false;
  }
  return true;
}

inline bool CompareVectors(const std::vector<std::vector<double>> &a,
                           const std::vector<double4> &b,
                           const double tol = 1e-12) {
  if (a.size() != b.size()) {
    std::cout << "a.size() != b.size()" << std::endl;
    return false;
  }
  for (std::size_t i = 0; i < a.size(); i++) {
    assert((a[i].size() == 3) && "vect A secondary dimension is not 3");
    bool flag = true;
    if (std::abs(a[i][0] - b[i].x) > tol)
      flag = false;
    if (std::abs(a[i][1] - b[i].y) > tol)
      flag = false;
    if (std::abs(a[i][2] - b[i].z) > tol)
      flag = false;

    if (!flag) {
      std::cout << "a[" << i << "]: " << a[i][0] << " " << a[i][1] << " "
                << a[i][2] << std::endl;
      std::cout << "b[" << i << "]: " << b[i].x << " " << b[i].y << " "
                << b[i].z << std::endl;
      return flag;
    }
  }
  return true;
}

// give me a std::vect<vect<T>> of dimensions N * 3,  and I'll give back
// vect<T3>
template <typename T, typename T3>
std::vector<T3> flattenVector(const std::vector<std::vector<T>> &inputVector) {
  std::vector<T3> returnVector;
  for (std::size_t i = 0; i < inputVector.size(); i++) {
    assert(inputVector[i].size() == 3);
    T3 tempval = {inputVector[i][0], inputVector[i][1], inputVector[i][2]};
    returnVector.push_back(tempval);
  }
  return returnVector;
}

// Same as CompareVectors (check if one by one identical), except that it takes
// into account potential image recentering
inline bool CompareVectorsPBC(const std::vector<std::vector<double>> &a,
                              const std::vector<double4> &b,
                              const std::vector<double> &boxDim,
                              const double tol = 1e-12) {
  if (a.size() != b.size()) {
    std::cout << "a.size() != b.size()" << std::endl;
    return false;
  }
  for (std::size_t i = 0; i < a.size(); i++) {
    assert((a[i].size() == 3) && "vect A secondary dimension is not 3");
    bool flag = true;

    double3 diff = {std::abs(a[i][0] - b[i].x), std::abs(a[i][1] - b[i].y),
                    std::abs(a[i][2] - b[i].z)};

    if (diff.x > tol) {
      if (std::abs(diff.x - boxDim[0]) > tol)
        flag = false;
    }
    if (diff.y > tol) {
      if (std::abs(diff.y - boxDim[1]) > tol)
        flag = false;
    }
    if (diff.z > tol) {
      if (std::abs(diff.z - boxDim[2]) > tol)
        flag = false;
    }

    if (!flag) {
      std::cout << "a[" << i << "]: " << a[i][0] << " " << a[i][1] << " "
                << a[i][2] << std::endl;
      std::cout << "b[" << i << "]: " << b[i].x << " " << b[i].y << " "
                << b[i].z << std::endl;
      return flag;
    }
  }
  return true;
}

// jeg231113: I think these will be "safer" comparison functions for the
// future, so we should switch to these eventually
template <typename Type>
bool CompareVectors1(const std::vector<Type> &vecA,
                     const std::vector<Type> &vecB, const double tol = 1e-12,
                     const bool verbose = false) {
  if (vecA.size() != vecB.size()) {
    if (verbose) {
      std::cout << "Vectors are not the same size:\n";
      std::cout << " Vector A size = " << vecA.size() << "\n";
      std::cout << " Vector B size = " << vecB.size() << std::endl;
    }
    return false;
  }

  const std::size_t n = vecA.size();

  for (std::size_t i = 0; i < n; i++) {
    Type diff = std::abs(vecA[i] - vecB[i]);
    if (diff > tol) {
      if (verbose) {
        int pwr = static_cast<int>(std::ceil(std::abs(std::log10(tol))));
        std::ios fmt(nullptr);
        fmt.copyfmt(std::cout);
        std::cout << "Vectors differ at index " << i << ":\n";
        std::cout << std::scientific << std::setprecision(pwr);
        std::cout << " Vector A value = " << vecA[i] << "\n";
        std::cout << " Vector B value = " << vecB[i] << "\n";
        std::cout << "     Abs. Diff. = " << diff << std::endl;
        std::cout.copyfmt(fmt);
      }
      return false;
    }
  }

  return true;
}

template <>
bool CompareVectors1(const std::vector<double3> &vecA,
                     const std::vector<double3> &vecB, const double tol,
                     const bool verbose) {
  if (vecA.size() != vecB.size()) {
    if (verbose) {
      std::cout << "Vectors are not the same size:\n";
      std::cout << " Vector A size = " << vecA.size() << "\n";
      std::cout << " Vector B size = " << vecB.size() << std::endl;
    }
    return false;
  }

  const std::size_t n = vecA.size();

  for (std::size_t i = 0; i < n; i++) {
    double3 diff = {std::abs(vecA[i].x - vecB[i].x),
                    std::abs(vecA[i].y - vecB[i].y),
                    std::abs(vecA[i].z - vecB[i].z)};

    if ((diff.x > tol) || (diff.y > tol) || (diff.z > tol)) {
      if (verbose) {
        int pwr = static_cast<int>(std::ceil(std::abs(std::log10(tol))));
        std::ios fmt(nullptr);
        fmt.copyfmt(std::cout);
        std::cout << "Vectors differ at index " << i << ":\n";
        std::cout << std::scientific << std::setprecision(pwr);
        std::cout << " Vector A value = {" << vecA[i].x << ", " << vecA[i].y
                  << ", " << vecA[i].z << "}\n";
        std::cout << " Vector B value = {" << vecB[i].x << ", " << vecB[i].y
                  << ", " << vecB[i].z << "}\n";
        std::cout << "     Abs. Diff. = {" << diff.x << ", " << diff.y << ", "
                  << diff.z << "}" << std::endl;
        std::cout.copyfmt(fmt);
      }
      return false;
    }
  }

  return true;
}

template <>
bool CompareVectors1(const std::vector<double4> &vecA,
                     const std::vector<double4> &vecB, const double tol,
                     const bool verbose) {
  if (vecA.size() != vecB.size()) {
    if (verbose) {
      std::cout << "Vectors are not the same size:\n";
      std::cout << " Vector A size = " << vecA.size() << "\n";
      std::cout << " Vector B size = " << vecB.size() << std::endl;
    }
    return false;
  }

  const std::size_t n = vecA.size();

  for (std::size_t i = 0; i < n; i++) {
    double4 diff = {
        std::abs(vecA[i].x - vecB[i].x), std::abs(vecA[i].y - vecB[i].y),
        std::abs(vecA[i].z - vecB[i].z), std::abs(vecA[i].w - vecB[i].w)};

    if ((diff.x > tol) || (diff.y > tol) || (diff.z > tol) || (diff.w > tol)) {
      if (verbose) {
        int pwr = static_cast<int>(std::ceil(std::abs(std::log10(tol))));
        std::ios fmt(nullptr);
        fmt.copyfmt(std::cout);
        std::cout << "Vectors differ at index " << i << ":\n";
        std::cout << std::scientific << std::setprecision(pwr);
        std::cout << " Vector A value = {" << vecA[i].x << ", " << vecA[i].y
                  << ", " << vecA[i].z << ", " << vecA[i].w << "}\n";
        std::cout << " Vector B value = {" << vecB[i].x << ", " << vecB[i].y
                  << ", " << vecB[i].z << ", " << vecB[i].w << "}\n";
        std::cout << "     Abs. Diff. = {" << diff.x << ", " << diff.y << ", "
                  << diff.z << ", " << diff.w << "}" << std::endl;
        std::cout.copyfmt(fmt);
      }
      return false;
    }
  }

  return true;
}

template <typename Type>
bool CompareVectorsPBC1(const std::vector<Type> &vecA,
                        const std::vector<Type> &vecB, const Type boxdim,
                        const double tol = 1e-12, const bool verbose = false);
//                         {
//   if (vecA.size() != vecB.size()) {
//     if (verbose) {
//       std::cout << "Vectors are not the same size:\n";
//       std::cout << " Vector A size = " << vecA.size() << "\n";
//       std::cout << " Vector B size = " << vecB.size() << std::endl;
//     }
//     return false;
//   }

//   const std::size_t n = vecA.size();

//   for (std::size_t i=0;i<n;i++) {

//   }

//   return true;
// }

template <>
bool CompareVectorsPBC1(const std::vector<double4> &vecA,
                        const std::vector<double4> &vecB, const double4 boxDim,
                        const double tol, const bool verbose) {
  if (vecA.size() != vecB.size()) {
    if (verbose) {
      std::cout << "Vectors are not the same size:\n";
      std::cout << " Vector A size = " << vecA.size() << "\n";
      std::cout << " Vector B size = " << vecB.size() << std::endl;
    }
    return false;
  }

  const std::size_t n = vecA.size();

  for (std::size_t i = 0; i < n; i++) {
    bool flag = true;
    double4 diff = {
        std::abs(vecA[i].x - vecB[i].x), std::abs(vecA[i].y - vecB[i].y),
        std::abs(vecA[i].z - vecB[i].z), std::abs(vecA[i].w - vecB[i].w)};

    if ((diff.x > tol) && (std::abs(diff.x - boxDim.x) > tol))
      flag = false;
    if ((diff.y > tol) && (std::abs(diff.y - boxDim.y) > tol))
      flag = false;
    if ((diff.z > tol) && (std::abs(diff.z - boxDim.z) > tol))
      flag = false;
    if ((diff.w > tol) && (std::abs(diff.w - boxDim.w) > tol))
      flag = false;

    if (!flag) {
      if (verbose) {
        int pwr = static_cast<int>(std::ceil(std::abs(std::log10(tol))));
        std::ios fmt(nullptr);
        fmt.copyfmt(std::cout);
        std::cout << "Vectors differ at index " << i << ":\n";
        std::cout << std::scientific << std::setprecision(pwr);
        std::cout << " Vector A value = {" << vecA[i].x << ", " << vecA[i].y
                  << ", " << vecA[i].z << ", " << vecA[i].w << "}\n";
        std::cout << " Vector B value = {" << vecB[i].x << ", " << vecB[i].y
                  << ", " << vecB[i].z << ", " << vecB[i].w << "}\n";
        std::cout << "     Abs. Diff. = {" << diff.x << ", " << diff.y << ", "
                  << diff.z << ", " << diff.w << "}" << std::endl;
        std::cout << " PBC Abs. Diff. = {" << std::abs(diff.x - boxDim.x)
                  << ", " << std::abs(diff.y - boxDim.y) << ", "
                  << std::abs(diff.z - boxDim.z) << ", "
                  << std::abs(diff.w - boxDim.w) << "}" << std::endl;
        std::cout.copyfmt(fmt);
      }
      return false;
    }
  }

  return true;
}
