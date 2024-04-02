
#pragma once
#include "catch.hpp"
#include <cstdio>
#include <fstream>

template <class T>
bool compareAbsoluteFromFile(std::vector<T> computed, std::string fileName,
                             T tol) {
  std::ifstream ref(fileName);
  for (auto val : computed) {
    T refVal;
    ref >> refVal;
    REQUIRE(abs(refVal) == Approx(val).margin(tol));
  }

  return true;
}

template <class T>
bool compareFromFile(std::vector<T> computed, std::string fileName, T tol) {
  std::ifstream ref(fileName);
  for (auto val : computed) {
    T refVal;
    ref >> refVal;
    REQUIRE(refVal == Approx(val).margin(tol));
  }

  return true;
}

template <class T>
bool compareFromFile(std::vector<T> computed, std::string fileName) {
  std::ifstream ref(fileName);
  for (auto val : computed) {
    T refVal;
    ref >> refVal;
    REQUIRE(refVal == val);
  }

  return true;
}

bool compareCoordsFromFile(std::vector<float4> coords, std::string fileName) {
  std::ifstream ref(fileName);
  float tol = 0.0001;
  for (auto coord : coords) {
    float x, y, z;
    ref >> x >> y >> z;

    REQUIRE(coord.x == Approx(x).margin(tol));
    REQUIRE(coord.y == Approx(y).margin(tol));
    REQUIRE(coord.z == Approx(z).margin(tol));
  }
  return true;
}

bool compareCoordsFromFile(std::vector<double4> coords, std::string fileName) {
  std::ifstream ref(fileName);
  double tol = 0.0005;
  for (auto coord : coords) {
    double x, y, z;
    ref >> x >> y >> z;

    std::cout << coord.x << " " << coord.y << " " << coord.z << " | " << x
              << " " << y << " " << z << std::endl;
    REQUIRE(coord.x == Approx(x).margin(tol));
    REQUIRE(coord.y == Approx(y).margin(tol));
    REQUIRE(coord.z == Approx(z).margin(tol));
  }
  return true;
}

bool calculateKineticEnergyTest(std::vector<double4> velMass,
                                double kineticEnergy) {
  double tol = 0.005;
  int numAtoms = velMass.size();
  double kineticEnergyExpected = 0.0;

  for (int i = 0; i < numAtoms; ++i) {
    kineticEnergyExpected +=
        (velMass[i].x * velMass[i].x + velMass[i].y * velMass[i].y +
         velMass[i].z * velMass[i].z) /
        velMass[i].w;
  }
  kineticEnergyExpected /= 2.0;
  REQUIRE(kineticEnergyExpected == Approx(kineticEnergy).margin(tol));

  return true;
}

// Compare Vectors. Should work for any type
template <typename Type>
bool compareVectors(Type vecA, Type vecB, double tol, bool verbose = false) {
  bool allTests = true;
  if (vecA.size() != vecB.size()) {
    throw std::invalid_argument(
        "compareVectors: vectors of different dimensions !");
  }
  for (int i = 0; i < vecA.size(); i++) {
    if (vecA[i] != Approx(vecB[i]).margin(tol)) {
      allTests = false;
      if (verbose) {
        std::cout << "A[" << i << "]: " << vecA[i] << " != " << vecB[i]
                  << std::endl;
      }
    }
  }
  return allTests;
}

template <typename Type>
bool compareVectors(Type vecA, Type vecB, bool verbose = false) {
  bool allTests = true;
  if (vecA.size() != vecB.size()) {
    throw std::invalid_argument(
        "compareVectors: vectors of different dimensions !");
  }
  for (int i = 0; i < vecA.size(); i++) {
    if (vecA[i] != Approx(vecB[i])) {
      allTests = false;
      if (verbose) {
        std::cout << "A[" << i << "]: " << vecA[i] << " != " << vecB[i]
                  << std::endl;
      }
    }
  }
  return allTests;
}

template <typename Type>
bool compareCudaVectorsTriples(Type vecA, Type vecB, double tol = 0.00001,
                               bool verbose = false) {
  bool allTests = true;
  if (vecA.size() != vecB.size()) {
    throw std::invalid_argument(
        "compareVectors: vectors of different dimensions !");
  }
  for (int i = 0; i < vecA.size(); i++) {
    if (!compareTriples(vecA[i], vecB[i], tol)) {
      allTests = false;
      if (verbose) {
        std::cout << "A[" << i << "]: " << vecA[i].x << " " << vecA[i].y << " "
                  << vecA[i].z << " != " << vecB[i].x << " " << vecB[i].y << " "
                  << vecB[i].z << std::endl;
      }
    }
  }
  return allTests;
}

template <typename Type>
bool compareCudaVectorsTriples(Type vecA, Type vecB, bool verbose = false) {
  bool allTests = true;

  if (vecA.size() != vecB.size()) {
    throw std::invalid_argument(
        "compareVectors: vectors of different dimensions !");
  }
  for (int i = 0; i < vecA.size(); i++) {
    if (!compareTriples(vecA[i], vecB[i], 0.00001)) {
      allTests = false;
      if (verbose) {
        std::cout << "A[" << i << "]: " << vecA[i].x << " " << vecA[i].y << " "
                  << vecA[i].z << " != " << vecB[i].x << " " << vecB[i].y << " "
                  << vecB[i].z << std::endl;
      }
    }
  }
  return allTests;
}

// Compare float4s (or double4s) element by element
template <typename Type4>
bool compareTriples(Type4 inpA, Type4 inpB, double tol = 0.00001) {
  if (inpA.x != Approx(inpB.x).margin(tol)) {
    return false;
  }
  if (inpA.y != Approx(inpB.y).margin(tol)) {
    return false;
  }
  if (inpA.z != Approx(inpB.z).margin(tol)) {
    return false;
  }
  return true;
}

// Write some little desciption in there
bool compareP21Forces(int numAtoms, int stride,
                      const std::vector<double> forces,
                      std::vector<std::vector<double>> trueForces) {

  float tol = 0.005;

  for (int i = 0; i < numAtoms; ++i) {
    CHECK(forces[i] == Approx(trueForces[i][0]).margin(tol));
    CHECK(forces[stride + i] == Approx(trueForces[i][1]).margin(tol));
    CHECK(forces[2 * stride + i] == Approx(trueForces[i][2]).margin(tol));
  }
  return true;
}

// line counting function
int lineCounter(std::string fname) {
  std::ifstream f(fname);
  std::string line;
  int i = 0;
  // f.open(fname);
  while (std::getline(f, line)) {
    i++;
    // std::cout << "line " << i << " : " << line << std::endl;
  }
  f.close();
  return i;
}
