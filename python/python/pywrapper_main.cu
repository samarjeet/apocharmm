#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "cuda_utils.h"

namespace py = pybind11;

void exportCharmmContext(py::module &module);
void exportCoordinates(py::module &module);
void exportCharmmCrd(py::module &module);
void exportCharmmParameters(py::module &module);
void exportCharmmPSF(py::module &module);
void exportIntegrator(py::module &module);
void exportForceManager(py::module &module);
void exportPDB(py::module &module);
void exportSubscriber(py::module &module);
void exportForceManagerGenerator(py::module &module);
void exportMinimizer(py::module &module);
void exportPBC(py::module &module);

PYBIND11_MODULE(apocharmm, module) {
  module.doc() = R"pbdoc(
     python interface for apocharmm
     ------------------------------
     
     This module contains all the cuda/C++ objects made available to the python
     API by pybind11.

  )pbdoc";

  std::vector<int> devices = {0, 1, 2, 3};
  start_gpu(1, 1, 0, devices);

  exportCharmmContext(module);
  exportCoordinates(module);
  exportCharmmCrd(module);
  exportCharmmPSF(module);
  exportCharmmParameters(module);
  exportForceManager(module);
  exportIntegrator(module);
  exportPDB(module);
  exportSubscriber(module);
  exportForceManagerGenerator(module);
  exportMinimizer(module);
  exportPBC(module);
}
