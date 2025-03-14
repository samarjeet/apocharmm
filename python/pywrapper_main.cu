//#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
#include <vector>

#include "cuda_utils.h"

namespace py = pybind11;

void exportCharmmContext(py::module &mod);
void exportCoordinates(py::module &mod);
void exportCharmmCrd(py::module &mod);
void exportCharmmParameters(py::module &mod);
void exportCharmmPSF(py::module &mod);
void exportIntegrator(py::module &mod);
void exportForceManager(py::module &mod);
void exportPDB(py::module &mod);
void exportSubscriber(py::module &mod);
void exportForceManagerGenerator(py::module &mod);
void exportMinimizer(py::module &mod);
void exportPBC(py::module &mod);

PYBIND11_MODULE(_core, mod) {
  mod.doc() = R"pbdoc(
     python interface for apocharmm
     ------------------------------
     
     This mod contains all the cuda/C++ objects made available to the python
     API by pybind11.

  )pbdoc";

  std::vector<int> devices = {0, 1, 2, 3};
  start_gpu(1, 1, 0, devices);

  exportCharmmContext(mod);
  exportCoordinates(mod);
  exportCharmmCrd(mod);
  exportCharmmPSF(mod);
  exportCharmmParameters(mod);
  exportForceManager(mod);
  exportIntegrator(mod);
  exportPDB(mod);
  exportSubscriber(mod);
  exportForceManagerGenerator(mod);
  exportMinimizer(mod);
  exportPBC(mod);
}
