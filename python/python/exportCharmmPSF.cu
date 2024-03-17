#include "CharmmPSF.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void exportCharmmPSF(py::module &module) {
  py::class_<CharmmPSF, std::shared_ptr<CharmmPSF>>(module, "CharmmPSF")
      .def(py::init<const std::string &>(), "Handle for charmm .psf file")
      .def("getNumAtoms", &CharmmPSF::getNumAtoms, "number of atoms.")
      .def("getNumBonds", &CharmmPSF::getNumBonds, "number of bonds")
      .def("getNumAngles", &CharmmPSF::getNumAngles, "number of angles")
      .def("getNumDihedrals", &CharmmPSF::getNumDihedrals,
           "number of dihedrals")
      .def("getNumImpropers", &CharmmPSF::getNumImpropers,
           "number of impropers");
}
