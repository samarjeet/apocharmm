#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PBC.h"

namespace py = pybind11;

void exportPBC(py::module &mod) {
  py::enum_<CRYSTAL>(mod, "CRYSTAL")
      .value("CUBIC", CRYSTAL::CUBIC)
      .value("ORTHORHOMBIC", CRYSTAL::ORTHORHOMBIC)
      .value("TETRAGONAL", CRYSTAL::TETRAGONAL);
}