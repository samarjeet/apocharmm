#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CharmmCrd.h"
#include "Coordinates.h"
#include "PDB.h"

namespace py = pybind11;

void exportCoordinates(py::module &mod) {
  py::class_<Coordinates, std::shared_ptr<Coordinates>>(mod, "Coordinantes")
      .def("getNumAtoms", &Coordinates::getNumAtoms, "Number of atoms.")
      .def("getCoordinatesD",
           static_cast<std::vector<double4> &(Coordinates::*)(void)>(
               &Coordinates::getCoordinatesD),
           "Double precision coordinates.")
      .def("getCoordinatesD",
           static_cast<std::vector<float4> &(Coordinates::*)(void)>(
               &Coordinates::getCoordinatesF),
           "Single precision coordinates.");
}

void exportCharmmCrd(py::module &mod) {
  py::class_<CharmmCrd, std::shared_ptr<CharmmCrd>, Coordinates>(mod,
                                                                 "CharmmCrd")
      .def(py::init<const std::string &>(), "Handle for charmm .crd file")
      //.def("getNumAtoms", &CharmmCrd::getNumAtoms, "number of atoms")
      //.def("getCoordinates", &CharmmCrd::getCoordinates, "coordinates of all
      // atoms");
      ;
}

void exportPDB(py::module &mod) {
  py::class_<PDB, std::shared_ptr<PDB>, Coordinates>(mod, "PDB", R"sitb(
     
     PDB file content

     Can be used to extract coordinate data via getCoordinates.
     
     )sitb")
      .def(py::init<const std::string &>(), "Handle for .pdb file")
      //.def("getNumAtoms", &CharmmCrd::getNumAtoms, "number of atoms")
      //.def("getCoordinates", &CharmmCrd::getCoordinates, "coordinates of all
      // atoms");
      ;
}
