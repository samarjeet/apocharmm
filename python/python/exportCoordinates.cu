#include<pybind11/pybind11.h>
#include<pybind11/stl.h>

#include "Coordinates.h"
#include "CharmmCrd.h"
#include "PDB.h"

namespace py = pybind11;

void exportCoordinates(py::module & module){
  py::class_<Coordinates, std::shared_ptr<Coordinates>>(module, "Coordinates")
    //.def(py::init<const std::string &>(), "Handle for charmm .crd file")
    .def("getNumAtoms", &CharmmCrd::getNumAtoms, R"pbdoc(
      :return: The number of atoms
      :rtype: int
      )pbdoc")
    .def("getCoordinates", &CharmmCrd::getCoordinates, R"sitb(
      :return: coordinates of all atoms
      :rtype: vector<float4>
      )sitb");

}
void exportCharmmCrd(py::module & module){
  py::class_<CharmmCrd, std::shared_ptr<CharmmCrd>, Coordinates>(module, "CharmmCrd")
    .def(py::init<const std::string &>(), "Handle for charmm .crd file")
    //.def("getNumAtoms", &CharmmCrd::getNumAtoms, "number of atoms")
    //.def("getCoordinates", &CharmmCrd::getCoordinates, "coordinates of all atoms");
    ;
}

void exportPDB(py::module & module){
  py::class_<PDB, std::shared_ptr<PDB>, Coordinates>(module, "PDB", R"sitb(
     
     PDB file content

     Can be used to extract coordinate data via getCoordinates.
     
     )sitb")
    .def(py::init<const std::string &>(), "Handle for .pdb file")
    //.def("getNumAtoms", &CharmmCrd::getNumAtoms, "number of atoms")
    //.def("getCoordinates", &CharmmCrd::getCoordinates, "coordinates of all atoms");
    ;
}

