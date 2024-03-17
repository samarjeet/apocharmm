#include<pybind11/pybind11.h>
#include<pybind11/stl.h>

#include "CharmmParameters.h"

namespace py = pybind11;

void exportCharmmParameters(py::module& module){
  py::class_<CharmmParameters, std::shared_ptr<CharmmParameters>>(module, "CharmmParameters", R"sitb(
     CHARMM parameters. Can be constructed from a file name or a list of file names.
     
     )sitb")
    .def(py::init<const std::string &>(), R"sitb(
      Constructor for CHARMM .prm file
      
      :param filename: CHARMM .prm or .str filename
      :type filename: str

      )sitb")
    .def(py::init<const std::vector<std::string> &>(), R"sitb(
      Constructor for multiple CHARMM .prm file

      :param filename: list of CHARMM .prm or .str filename
      :type filename: list[str]

      )sitb");
}
