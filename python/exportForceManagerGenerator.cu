#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ForceManagerGenerator.h"

namespace py = pybind11;

void exportForceManagerGenerator(py::module &mod) {
   py::class_<ForceManagerGenerator, std::shared_ptr<ForceManagerGenerator>>(
         mod, "ForceManagerGenerator")
      .def(py::init<std::shared_ptr<ForceManager> &>(), "thedoc")
      .def("generateForceManager", &ForceManagerGenerator::generateForceManager, "");

   py::class_<AlchemicalForceManagerGenerator,                       // 
              std::shared_ptr<AlchemicalForceManagerGenerator>,      // 
              ForceManagerGenerator>(                                // Parent class (fac)
                    mod,                                          // ?
                    "AlchemicalForceManagerGenerator",               // Name of the created class ?
                    R"sitb(                                          
                        AlchemicalForceManagerGenerator 

                        ForceManager factory, designed to create ForceManager
                        objects based on a basic one provided by the user.  

                 )sitb")                                             // doc

      .def(py::init<                                                 // python constructor
            std::shared_ptr<ForceManager>>(),                        // constructor's args types
            R"sitb(
         AlchemicalForceManagerGenerator constructor

         Creates an AlchemicalForceManagerGenerator from the base ForceManager
         object that will be used as model for all future generated
         ForceManager objects.
         :param baseForceManager: input ForceManager object

      )sitb")
   
      .def("setAlchemicalRegion",                                    // function name
           &AlchemicalForceManagerGenerator::setAlchemicalRegion,    // C++ function
           R"sitb(
         Sets the alchemical region as a list of indices.
         :param alchRegionIn: indices of the atoms in the alchemical region (0-based ?)
         :type alchRegionIn: list[int]
           )sitb")                                                   // doc

      .def("generateForceManager", 
            &AlchemicalForceManagerGenerator::generateForceManager, 
            R"sitb(
      Given a lambda_elec and a lambda_vdw value, returns a ForceManager with scaled electrostatics and van der Waals interactions. 
      /!\\ VDW NOT AVAILABLE
      :param lambdaElecIn: factor scaling electrostatic interactions
      :param lambdaVdWIn: factor scaling van der Waals interactions (NOT IMPL.)
      )sitb");
}
