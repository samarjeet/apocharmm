#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CharmmContext.h"
#include "Coordinates.h"
#include "PBC.h"

namespace py = pybind11;

void exportCharmmContext(py::module &mod) {
  py::enum_<PBC>(mod, "PBC")
      .value("P1", PBC::P1)
      .value("P21", PBC::P21)
      .export_values();

  py::class_<CharmmContext, std::shared_ptr<CharmmContext>>(mod,
                                                            "CharmmContext")
      .def(py::init<std::shared_ptr<ForceManager>>(),
           R"sitb(
            CHARMM context constructor
            
            :param fmIn: ForceManager object. 
            )sitb")
      //      .def("overloadTest",
      //           static_cast<void (CharmmContext::*)(
      //               const std::shared_ptr<Coordinates>)>(
      //               &CharmmContext::overloadTest),
      //           "overload test 1")
      //      .def("overloadTest",
      //           static_cast<void (CharmmContext::*)(
      //               const std::vector<std::vector<double>>)>(
      //               &CharmmContext::overloadTest),
      //           "overload test 2")
      .def("setCoordinates",
           static_cast<void (CharmmContext::*)(
               const std::shared_ptr<Coordinates>)>(
               &CharmmContext::setCoordinates),
           "takes CHARMM crd object")
      .def("setCoordinates",
           static_cast<void (CharmmContext::*)(
               const std::vector<std::vector<double>>)>(
               &CharmmContext::setCoordinates),
           "takes 3*N array of coordinates")
      //.def("setCoordinates", &CharmmContext::setCoordinates,
      //     "sets coordinates using either a CharmmCrd object or a 3*N array")
      .def("readRestart", &CharmmContext::readRestart,
           "reads initial positions, velocities and possibly box dimensions "
           "from a CHARMM-format restart file. Args: filename")
      .def("setPeriodicBoundaryCondition",
           &CharmmContext::setPeriodicBoundaryCondition,
           "Use P1 or P21 PBC for simulation")
      .def("getVolume", &CharmmContext::getVolume, "get box volume")
      .def("getKineticEnergy", &CharmmContext::getKineticEnergy,
           "get the kinetic energy")
      .def("assignVelocitiesAtTemperature",
           &CharmmContext::assignVelocitiesAtTemperature,
           "assign Boltzmann distributed velocities. Takes temperature (in K) "
           "as input")
      .def("calculatePotentialEnergy", &CharmmContext::calculatePotentialEnergy,
           "calculate potential energy")
      .def("calculateForces", &CharmmContext::calculateForces,
           "calculate potential energy and forces. Args : reset (bool), "
           "calcEnergy(bool), print(bool)")
      .def("computePressure", &CharmmContext::computePressure,
           "Calculate pressure")
      .def("computeTemperature", &CharmmContext::computeTemperature,
           "Computes temperature using Kinetic energy and number of dofs.")
      .def("getDegreesOfFreedom", &CharmmContext::getDegreesOfFreedom,
           "get the number of degrees of freedom")
      .def("useHolonomicConstraints", &CharmmContext::useHolonomicConstraints,
           "Use holonomic constraints")
      .def("getBoxDimensions", &CharmmContext::getBoxDimensions,
           "get box dimensions");
}
