#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaMinimizer.h"
#include "CudaNoseHooverThermostatIntegrator.h"
#include "CudaVMMSVelocityVerletIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "Subscriber.h"

namespace py = pybind11;

void exportIntegrator(py::module &module) {

  py::class_<CudaIntegrator, std::shared_ptr<CudaIntegrator>>(module,
                                                              "Integrator")
      .def(py::init<float>(), R"samaristhebest(
         Integrator. 
         
         Takes timestep size (in ps) as arg
         Requires a CharmmContext object as mediator (see setCharmmContext() )

         :param timeStep: integrator timestep, in ps
         :type timeStep: float
         
         )samaristhebest")
      /*.def("setCharmmContext",
           &CudaVelocityVerletIntegrator::setCharmmContext,
           "set the charmm context ")
      */
      /*.def("setReportSteps", &CudaVelocityVerletIntegrator::setReportSteps,
           R"samaristhebest(
            Report every reportSteps steps

            :param reportSteps:
            :type reportSteps: int
            )samaristhebest")
            */

      .def("propagate", &CudaVelocityVerletIntegrator::propagate,
           R"samaristhebest(
            Run numSteps steps

            :param numSteps: 
            :type numSteps: int
            )samaristhebest")
      .def("subscribe",
           static_cast<void (CudaIntegrator::*)(std::shared_ptr<Subscriber>)>(
               &CudaIntegrator::subscribe),
           R"sitb(
         Add a subscriber to the Integrator.

         :param sub: Subscriber to add

         )sitb")
      .def("subscribe",
           static_cast<void (CudaIntegrator::*)(
               std::vector<std::shared_ptr<Subscriber>>)>(
               &CudaIntegrator::subscribe),
           R"sitb(
      Add a list of subscribers to the Integrator.

      :param sublist: list of subscribers to add

      )sitb")
      .def("unsubscribe",
           static_cast<void (CudaIntegrator::*)(std::shared_ptr<Subscriber>)>(
               &CudaIntegrator::unsubscribe),
           R"sitb( Remove a subscriber.
         
         :param sub: Subscriber to remove
      )sitb")
      .def("unsubscribe",
           static_cast<void (CudaIntegrator::*)(
               std::vector<std::shared_ptr<Subscriber>>)>(
               &CudaIntegrator::unsubscribe),
           R"sitb(
         Remove subscribers given in a list.
         )sitb")
      .def("setDebugPrintFrequency", &CudaIntegrator::setDebugPrintFrequency,
           R"sitb(
         Set the frequency of debug printing.
         
         :param freq: Frequency of debug printing
         )sitb");

  // Velocity-Verlet
  //================
  py::class_<CudaVelocityVerletIntegrator,
             std::shared_ptr<CudaVelocityVerletIntegrator>, CudaIntegrator>(
      module, "VelocityVerletIntegrator")
      .def(py::init<float>(),
           "Velocity Verlet Integrator. Takes timestep (in ps) as arg")
      .def("setCharmmContext",
           &CudaVelocityVerletIntegrator::setCharmmContext,
           "set the charmm context ")
      //.def("setReportSteps", &CudaVelocityVerletIntegrator::setReportSteps,
      //"Report every <arg> steps") .def("propagate",
      //&CudaVelocityVerletIntegrator::propagate,"Run <arg> steps")
      ;
  py::class_<CudaVMMSVelocityVerletIntegrator,
             std::shared_ptr<CudaVMMSVelocityVerletIntegrator>, CudaIntegrator>(
      module, "VMMSVelocityVerletIntegrator")
      .def(py::init<float>(),
           "VMMS velocity Verlet Integrator. Takes timestep (in ps) as arg")
      .def("setCharmmContexts",
           &CudaVMMSVelocityVerletIntegrator::setCharmmContexts,
           "set the charmm context ")
      .def("setSoluteAtoms", &CudaVMMSVelocityVerletIntegrator::setSoluteAtoms,
           "set the solute atoms ")
      //.def("setReportSteps",
      //&CudaVMMSVelocityVerletIntegrator::setReportSteps, "Report every <arg>
      // steps") .def("propagate", &CudaVelocityVerletIntegrator::propagate,"Run
      //<arg> steps")
      ;

  // LANGEVIN THERMOSTAT
  //=====================
  py::class_<CudaLangevinThermostatIntegrator,
             std::shared_ptr<CudaLangevinThermostatIntegrator>, CudaIntegrator>(
      module, "LangevinThermostatIntegrator")
      .def(py::init<float>(),
           R"sitb(
               Langevin Thermostat Integrator. Takes timestep (in ps) as arg. If
               ran without a friction coefficient nor a bath temperature, will
               propagate as a NVE integrator.
               )sitb")
      .def(py::init<float, float, float>(),
           R"sitb(
               Langevin Thermostat Integrator. Takes timestep (ps), friction
               coefficient (ps-1) and bath temperature (K) as args.
               :param timeStep: integrator timestep, in ps
               :type timeStep: float
               :param bathTemperature: bath temperature, in K
               :type bathTemperature: float
               :param friction: friction coefficient, in units /ps
               :type friction: float
               )sitb")
      .def("setCharmmContext",
           &CudaLangevinThermostatIntegrator::setCharmmContext,
           "set the charmm context ")
      .def("setBathTemperature",
           &CudaLangevinThermostatIntegrator::setBathTemperature,
           "set the bath temperature")
      .def("setFriction", &CudaLangevinThermostatIntegrator::setFriction,
           "set the friction (in units /ps)");

  // NOSE-HOOVER
  //=====================
  py::class_<CudaNoseHooverThermostatIntegrator,
             std::shared_ptr<CudaNoseHooverThermostatIntegrator>,
             CudaIntegrator>(module, "NoseHooverThermostatIntegrator")
      .def(py::init<float>(),
           "Nose-Hoover Thermostat Integrator. Takes timestep (in ps) as arg");

  // Langevin Piston
  //================
  py::class_<CudaLangevinPistonIntegrator,
             std::shared_ptr<CudaLangevinPistonIntegrator>, CudaIntegrator>(
      module, "LangevinPistonIntegrator", R"sitb(
          Langevin Piston integrator. 
          Default bath temperature value: 300K. 
          Default piston mass value: 500 AU.
          )sitb")
      .def(py::init<float>(),
           "Langevin piston integrator. Takes timestep (in ps) as arg")
      .def("setCharmmContext",
           &CudaLangevinPistonIntegrator::setCharmmContext,
           "set the CharmmContext ")
      //      .def("setPistonMass",
      //           py::overload_cast<double>(
      //               &CudaLangevinPistonIntegrator::setPistonMass),
      //           "set the piston mass. Default: 500. DO NOT GIVE A SINGLE
      //           FLOAT " "VALUE 0. (worst case give a list [0.]) (this should
      //           be fixed in " "the cuda code)")
      .def("setPistonMass", &CudaLangevinPistonIntegrator::setPistonMass,
           "set the piston mass using an array. Default: 500., 500.")
      .def("getPistonMass", &CudaLangevinPistonIntegrator::getPistonMass,
           "get the pressure piston mass")
      .def("setPistonFriction",
           &CudaLangevinPistonIntegrator::setPistonFriction,
           "set the friction coefficient. Has to be set. Default: 0.0 ps^-1")
      .def("setBathTemperature",
           &CudaLangevinPistonIntegrator::setBathTemperature,
           "set the bath temperature. Default: 300.0")
      .def("setCrystalType", &CudaLangevinPistonIntegrator::setCrystalType,
           "set the crystal type. Default: CRYSTAL.ORTHORHOMBIC")
      .def("setNoseHooverPistonMass",
           &CudaLangevinPistonIntegrator::setNoseHooverPistonMass,
           "set the Nose-Hoover piston mass. Default: 500.0")
      .def("setNoseHooverFlag",
           &CudaLangevinPistonIntegrator::setNoseHooverFlag,
           "set the Nose-Hoover flag. Default: true")
      .def("setSurfaceTension",
           &CudaLangevinPistonIntegrator::setSurfaceTension,
           "set the surface tension. Default: 0.0")
      .def("getNoseHooverPistonMass",
           &CudaLangevinPistonIntegrator::getNoseHooverPistonMass,
           "get the Nose-Hoover piston mass")
      .def("setNoseHooverFlag",
           &CudaLangevinPistonIntegrator::setNoseHooverFlag,
           "set the Nose-Hoover flag. Set it to False to run NPH rather than "
           "NPT. Default: true")
     .def("initialize", &CudaLangevinPistonIntegrator::initialize,
           "initialize the integrator variables. Should be used when starting a"
           " simulation of another system using the same integrator. ");
  //      .def("setDebugPrintFrequency",
  //           &CudaLangevinPistonIntegrator::setDebugPrintFrequency,
  //           "set the debug frequency. Default: 0");
}

void exportMinimizer(py::module &module) {
  // Minimizers ?
  //================
  py::class_<CudaMinimizer, std::shared_ptr<CudaMinimizer>>(module, "Minimizer")
      .def(py::init(), R"samaristhebest(
         LBFGS Minimizer. 

         )samaristhebest")
      .def("setCharmmContext", &CudaMinimizer::setCharmmContext,
           R"sitb(
            set the charmm context 

            :param csc: CharmmContext
            :type csc: CharmmContext

            )sitb")
      .def("minimize", py::overload_cast<int>(&CudaMinimizer::minimize),
           R"sitb(
           Takes numSteps minimization steps [default 100]
           :param numSteps: 
           :type numSteps: int
           )sitb")
      .def("minimize", py::overload_cast<>(&CudaMinimizer::minimize),
           "Takes number of steps [default 100]");
}
