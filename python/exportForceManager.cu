#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "EDSForceManager.h"
#include "FEPEIForceManager.h"
#include "ForceManager.h"
#include "MBARForceManager.h"

namespace py = pybind11;

enum NonBondedType {
  VDW_VFSW_c = 3,
  VDW_DBEXP_c = 6,
};

void exportForceManager(py::module &mod) {

  py::enum_<NonBondedType>(mod, "NonBondedType")
      .value("VDW_VFSW", NonBondedType::VDW_VFSW_c)
      .value("VDW_DBEXP", NonBondedType::VDW_DBEXP_c)
      .export_values();

  py::class_<ForceManager, std::shared_ptr<ForceManager>>(mod, "ForceManager",
                                                          R"sitb(
     ForceManager object.
     
     Creates a ForceManager, assembling topology and force-field parameters. 
     Contains simulation parameters regarding PME (FFT options, cutoffs).


     )sitb")
      .def(py::init<std::shared_ptr<CharmmPSF>,
                    std::shared_ptr<CharmmParameters>>(),
           R"pbdoc(
                  ForceManager constructor.
                  
                  Creates a ForceManager object based on a CharmmPSF and a CharmmParameters object.

                  :param psfIn: input CharmmPSF object
                  :type psfIn: CharmmPSF
                  :param prmIn: input CharmmParameters object
                  :type prmIn: CharmmParameters

            )pbdoc")
      .def("setBoxDimensions", &ForceManager::setBoxDimensions, R"pbdoc(
               Sets x,y,z dimensions of simulation box

               :param x, y, z: box dimensions in AA
               :type x, y, z: float

            )pbdoc")
      .def(
          "getBoxDimensions",
          static_cast<const std::vector<double> &(ForceManager::*)(void) const>(
              &ForceManager::getBoxDimensions),
          R"pbdoc(
               Gets x,y,z dimensions of simulation box

               :return: box dimensions in AA
               :rtype: tuple of floats

            )pbdoc")
      .def("getBoxDimensions",
           static_cast<std::vector<double> &(ForceManager::*)(void)>(
               &ForceManager::getBoxDimensions),
           R"pbdoc(
               Gets x,y,z dimensions of simulation box

               :return: box dimensions in AA
               :rtype: tuple of floats

            )pbdoc")
      .def("setPeriodicBoundaryCondition",
           &ForceManager::setPeriodicBoundaryCondition,
           "Use P1 or P21 PBC for simulation")
      .def("setFFTGrid", &ForceManager::setFFTGrid,
           "grid dimensions along x, y and Z axes")
      .def("setKappa", &ForceManager::setKappa,
           "kappa value for PME. Default value : 0.34")
      .def("setPmeSplineOrder", &ForceManager::setPmeSplineOrder,
           "spline order value for PME. Default value : 4")
      .def("setCutoff", &ForceManager::setCutoff, R"sitb(
         cutoff for non-bonded list preparation. Default: 11.0 Angstrom.

         :param cutoffIn: cutoff value, in Angstrom
         :type cutoffIn: float
         )sitb")
      .def("setVdwType", &ForceManager::setVdwType, R"sitb(
         VdW type. 

         :param vdwTypeIn: VdW type
         :type vdwTypeIn: int
         )sitb")
      .def("setCtonnb", &ForceManager::setCtonnb, R"sitb(
         cut-on value for non-bonded forces. Default: 8.0 Angstrom.

         :param ctonnbIn: cut-on value, in Angstrom
         :type ctonnbIn: float

         )sitb")
      .def("setCtofnb", &ForceManager::setCtofnb, R"sitb(
         cut-off value for non-bonded forces.  Default: 9.5 Angstrom.
         
         :param ctofnbIn: cut-off value, in Angstrom
         :type ctofnbIn: float

         )sitb")
      .def("setPrintEnergyDecomposition",
           &ForceManager::setPrintEnergyDecomposition,
           "print energy decomposition")
      .def("initialize", &ForceManager::initialize,
           R"sitb(
             Create all force terms added to the ForceManager.
             )sitb");

  py::class_<ForceManagerComposite, std::shared_ptr<ForceManagerComposite>,
             ForceManager>(mod, "ForceManagerComposite")
      .def(py::init<>(), "ForceManagerComposite constructor")
      .def(py::init<std::vector<std::shared_ptr<ForceManager>> &>(), R"sitb(
            Constructor based on a list of ForceManager. Each will be added to
            the composite.
            )sitb")
      .def("addForceManager", &ForceManagerComposite::addForceManager,
           "add a forceManager to this composite object")
      .def("setBoxDimensions", &ForceManagerComposite::setBoxDimensions,
           "set x,y,z dimensions of box for all force managers")
      .def("setFFTGrid", &ForceManagerComposite::setFFTGrid,
           "grid dimensions along x, y and Z axes for all force managers")
      .def("setKappa", &ForceManagerComposite::setKappa,
           "kappa value for PME for all force managers. Default value : 0.34")
      .def("setCutoff", &ForceManagerComposite::setCutoff,
           "cutoff for non-bonded list preparation for all force managers")
      .def("setCtonnb", &ForceManagerComposite::setCtonnb,
           "cut-on value for non-bonded forces for all force managers")
      .def("setCtofnb", &ForceManagerComposite::setCtofnb,
           "cut-off value for non-bonded forces for all force managers ")
      .def("initialize", &ForceManagerComposite::initialize,
           "Initialize all ForceManagers after setting the values")
      .def("getLambda", &ForceManagerComposite::getLambda,
           "get the lambda value")
      .def("setLambda", &ForceManagerComposite::setLambda,
           "set the lambda value")
      .def("setSelectorVec", &ForceManagerComposite::setSelectorVec,
           "set the lambda value");

  py::class_<EDSForceManager, std::shared_ptr<EDSForceManager>,
             ForceManagerComposite>(mod, "EDSForceManager")
      .def(py::init<>(), "EDSForceManager constructor")
      .def(py::init<std::shared_ptr<ForceManager>,
                    std::shared_ptr<ForceManager>>(),
           R"sitb(
         Constructor taking two ForceManager objects as arguments.
         )sitb")
      .def("initialize", &EDSForceManager::initialize,
           "Initialize all ForceManagers after setting the values")
      .def("setSValue", &EDSForceManager::setSValue, "set the s value for EDS")
      .def("setEnergyOffsets", &EDSForceManager::setEnergyOffsets,
           "set the energy offsets for EDS");

  py::class_<MBARForceManager, std::shared_ptr<MBARForceManager>,
             ForceManagerComposite>(mod, "MBARForceManager",
                                    R"sitb(
                MBARForceManager object.
                Currently, initialization must be done following a specific order. 
                - create the MBARForceManager with a list of FM (or use the addForceManager function)
                - create a CharmmContext with that MBARForceManager
                - set coordinates of the context (setCoordinates function)
                - set MBARForceManager selector vector (deciding which alchemical state drives the simulation, using the setSelectorVec function)
          )sitb")
      .def(py::init<>(), "MBARForceeManager constructor. ")
      .def(py::init<std::vector<std::shared_ptr<ForceManager>> &>(),
           R"sitb(
         Constructor taking a list of ForceManager objects as arguments.
         )sitb");

  py::class_<FEPEIForceManager, std::shared_ptr<FEPEIForceManager>,
             ForceManagerComposite>(mod, "FEPEIForceManager")
      .def(py::init<>(), "FEPEIForceManager constructor")
      .def("setLambda", &FEPEIForceManager::setLambda, "Set the driving state")
      .def("setLambdas", &FEPEIForceManager::setLambdas,
           "Set the alchemical windows schedule (each value defines an EI "
           "state)");
}
