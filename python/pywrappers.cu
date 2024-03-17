// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author:  Andrew C. Simmonett
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CudaBondedForce.h"
#include "CudaExternalForce.h"
#include "CudaPMEDirectForce.h"
#include "CudaRestraintForce.h"
#include "ForceManager.h"
#include "PDB.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <string>

namespace py = pybind11;

namespace {
    // Accumulation, Computation precision mode
    template<typename A, typename C>
    void setupForces(py::module& mod, py::class_<ForceManager, std::shared_ptr<ForceManager> > &fm, std::string const& suffix) {
        //fm.def("emplace_back", &ForceManager::emplace_back<CudaBondedForce<A, C>>, "Add a CudaBondedForce to apply to the system");
        //fm.def("emplace_back", &ForceManager::emplace_back<CudaExternalForce<A, C>>, "Add a CudaExternalForce to apply to the system");
        //fm.def("emplace_back", &ForceManager::emplace_back<CudaRestraintForce<A, C>>, "Add a CudaRestraintForce to apply to the system");
        //fm.def("emplace_back", &ForceManager::emplace_back<CudaPMEDirectForce<A, C>>, "Add a CudaPMEDirectForce to apply to the system");

        py::class_<CudaBondedForce<A,C>, std::shared_ptr<CudaBondedForce<A,C>> > pyCBF(mod, ("CudaBondedForce"+suffix).c_str(), "Bonded force computer");
        pyCBF.def(py::init<CudaEnergyVirial&,const char*, const char*, const char*, const char*, const char*, const char* >());
        pyCBF.def("setup_coef", py::overload_cast<const std::vector<int> &, const std::vector<std::vector<float>> &>(&CudaBondedForce<A,C>::setup_coef),
                  "Setup coefficients for bonded force");
    }
}  // namespace

PYBIND11_MODULE(apocharmm, m) {
    m.doc() = R"pbdoc(
        apocharmm: 
                 ----------
                 .. currentmodule:: apocharmm

                 .. autosummary::
                    :toctree: _generate

    )pbdoc";

    py::class_<CharmmCrd, std::shared_ptr<CharmmCrd> > pyCharmmCrd(m, "CharmmCrd", "CHARMM coordinate file object");
    pyCharmmCrd.def(py::init<std::string>());
    //pyCharmmCrd.def("read_charmm_crd_file", &CharmmCrd::readCharmmCrdFile, "Read a charmm coordinate file");
    pyCharmmCrd.def("get_coordinates", &CharmmCrd::getCoordinates, "Get the list of coordinates");

    py::class_<PDB, std::shared_ptr<PDB> > pyPDB(m, "PDB", "PDB file object");
    pyPDB.def(py::init<std::string>());
    pyPDB.def("read_pdb_file", &PDB::readPDBFile, "Read a PDB file");

    py::class_<CharmmPSF, std::shared_ptr<CharmmPSF> > pyPSF(m, "PSF", "PSF file object");
    pyPSF.def(py::init<std::string>());

    py::class_<ForceManager, std::shared_ptr<ForceManager> > pyForceManager(m, "ForceManager", "Object to control forces to be applied to the system");
    pyForceManager.def(py::init<>());
    pyForceManager.def("calc_force", &ForceManager::calc_force, "Apply all forces added to this manager to the system");
    setupForces<long long, float>(m, pyForceManager, "Mixed");

    py::class_<CudaEnergyVirial, std::shared_ptr<CudaEnergyVirial> > pyCEV(m, "CudaEnergyVirial", "Stores energies and derivative components");
    pyCEV.def(py::init<>());
    pyCEV.def("get_energy", py::overload_cast<std::string&>(&CudaEnergyVirial::getEnergy), "Returns the energy of the component associated with the input label");

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif
}
