#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BEDSSubscriber.h"
#include "CharmmContext.h"
#include "CompositeSubscriber.h"
#include "DcdSubscriber.h"
#include "DualTopologySubscriber.h"
#include "FEPSubscriber.h"
#include "MBARSubscriber.h"
#include "NetCDFSubscriber.h"
#include "RestartSubscriber.h"
#include "StateSubscriber.h"
#include "Subscriber.h"
#include "XYZSubscriber.h"

namespace py = pybind11;

void exportSubscriber(py::module &mod) {

  py::class_<Subscriber, std::shared_ptr<Subscriber>> pySubscriber(
      mod, "Subscriber");
  pySubscriber
      //.def(py::init<const std::string &, std::shared_ptr<CharmmContext>>())
      .def("setFileName", &Subscriber::setFileName, R"sitb(
         :param fileNameIn: name of the output file
         :type fileNameIn: str
         )sitb")
      .def("addCommentSection", &Subscriber::addCommentSection, R"sitb(
            Add some text to the Subscriber's output file. 
            :param commentIn: text to be added. If the last character is not a
            line break, will add it to keep formatting alright.
            :type commentIn: string
        )sitb");
  ;

  // NetCDFSubscriber
  //=================
  py::class_<NetCDFSubscriber, std::shared_ptr<NetCDFSubscriber>>(
      mod, "NetCDFSubscriber", pySubscriber, "NetCDF subscriber")
      .def(py::init<const std::string &>(),
           "NetCDF output. Takes .nc file name and the CHARMM context as args");

  // DcdSubscriber
  //==============
  py::class_<DcdSubscriber, std::shared_ptr<DcdSubscriber>>(
      mod, "DcdSubscriber", pySubscriber,
      R"sitb(
            DCD output. Takes .dcd file name and possibly a report frequency
            (default: every 1000 steps) as arguments
            )sitb")
      .def(py::init<const std::string &>(),
           "DCD Susbcriber constructor. Takes output file name as argument.")
      .def(py::init<const std::string &, int>(),
           R"sitb(
            DCD output constructor
            :param fileName: output file name
            :type fileName: str
            :param reportFreq: report interval (number of steps)
            :type reportFreq: int
            )sitb");

  // XYZSubscriber
  //==============
  py::class_<XYZSubscriber, std::shared_ptr<XYZSubscriber>>(
      mod, "XYZSubscriber", pySubscriber,
      R"sitb(
            XYZ subscriber. Writes out .xyz files at a given frequency (default: every 1000 steps)
        )sitb")
      .def(py::init<const std::string &>(),
           R"sitb(
            XYZSubscriber constructor. Takes output file name as argument

            :param fileName: desired output file
            :type fileName: str
           )sitb")
      .def(py::init<const std::string &, int>(),
           R"sitb(
            XYZSubscriber constructor. Takes output file name and report frequency (default 1000) as arguments.

            :param fileName: desired output file
            :type fileName: str
            :param reportFreq: intervals between two reports (number of steps)
            :type reportFreq: int
           )sitb");

  // StateSubscriber
  //================
  py::class_<StateSubscriber, std::shared_ptr<StateSubscriber>>(
      mod, "StateSubscriber", pySubscriber, R"sitb(
         Reports, at a specified frequency, quantities of interest. 
         Choosing the quantities to be reported can be done through setReportFlags. 
         Available quantities (and their corresponding flags) are:
          - kinetic energy (kineticenergy) *
          - potential energy (potentialenergy) * 
          - total energy (totalenergy) *
          - temperature (temperature) *
          - pressure (pressurescalar) *
          - pressure components (pressurecomponents)
          - box dimensions (boxsizecomponents)
          - box volume (volume)
          - density (density) 

        A star (*) denotes that the quantity is reported by default.
         )sitb")
      .def(py::init<const std::string &>(),
           R"sitb( 
         StateSubscriber constructor. 
         
         :param string: [optional] name of the output file
         :type ???: string
            )sitb")
      .def(py::init<const std::string &, int>(),
           R"sitb( 
                        StateSusbcriber constructor, uses a report frequency

                        :param reportFreq: interval between two reports
                        :type reportFreq: int
                    )sitb")
      .def("setReportFlags",
           static_cast<void (StateSubscriber::*)(std::string)>(
               &StateSubscriber::readReportFlags),
           R"sitb(
            Set the output quantities for the StateSubscriber. Ex: "all", or "kineticEnergy, potentialenergy, VOLUME". 
            :param flags: string of quantities to be reported, separated by a comma.
            :type flags: str
            )sitb")
      .def("setReportFlags",
           static_cast<void (StateSubscriber::*)(std::vector<std::string>)>(
               &StateSubscriber::readReportFlags),
           R"sitb(
            Set the output quantities for the StateSubscriber. Ex: ["kineticEnergy", "potentialenergy", "VOLUME"]. 
            :param flags: quantities to be reported.
            :type flags: list[str]
            )sitb");

  // DualTopologySubscriber
  //========================
  py::class_<DualTopologySubscriber, std::shared_ptr<DualTopologySubscriber>>(
      mod, "DualTopologySubscriber", pySubscriber, "Dual Topology Subscriber")
      .def(py::init<const std::string &>(),
           "PE0, PE1, (1-lambda)PE0 + (lambda)PE1 reporter. Takes .txt file "
           "name and the CHARMM context as args")
      .def(py::init<const std::string &, int>(),
           "PE0, PE1, (1-lambda)PE0 + (lambda)PE1 reporter. Takes .txt file "
           "name and the CHARMM context as args");

  // CompositeSubscriber
  //====================
  py::class_<CompositeSubscriber, std::shared_ptr<CompositeSubscriber>>(
      mod, "CompositeSubscriber", pySubscriber, "Composite Subscriber")
      .def(py::init<const std::string &>(),
           "lambda potential energies reporter. Takes .txt file name and the "
           "CHARMM context as args")
      .def(py::init<const std::string &, int>(), R"sitb(
                CompositeSubscriber constructor, using a file name and a report frequency.

                :param reportFileName: output file name
                :type reportFileName: str
                :param reportFreq: interval between two reports
                :type reportFreq: int 
            )sitb");

  // RestartSubscriber
  //==================
  py::class_<RestartSubscriber, std::shared_ptr<RestartSubscriber>>(
      mod, "RestartSubscriber", pySubscriber,
      R"sitb(
          Restart Subscriber.
          Writes data required to restart simulations every "
      "reportFreq steps. To read a restart file, use readRestart function.
      )sitb")
      .def(py::init<const std::string &>(),
           "Restart Subscriber constructor. Takes output file name as argument")
      .def(py::init<const std::string &, int>(),
           R"sitb(
                RestartSubscriber constructor, using a file name and a report frequency.

                :param reportFileName: output file name
                :type reportFileName: str
                :param reportFreq: interval between two reports
                :type reportFreq: int 
            )sitb")
      .def("readRestart", &RestartSubscriber::readRestart, R"sitb(
            Read the restart file associated with the current subscriber and
            sets the simulation parameters accordingly
            :param restartFileName: name of the restart file 
            :type restartFileName: str
            )sitb");

  // MBARSubscriber
  //===============
  py::class_<MBARSubscriber, std::shared_ptr<MBARSubscriber>>(
      mod, "MBARSubscriber", pySubscriber, "MBAR Subscriber")
      .def(py::init<const std::string &>(),
           "MBAR Subscriber constructor. Takes output file name as argument")
      .def(py::init<const std::string &, int>(),
           R"sitb(
                MBARSubscriber constructor, using a file name and a report frequency.

                :param reportFileName: output file name
                :type reportFileName: str
                :param reportFreq: interval between two reports
                :type reportFreq: int 
            )sitb");

  // BEDSSubsriber
  //===============
  py::class_<BEDSSubscriber, std::shared_ptr<BEDSSubscriber>>(
      mod, "BEDSSubscriber", pySubscriber, "BEDS Subscriber")
      .def(py::init<const std::string &>(),
           "BEDS Subscriber constructor. Takes output file name as argument")
      .def(py::init<const std::string &, int>(),
           R"sitb(
                    BEDSSubscriber constructor, using a file name and a report frequency.
     
                    :param reportFileName: output file name
                    :type reportFileName: str
                    :param reportFreq: interval between two reports
                    :type reportFreq: int 
               )sitb");

  // FEPSubscriber
  //==============
  py::class_<FEPSubscriber, std::shared_ptr<FEPSubscriber>>(
      mod, "FEPSubscriber", pySubscriber, "FEP Subscriber")
      .def(py::init<const std::string &>(),
           "FEP Subscriber constructor. Takes output file name as argument")
      .def(py::init<const std::string &, int>(),
           R"sitb(
                FEPSubscriber constructor, using a file name and a report frequency.

                :param reportFileName: output file name
                :type reportFileName: str
                :param reportFreq: interval between two reports
                :type reportFreq: int 
            )sitb");
}
