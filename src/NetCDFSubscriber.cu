// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "NetCDFSubscriber.h"
#include <iostream>

NetCDFSubscriber::NetCDFSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  ncid = -1;
  initialize();
  xyzNC = new float[this->charmmContext->getNumAtoms() * 3];
  numFramesWritten = 0;
}
NetCDFSubscriber::NetCDFSubscriber(const std::string &fileName, int reportFreq)
    : Subscriber(fileName, reportFreq) {
  ncid = -1;
  initialize();
  xyzNC = new float[this->charmmContext->getNumAtoms() * 3];
  numFramesWritten = 0;
}

NetCDFSubscriber::~NetCDFSubscriber() {
  std::cout << "Trying to close the ncid\n";
  int status = nc_close(ncid);
  if (status != NC_NOERR)
    std::cerr << "Error: Can't close the netcdf file\n";
    //throw std::invalid_argument("Error: Can't close the netcdf file\n");
  if (xyzNC != 0)
    delete[](xyzNC);
}

void NetCDFSubscriber::initialize() {
  int status;

  int dimensionID[NC_MAX_VAR_DIMS];
  status = nc_create(fileName.c_str(), NC_64BIT_OFFSET, &ncid);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't create the netcdf file\n";
    throw std::invalid_argument("ERROR : Can't create the netcdf file\n");

  // Dimensions
  status = nc_def_dim(ncid, "frame", NC_UNLIMITED, &frameDimId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define frame dimension.\n";
    throw std::invalid_argument("ERROR : Can't define frame dimension.\n");

  status = nc_def_dim(ncid, "spatial", 3, &spatialDimId);
  if (status != NC_NOERR)
    //std::cerr << "Can't define frame dimension.\n";
    throw std::invalid_argument("Can't define frame dimension.\n");

  status =
      nc_def_dim(ncid, "atom", this->charmmContext->getNumAtoms(), &atomDimId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define frame dimension.\n";
    throw std::invalid_argument("ERROR : Can't define frame dimension.\n");
  status = nc_def_dim(ncid, "cell_spatial", 3, &cellSpatialDimId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define cell spatial dimension.\n";
    throw std::invalid_argument("ERROR : Can't define cell spatial dimension.\n");

  status = nc_def_dim(ncid, "cell_angular", 3, &cellAngularDimId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define cell angular dimension.\n";
    throw std::invalid_argument("ERROR : Can't define cell angular dimension.\n");

  status = nc_def_dim(ncid, "label", 5, &labelDimId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define cell angular dimension.\n";
    throw std::invalid_argument("ERROR : Can't define cell angular dimension.\n");

  // Variables
  dimensionID[0] = frameDimId;
  status = nc_def_var(ncid, "time", NC_FLOAT, 1, dimensionID, &timeVariableId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define time variable.\n";
    throw std::invalid_argument("ERROR : Can't define time variable.\n");
  status = nc_put_att_text(ncid, timeVariableId, "units", 10, "picosecond");
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't put time variable unit attributes.\n";
    throw std::invalid_argument("ERROR : Can't put time variable unit attributes.\n");

  dimensionID[0] = spatialDimId;
  status =
      nc_def_var(ncid, "spatial", NC_CHAR, 1, dimensionID, &spatialVariableId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define spatial variable.\n";
    throw std::invalid_argument("ERROR : Can't define spatial variable.\n");

  dimensionID[0] = frameDimId;
  dimensionID[1] = cellSpatialDimId;
  status = nc_def_var(ncid, "cell_lengths", NC_FLOAT, 2, dimensionID,
                      &cellLengthsVariableId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define cell_lengths variable.\n";
    throw std::invalid_argument("ERROR : Can't define cell_lengths variable.\n");
  status = nc_put_att_text(ncid, cellLengthsVariableId, "units", 8, "angstrom");
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't cell_lengths units attribute.\n";
    throw std::invalid_argument("ERROR : Can't cell_lengths units attribute.\n");

  dimensionID[0] = frameDimId;
  dimensionID[1] = cellAngularDimId;
  status = nc_def_var(ncid, "cell_angles", NC_FLOAT, 2, dimensionID,
                      &cellAnglesVariableId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define cell_angles variable.\n";
    throw std::invalid_argument("ERROR : Can't define cell_angles variable.\n");
  status = nc_put_att_text(ncid, cellAnglesVariableId, "units", 6, "degree");
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't cell_angles units attribute.\n";
    throw std::invalid_argument("ERROR : Can't cell_angles units attribute.\n");

  dimensionID[0] = cellSpatialDimId;
  status = nc_def_var(ncid, "cell_spatial", NC_CHAR, 1, dimensionID,
                      &cellSpatialVariableId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define cell spatial variable.\n";
    throw std::invalid_argument("ERROR : Can't define cell spatial variable.\n");

  dimensionID[0] = frameDimId;
  dimensionID[1] = atomDimId;
  dimensionID[2] = spatialDimId;
  status = nc_def_var(ncid, "coordinates", NC_FLOAT, 3, dimensionID,
                      &coordVariableId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define coordinates variable.\n";
    throw std::invalid_argument("ERROR : Can't define coordinates variable.\n");
  status = nc_put_att_text(ncid, coordVariableId, "units", 8, "angstrom");
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't put coordinates variable unit attributes.\n";
    throw std::invalid_argument("ERROR : Can't put coordinates variable unit attributes.\n");

  dimensionID[0] = cellAngularDimId;
  dimensionID[1] = labelDimId;
  status = nc_def_var(ncid, "cell_angular", NC_CHAR, 2, dimensionID,
                      &cellAngularVariableId);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't define cell angular variable.\n";
    throw std::invalid_argument("ERROR : Can't define cell angular variable.\n");

  std::string title = "NetCDF title string";
  status =
      nc_put_att_text(ncid, NC_GLOBAL, "title", title.size(), title.c_str());
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't write title.\n";
    throw std::invalid_argument("ERROR : Can't write title.\n");

  status = nc_put_att_text(ncid, NC_GLOBAL, "application", 6, "CHARMM");
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't write application.\n";
    throw std::invalid_argument("ERROR : Can't write application.\n");

  status = nc_put_att_text(ncid, NC_GLOBAL, "program", 6, "CHARMM");
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't write program.\n";
    throw std::invalid_argument("ERROR : Can't write program.\n");

  std::string programVersion = "0.1";
  status = nc_put_att_text(ncid, NC_GLOBAL, "programVersion",
                           programVersion.size(), programVersion.c_str());
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't write programVersion.\n";
    throw std::invalid_argument("ERROR : Can't write programVersion.\n");

  std::string conventions = "AMBER";
  status = nc_put_att_text(ncid, NC_GLOBAL, "Conventions", conventions.size(),
                           conventions.c_str());
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't write conventions.\n";
    throw std::invalid_argument("ERROR : Can't write conventions.\n");

  status = nc_put_att_text(ncid, NC_GLOBAL, "ConventionVersion", 3, "1.0");
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't write convention version.\n";
    throw std::invalid_argument("ERROR : Can't write convention version.\n");

  status = nc_enddef(ncid);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't end definitions.\n";
    throw std::invalid_argument("ERROR : Can't end definitions.\n");

  start[0] = 0;
  count[0] = 3;
  char op[3];
  op[0] = 'x';
  op[1] = 'y';
  op[2] = 'z';
  status = nc_put_vara_text(ncid, spatialVariableId, start, count, op);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't write to spatial variable id.\n";
    throw std::invalid_argument("ERROR : Can't write to spatial variable id.\n");

  start[0] = 0;
  count[0] = 3;
  op[0] = 'a';
  op[1] = 'b';
  op[2] = 'c';
  status = nc_put_vara_text(ncid, cellSpatialVariableId, start, count, op);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't write to abc to cell spatial variable id.\n";
    throw std::invalid_argument("ERROR : Can't write to abc to cell spatial variable id.\n");

  char opAng[15] = {'a', 'l', 'p', 'h', 'a', 'b', 'e',
                    't', 'a', 'g', 'a', 'm', 'm', 'a'};
  start[0] = 0;
  start[1] = 0;
  count[0] = 3;
  count[1] = 5;
  status = nc_put_vara_text(ncid, cellAngularVariableId, start, count, opAng);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't write to abc to cell angular variable id.\n";
    throw std::invalid_argument("ERROR : Can't write to abc to cell angular variable id.\n");
}

void NetCDFSubscriber::formatDataForInput(const std::vector<float4> &xyzq) {
  for (int atomId = 0; atomId < this->charmmContext->getNumAtoms(); ++atomId) {
    xyzNC[3 * atomId] = xyzq[atomId].x;
    xyzNC[3 * atomId + 1] = xyzq[atomId].y;
    xyzNC[3 * atomId + 2] = xyzq[atomId].z;
  }
}

void convertDataFormat(int numAtoms, const std::vector<float4> &xyzq_in,
                       float *xyz) {
  for (int i = 0; i < numAtoms; ++i) {
    xyz[3 * i] = xyzq_in[i].x;
    xyz[3 * i + 1] = xyzq_in[i].y;
    xyz[3 * i + 2] = xyzq_in[i].z;
  }
}

void NetCDFSubscriber::update() {
  // std::cout << "In ncdf update\n";

  // vector in the shared ptr
  auto xyzq = *(this->charmmContext->getXYZQ()->getHostXYZQ());

  // should I just store numAtoms here, probably ok
  int numAtoms = this->charmmContext->getNumAtoms();

  start[0] = numFramesWritten;
  start[1] = 0;
  start[2] = 0;

  count[0] = 1;
  count[1] = numAtoms;
  count[2] = 3;

  // std::cout << xyzq[0].x << "\n";
  convertDataFormat(numAtoms, xyzq, xyzNC);

  int status = nc_put_vara_float(ncid, coordVariableId, start, count, xyzNC);
  if (status != NC_NOERR)
    //std::cerr << "ERROR : Can't write the frame\n";
    throw std::invalid_argument("ERROR : Can't write the frame\n");

  nc_sync(ncid);
  ++numFramesWritten;
}
