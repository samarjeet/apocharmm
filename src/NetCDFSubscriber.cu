// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "NetCDFSubscriber.h"

#include "CharmmContext.h"
#include <iostream>

NetCDFSubscriber::NetCDFSubscriber(const std::string &fileName)
    : Subscriber(fileName) {
  m_NcId = -1;
  this->initialize();
  m_XYZ.resize(3 * m_CharmmContext->getNumAtoms());
  m_NumFramesWritten = 0;
}

NetCDFSubscriber::NetCDFSubscriber(const std::string &fileName,
                                   int reportFrequency)
    : Subscriber(fileName, reportFrequency) {
  m_NcId = -1;
  this->initialize();
  m_XYZ.resize(3 * m_CharmmContext->getNumAtoms());
  m_NumFramesWritten = 0;
}

NetCDFSubscriber::~NetCDFSubscriber(void) {
  std::cout << "Trying to close the ncid\n";
  int status = nc_close(m_NcId);
  if (status != NC_NOERR)
    std::cerr << "Error: Can't close the netcdf file\n";
  // throw std::invalid_argument("Error: Can't close the netcdf file\n");
}

void NetCDFSubscriber::update() {
  m_Start[0] = m_NumFramesWritten;
  m_Start[1] = 0;
  m_Start[2] = 0;

  m_Count[0] = 1;
  m_Count[1] = m_CharmmContext->getNumAtoms();
  m_Count[2] = 3;

  // this->formatDataForInput(*(m_CharmmContext->getXYZQ()->getHostXYZQ()));
  this->formatDataForInput(m_CharmmContext->getXYZQ().getHostArray());

  int status = nc_put_vara_float(m_NcId, m_CoordVariableId, m_Start.data(),
                                 m_Count.data(), m_XYZ.data());
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't write the frame\n");

  nc_sync(m_NcId);
  m_NumFramesWritten++;

  return;
}

void NetCDFSubscriber::initialize(void) {
  int status;
  int dimensionID[NC_MAX_VAR_DIMS];

  status = nc_create(m_FileName.c_str(), NC_64BIT_OFFSET, &m_NcId);
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't create the netcdf file\n");

  // Dimensions
  status = nc_def_dim(m_NcId, "frame", NC_UNLIMITED, &m_FrameDimId);
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't define frame dimension.\n");

  status = nc_def_dim(m_NcId, "spatial", 3, &m_SpatialDimId);
  if (status != NC_NOERR)
    throw std::invalid_argument("Can't define frame dimension.\n");

  status =
      nc_def_dim(m_NcId, "atom", m_CharmmContext->getNumAtoms(), &m_AtomDimId);
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't define frame dimension.\n");

  status = nc_def_dim(m_NcId, "cell_spatial", 3, &m_CellSpatialDimId);
  if (status != NC_NOERR)
    throw std::invalid_argument(
        "ERROR : Can't define cell spatial dimension.\n");

  status = nc_def_dim(m_NcId, "cell_angular", 3, &m_CellAngularDimId);
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't define cell angular dimension.\n");
  }

  status = nc_def_dim(m_NcId, "label", 5, &m_LabelDimId);
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't define cell angular dimension.\n");
  }

  // Variables
  dimensionID[0] = m_FrameDimId;
  status =
      nc_def_var(m_NcId, "time", NC_FLOAT, 1, dimensionID, &m_TimeVariableId);
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't define time variable.\n");

  status = nc_put_att_text(m_NcId, m_TimeVariableId, "units", 10, "picosecond");
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't put time variable unit attributes.\n");
  }

  dimensionID[0] = m_SpatialDimId;
  status = nc_def_var(m_NcId, "spatial", NC_CHAR, 1, dimensionID,
                      &m_SpatialVariableId);
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't define spatial variable.\n");

  dimensionID[0] = m_FrameDimId;
  dimensionID[1] = m_CellSpatialDimId;
  status = nc_def_var(m_NcId, "cell_lengths", NC_FLOAT, 2, dimensionID,
                      &m_CellLengthsVariableId);
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't define cell_lengths variable.\n");
  }

  status =
      nc_put_att_text(m_NcId, m_CellLengthsVariableId, "units", 8, "angstrom");
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't cell_lengths units attribute.\n");
  }

  dimensionID[0] = m_FrameDimId;
  dimensionID[1] = m_CellAngularDimId;
  status = nc_def_var(m_NcId, "cell_angles", NC_FLOAT, 2, dimensionID,
                      &m_CellAnglesVariableId);
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't define cell_angles variable.\n");

  status =
      nc_put_att_text(m_NcId, m_CellAnglesVariableId, "units", 6, "degree");
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't cell_angles units attribute.\n");

  dimensionID[0] = m_CellSpatialDimId;
  status = nc_def_var(m_NcId, "cell_spatial", NC_CHAR, 1, dimensionID,
                      &m_CellSpatialVariableId);
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't define cell spatial variable.\n");
  }

  dimensionID[0] = m_FrameDimId;
  dimensionID[1] = m_AtomDimId;
  dimensionID[2] = m_SpatialDimId;
  status = nc_def_var(m_NcId, "coordinates", NC_FLOAT, 3, dimensionID,
                      &m_CoordVariableId);
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't define coordinates variable.\n");

  status = nc_put_att_text(m_NcId, m_CoordVariableId, "units", 8, "angstrom");
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't put coordinates variable unit attributes.\n");
  }

  dimensionID[0] = m_CellAngularDimId;
  dimensionID[1] = m_LabelDimId;
  status = nc_def_var(m_NcId, "cell_angular", NC_CHAR, 2, dimensionID,
                      &m_CellAngularVariableId);
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't define cell angular variable.\n");
  }

  std::string title = "NetCDF title string";
  status =
      nc_put_att_text(m_NcId, NC_GLOBAL, "title", title.size(), title.c_str());
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't write title.\n");

  status = nc_put_att_text(m_NcId, NC_GLOBAL, "application", 6, "CHARMM");
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't write application.\n");

  status = nc_put_att_text(m_NcId, NC_GLOBAL, "program", 6, "CHARMM");
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't write program.\n");

  std::string programVersion = "0.1";
  status = nc_put_att_text(m_NcId, NC_GLOBAL, "programVersion",
                           programVersion.size(), programVersion.c_str());
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't write programVersion.\n");

  std::string conventions = "AMBER";
  status = nc_put_att_text(m_NcId, NC_GLOBAL, "Conventions", conventions.size(),
                           conventions.c_str());
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't write conventions.\n");

  status = nc_put_att_text(m_NcId, NC_GLOBAL, "ConventionVersion", 3, "1.0");
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't write convention version.\n");

  status = nc_enddef(m_NcId);
  if (status != NC_NOERR)
    throw std::invalid_argument("ERROR : Can't end definitions.\n");

  m_Start[0] = 0;
  m_Count[0] = 3;
  char op[3];
  op[0] = 'x';
  op[1] = 'y';
  op[2] = 'z';
  status = nc_put_vara_text(m_NcId, m_SpatialVariableId, m_Start.data(),
                            m_Count.data(), op);
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't write to spatial variable id.\n");
  }

  m_Start[0] = 0;
  m_Count[0] = 3;
  op[0] = 'a';
  op[1] = 'b';
  op[2] = 'c';
  status = nc_put_vara_text(m_NcId, m_CellSpatialVariableId, m_Start.data(),
                            m_Count.data(), op);
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't write to abc to cell spatial variable id.\n");
  }

  char opAng[15] = {'a', 'l', 'p', 'h', 'a', 'b', 'e',
                    't', 'a', 'g', 'a', 'm', 'm', 'a'};
  m_Start[0] = 0;
  m_Start[1] = 0;
  m_Count[0] = 3;
  m_Count[1] = 5;
  status = nc_put_vara_text(m_NcId, m_CellAngularVariableId, m_Start.data(),
                            m_Count.data(), opAng);
  if (status != NC_NOERR) {
    throw std::invalid_argument(
        "ERROR : Can't write to abc to cell angular variable id.\n");
  }

  return;
}

void NetCDFSubscriber::formatDataForInput(const std::vector<float4> &xyzq) {
  for (int i = 0; i < m_CharmmContext->getNumAtoms(); i++) {
    m_XYZ[3 * i + 0] = xyzq[i].x;
    m_XYZ[3 * i + 1] = xyzq[i].y;
    m_XYZ[3 * i + 2] = xyzq[i].z;
  }
  return;
}
