// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include "Subscriber.h"
#include <netcdf.h>

class NetCDFSubscriber : public Subscriber {
public:
  NetCDFSubscriber(const std::string &fileName);
  NetCDFSubscriber(const std::string &fileName, int reportFreq);
  void update() override;
  ~NetCDFSubscriber();

private:
  void initialize();
  int ncid;
  int frameDimId;
  int spatialDimId;
  int atomDimId, cellSpatialDimId, cellAngularDimId, labelDimId;
  int timeVariableId, spatialVariableId, cellLengthsVariableId,
      cellAnglesVariableId, coordVariableId, cellSpatialVariableId,
      cellAngularVariableId;
  void formatDataForInput(const std::vector<float4> &xyzq);
  float *xyzNC;
  size_t start[3];
  size_t count[3];

  int numFramesWritten;
};
