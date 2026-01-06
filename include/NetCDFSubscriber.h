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
#include <array>
#include <netcdf.h>
#include <vector>

class NetCDFSubscriber : public Subscriber {
public:
  NetCDFSubscriber(const std::string &fileName);
  NetCDFSubscriber(const std::string &fileName, int reportFrequency);
  ~NetCDFSubscriber(void);

public:
  void update(void) override;

private:
  void initialize(void);
  void formatDataForInput(const std::vector<float4> &xyzq);

private:
  int m_NcId;
  int m_FrameDimId;
  int m_SpatialDimId;
  int m_AtomDimId;
  int m_CellSpatialDimId;
  int m_CellAngularDimId;
  int m_LabelDimId;
  int m_TimeVariableId;
  int m_SpatialVariableId;
  int m_CellLengthsVariableId;
  int m_CellAnglesVariableId;
  int m_CoordVariableId;
  int m_CellSpatialVariableId;
  int m_CellAngularVariableId;
  std::vector<float> m_XYZ;
  std::array<std::size_t, 3> m_Start;
  std::array<std::size_t, 3> m_Count;
  int m_NumFramesWritten;
};
