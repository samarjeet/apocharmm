// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Nathan Zimmerberg
//
// ENDLICENSE

/** \file
 * \author Nathan Zimmerberg (nhz2@cornell.edu)
 * \date
 * \brief Utility structs and inline functions for NPT.
 */
#pragma once
/** Type to store an array of atom ids*/
typedef struct AtomIdLists {
  int numAtoms; /**< Number of atom ids in the array. */
  int idarray;  /**< offset to first id in the array of atom ids. */
} AtomIdList_t;

/** Type to store the center of mass of a group, and its atom id list*/
typedef struct ComIDs {
  double x;
  double y;
  double z;
  AtomIdList_t ids;
} ComID_t;
