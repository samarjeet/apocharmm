# Script to generate SAI parameters for the absolute solvation free energy via annihilation.
#
import charmm as ch
from charmm.experimental import *
import copy
import os


# Folder with original psf/crd/prm files
dataFolder = "../test/data"
prmFolder = "../test/data"
# phase ("solv" or "vacuum")
#phase = "vacuum"
phase = "solv"
# Main output folder
outputFolder = "sai_params"
if phase=="vacuum": outputFolder += "_vac"

# Prepare folder
if not os.path.isdir(outputFolder):
    os.mkdir(outputFolder)


# INPUTS
molid = 9979854
originalPsf = PSF(
    f'{dataFolder}/{molid}.{phase}.psf')

originalPrm = PRM(
    f'{dataFolder}/mobley_{molid}.str')

cgenffPrm = PRM(f'{dataFolder}/par_all36_cgenff.prm')
originalPrm.append(cgenffPrm)
# This should be detected differently maybe ?
alchemicalRegion = [i for i in range(12)]
electrostaticDeactivationSchedule = [1.0, 0.6, 0.3, 0.0]
# PRM file to be augmented with all prms (not to be printed, simply to keep track)
allPrm = copy.copy(originalPrm)


# SCALING DOWN ELECTROSTATICS
stepCount = 0
for electrostaticScalingValue in electrostaticDeactivationSchedule:
    newpsf = copy.copy(originalPsf)
    newpsf.scaleElectrostatics(electrostaticScalingValue, alchemicalRegion)

    intFolderName = f"{outputFolder}/intst{stepCount}"
    if not os.path.isdir(intFolderName):
        os.mkdir(intFolderName)

    newpsf.write(f"{intFolderName}/ligand.psf")
    stepCount += 1

# SCALING DOWN VDW FOR HYDROGENS
hydrogenIndices = originalPsf.allHydrogenIndices(alchemicalRegion)
print(hydrogenIndices)
for vdwScalingValue in [0.5, 0.0]:
    newAtomType = "DDH"
    alchemicalPrm = PRM()
    for i in range(len(hydrogenIndices)):
        newpsf, _ = scaleOffVdwForSingleAtom(
            newpsf, allPrm, alchemicalPrm, hydrogenIndices[i], vdwScalingValue, newAtomType)
        allPrm.append(alchemicalPrm)

    intFolderName = f"{outputFolder}/intst{stepCount}"
    if not os.path.isdir(intFolderName):
        os.mkdir(intFolderName)

    alchemicalPrm.write(f"{intFolderName}/dummy_parameters.prm")
    newpsf.write(f"{intFolderName}/ligand.psf")
    stepCount += 1


# SCALING DOWN VDW FOR EACH HEAVY ATOM, ONE BY ONE
# We will need a heuristic to choose this
deactivationSequence = [i for i in range(7)]
for i in deactivationSequence:
    newAtomType = "DD" + str(11 + i)
    newpsf, _ = scaleOffVdwForSingleAtom(newpsf, allPrm, alchemicalPrm,
                                         deactivationSequence[i], 0.0, newAtomType)
    allPrm.append(alchemicalPrm)

    intFolderName = f"{outputFolder}/intst{stepCount}"
    if not os.path.isdir(intFolderName):
        os.mkdir(intFolderName)

    alchemicalPrm.write(f"{intFolderName}/dummy_parameters.prm")
    newpsf.write(f"{intFolderName}/ligand.psf")
    stepCount += 1


# Basic idea: regardless of the alchemical path chosen, generate an entire new
# set of parameters for the entire alchemical region.
# For SAI, we'll need different atom types for each multiple occurrence of each
# atom type. Ex: if there are 3 atom of type CA, we'll need CA_al_1, CA_al_2,
# CA_al_3.
# For single topology (turn off vdw+elec for the whole region), we'll need one
# param type per unique atom type in the alchemical region. Ex: if there are 3
# atom of type CA, we'll need CA_al.
#
# For each atom type in the alchemical region, we will thus need to extract its
# associated parameters: atom type, charge, sigma (vdw), epsilon (vdw), r0/k0
# with any atom it might bind to, etc.
# We will then produce PSF, PDB and PRM files for each alchemical state.
#

# 1. Choose alchemical region
# 2. Identify unique atom types in the alchemical region
# 3. Extract all parameters associated with each unique atom type
# 4. Design the thermodynamic cycle wanted (based on SAI or single topo)
#    a. If single topology, for the alchemical region,
#           - deactivate all electrostatics
#           - deactivate all vdw
#    b. If SAI,
#           - deactivate all electrostatics
#           - deactivate vdw of all H atoms
#           - choose an order in which to deactivate heavy atoms' vdw
#           - deactivate vdw of heavy atoms in that order

# 5. Create alchemical atom types needed based on the alchemical path

# After analyzing Stefan's earlier methane to toluene inputs, we'll actually
# only create new atom  types as needed, to avoid inflating the number of
# params/the size of prm file.   While this is not a performance concern, it
# loses the readability we were aiming for.
