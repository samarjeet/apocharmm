import numpy as np
import pdb
from typing import Optional


class Atom:
    """
    Atom class

    Represents an atom, contains all infos usually found in a PSF file
    """
    sequenceNumber: int
    segName: str
    residueNumber: int
    residueName: str
    atomName: str
    atomType: str
    charge: float
    mass: float
    x: float
    y: float
    z: float

    def __str__(self):
        str = f"Atom: #{self.sequenceNumber} "
        str += f"segName: {self.segName} "
        str += f"residueNumber: {self.residueNumber} "
        str += f"residueName: {self.residueName} "
        str += f"atomName: {self.atomName} "
        str += f"atomType: {self.atomType} "
        str += f"charge: {self.charge} "
        str += f"mass: {self.mass} "
        str += f"coords: ({self.x}, {self.y}, {self.z}))"
        return str

    def getResidueName(self):
        return self.residueName

    def getAtomName(self):
        return self.atomName


class PSF:
    """
    PSF input/output/modification

    Represents the contents of a PSF file, parsed into lists of Atom objects, bonds, angles, etc.
    Each bond is represented as a list of two atom indices (angle as three, etc.)

    When dealing with alchemical transformation, creates new atom types for each atom type present in the alchemical region with suffix "_A".
    """

    def __init__(self, fileName: None) -> None:
        """"
        Constructor. If given an input file name, opens and parses it.
        """
        self.numAtoms = 0
        self.atoms = []  # list of Atom objects
        self.urey_bratomleys = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.alchemicalRegion = []

        if fileName is not None:
            self.fileName = fileName
            self.read()

    def read(self):
        f = open(self.fileName, "r")
        currentSection = None

        for line in f:
            line = line.strip()
            # print(line)
            if not line:
                continue

            parts = line.split()

            if "!NATOM" in line:
                currentSection = "atoms"
                self.numAtoms = int(parts[0])
                print(self.numAtoms)
                continue
            elif "!NBOND" in line:
                currentSection = "bonds"
                continue
            elif "!NTHETA" in line:
                currentSection = "angles"
                continue
            elif "!NPHI" in line:
                currentSection = "dihedrals"
                continue
            elif "!NIMPHI" in line:
                currentSection = "impropers"
                continue
            elif "!N" in line:  # type of section not handled
                currentSection = None
                continue

            if currentSection == "atoms":
                atom = Atom()
                atom.sequenceNumber = int(parts[0])
                atom.segName = parts[1]
                atom.residueNumber = int(parts[2])
                atom.residueName = parts[3]
                atom.atomName = parts[4]
                atom.atomType = parts[5]
                atom.charge = float(parts[6])
                atom.mass = float(parts[7])
                self.atoms.append(atom)
                # print(atom)

            elif currentSection == "bonds":
                # print(parts)
                for i in range(0, len(parts), 2):
                    # print(parts[i], parts[i+1])
                    self.bonds.append([int(parts[i]), int(parts[i+1])])

            elif currentSection == "angles":
                for i in range(0, len(parts), 3):
                    self.angles.append(
                        [int(parts[i]), int(parts[i+1]), int(parts[i+2])])
            elif currentSection == "dihedrals":
                for i in range(0, len(parts), 4):
                    self.dihedrals.append([int(parts[i]), int(parts[i+1]),
                                           int(parts[i+2]), int(parts[i+3])])
            elif currentSection == "impropers":
                for i in range(0, len(parts), 4):
                    self.impropers.append([int(parts[i]), int(parts[i+1]),
                                           int(parts[i+2]), int(parts[i+3])])

        f.close()

    def __str__(self):
        str = f"Number of atoms : {len(self.atoms)}"
        str += f"\nNumber of bonds : {len(self.bonds)}"
        str += f"\nNumber of angles : {len(self.angles)}"
        str += f"\nNumber of dihedrals : {len(self.dihedrals)}"
        str += f"\nNumber of impropers : {len(self.impropers)}"

        return str

    def setAlchemicalRegion(self, alchemicalRegion: None):
        """
        Sets the wanted alchemical region for the PSF (to be modified). The alchemical region is a list of atom indices, STARTING WITH 0 !
        """
        self.alchemicalRegion = alchemicalRegion

    def scaleElectrostatics(self, scale: float, alchemicalRegion=[]):
        """
        Scale the electrostatics of the atoms in the alchemical region by a given factor
        """
        for i in alchemicalRegion:
            self.atoms[i].charge *= scale

    def write(self, outputFileName: str):
        """
        Writes the PSF to a file
        """

        with open(outputFileName, "w") as f:
            f.write("PSF EXT CMAP CHEQ XPLOR\n")
            f.write("\n")
            f.write(f"{1:>10} !NTITLE\n")
            f.write(f"* GENERATED BY APOCHARMMs UTILS\n")
            atomLines = self.writeAtomsSection()
            f.writelines(atomLines)
            f.write(f"\n{len(self.bonds):>10} !NBOND: bonds\n")
            count = 0
            for i in range(0, len(self.bonds)):
                f.write(f"{self.bonds[i][0]:>10}{self.bonds[i][1]:>10}")
                count += 1
                if count == 4 or i == len(self.bonds) - 1:
                    f.write("\n")
                    count = 0

            f.write(f"\n{len(self.angles):>10} !NTHETA: angles\n")
            for i in range(len(self.angles)):
                count += 1
                f.write(
                    f"{self.angles[i][0]:>10}{self.angles[i][1]:>10}{self.angles[i][2]:>10}")
                if count == 3 or i == len(self.angles) - 1:
                    f.write("\n")
                    count = 0

            f.write(f"\n{len(self.dihedrals):>10} !NPHI: dihedrals\n")
            for i in range(len(self.dihedrals)):
                count += 1
                f.write(
                    f"{self.dihedrals[i][0]:>10}{self.dihedrals[i][1]:>10}{self.dihedrals[i][2]:>10}{self.dihedrals[i][3]:>10}")
                if count == 2 or i == len(self.dihedrals) - 1:
                    f.write("\n")
                    count = 0

            f.write(f"\n{len(self.impropers):>10} !NIMPHI: impropers\n")
            for i in range(len(self.impropers)):
                count += 1
                f.write(
                    f"{self.impropers[i][0]:>10}{self.impropers[i][1]:>10}{self.impropers[i][2]:>10}{self.impropers[i][3]:>10}")
                if count == 2 or i == len(self.impropers) - 1:
                    f.write("\n")
                    count = 0

    def writeAtomsSection(self):
        lines = []
        lines.append(f"{self.numAtoms} !NATOM\n")
        for atom in self.atoms:
            lines.append(self.formatAtomLine(atom) + "\n")
        return lines

    def formatAtomLine(self, atom: Atom):
        '''
        Given an atom, returns a PSF-formatted atom line
        '''
        l = f"{atom.sequenceNumber:>10}"
        l += f" {atom.segName:<8}"
        l += f" {atom.residueNumber:<8}"
        l += f" {atom.residueName:<8}"
        l += f" {atom.atomName:<8}"
        l += f" {atom.atomType:<7}"
        l += f" {atom.charge:<012.8e}"
        l += f"   {atom.mass:<7f}"
        l += f" 0  0   0 lol "
        return l

    def allHydrogenIndices(self, regionToSearch=[]):
        """
        Returns a list of all hydrogen indices in the PSF
        """
        hydrogenIndices = []
        for i in regionToSearch:
            if self.atoms[i].atomType[0] == "H":
                hydrogenIndices.append(i)
        return hydrogenIndices


class BondKey:
    atom0: str
    atom1: str

    def __init__(self, atom0: str = None, atom1: str = None) -> None:
        if atom0:
            self.atom0 = atom0
        if atom1:
            self.atom1 = atom1

    def __eq__(self, other):
        return self.atom0 == other.atom0 and self.atom1 == other.atom1


class BondParams:
    k: float
    r0: float

    def __init__(self, k: float = None, r0: float = None) -> None:
        if k:
            self.k = k
        if r0:
            self.r0 = r0


class AngleKey:
    atom0: str
    atom1: str
    atom2: str

    def __init__(self, atom0: str = None, atom1: str = None, atom2: str = None) -> None:
        if atom0:
            self.atom0 = atom0
        if atom1:
            self.atom1 = atom1
        if atom2:
            self.atom2 = atom2

    def __eq__(self, other):
        return self.atom0 == other.atom0 and self.atom1 == other.atom1 and self.atom2 == other.atom2


class AngleParams:
    k: float
    theta0: float

    def __init__(self, k: float = None, theta0: float = None) -> None:
        if k:
            self.k = k
        if theta0:
            self.theta0 = theta0


class DihedralKey:
    atom0: str
    atom1: str
    atom2: str
    atom3: str

    def __init__(self, atom0: str = None, atom1: str = None, atom2: str = None, atom3: str = None) -> None:
        if atom0:
            self.atom0 = atom0
        if atom1:
            self.atom1 = atom1
        if atom2:
            self.atom2 = atom2
        if atom3:
            self.atom3 = atom3

    def __eq__(self, other):
        return self.atom0 == other.atom0 and self.atom1 == other.atom1 and self.atom2 == other.atom2 and self.atom3 == other.atom3


class DihedralParams:
    k: float
    n: int
    phi0: float

    def __init__(self, k: float = None, n: int = None, phi0: float = None) -> None:
        if k:
            self.k = k
        if n:
            self.n = n
        if phi0:
            self.phi0 = phi0


class AtomParameters:
    atomType: str
    mass: float
    sigma: float
    epsilon: float

    def __init__(self, atomType: str = None, mass: float = None, sigma: float = None, epsilon: float = None) -> None:
        self.atomType = atomType
        self.mass = mass
        self.sigma = sigma
        self.epsilon = epsilon

    def __str__(self) -> str:
        return f"AtomParameters: type {self.atomType} mass {self.mass} charge {self.charge} sigma {self.sigma} epsilon {self.epsilon}"

    def __eq__(self, other):
        return self.atomType == other.atomType


class PRM:
    """
    prm file reading, parsing and outputting. Designed to only output prm files for given, specific alchemical atom types (not big, general prm files)
    """

    def __init__(self, fileName: str = None) -> None:
        """
        Constructor. If given an input file, opens and parses it
        """

        self.atomParameters = []
        self.bonds = []  # [(bondkey, bondParam), (bondkey2, bondparam2)]
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        if fileName is not None:
            self.fileName = fileName
            self.read()

    def getAtomParameter(self, atomType: str) -> AtomParameters:
        """
        Given an atom type, returns the AtomParameters object associated with it
        """
        for atomParameter in self.atomParameters:
            if atomParameter.atomType == atomType:
                return atomParameter
        print("Atom type not found : ",  atomType)
        raise "Atom type not found in PRM"

    def read(self):
        with open(self.fileName, "r") as f:
            lines = f.readlines()

        currentSection = None
        for line in lines:
            line = line.strip()
            # remove comments in the line begining with !
            line = line.split("!")[0]

            if not line:
                continue
            if line[0] == '*':
                continue

            parts = line.split()
            # print(line)

            if "ATOMS" in line:
                currentSection = "atoms"
                continue
            elif "BONDS" in line:
                currentSection = "bonds"
                continue
            elif "ANGLES" in line:
                currentSection = "angles"
                continue
            elif "DIHEDRALS" in line:
                currentSection = "dihedrals"
                continue
            elif "IMPROPER" in line:
                currentSection = "impropers"
                continue
            elif "NONBONDED" in line or "cutnb" in line:
                currentSection = "nonbonded"
                continue
            elif "CMAP" in line:
                currentSection = "cmap"
                continue
            elif "HBOND" in line:
                currentSection = "hbond"
                continue
            elif "NBFIX" in line:
                currentSection = "nbfix"
                continue
            elif "END" in line:
                currentSection = None
                continue

            if currentSection == "atoms":
                atomParameter = AtomParameters()
                atomParameter.atomType = parts[2]
                atomParameter.mass = float(parts[3])
                self.atomParameters.append(atomParameter)

            elif currentSection == "bonds":
                bondKey = BondKey()
                if parts[0] > parts[1]:
                    parts[1], parts[0] = parts[0], parts[1]
                bondKey.atom0 = parts[0]
                bondKey.atom1 = parts[1]
                bondParam = BondParams()
                bondParam.k = float(parts[2])
                bondParam.r0 = float(parts[3])
                self.bonds.append((bondKey, bondParam))

            elif currentSection == "angles":
                angleKey = AngleKey()
                if parts[0] > parts[2]:
                    parts[2], parts[0] = parts[0], parts[2]
                angleKey.atom0 = parts[0]
                angleKey.atom1 = parts[1]
                angleKey.atom2 = parts[2]
                angleParam = AngleParams()
                angleParam.k = float(parts[3])
                angleParam.theta0 = float(parts[4])
                self.angles.append((angleKey, angleParam))

            elif currentSection == "dihedrals":
                dihedralKey = DihedralKey()
                dihedralKey.atom0 = parts[0]
                dihedralKey.atom1 = parts[1]
                dihedralKey.atom2 = parts[2]
                dihedralKey.atom3 = parts[3]
                dihedralParam = DihedralParams()
                dihedralParam.k = float(parts[4])
                dihedralParam.n = int(parts[5])
                dihedralParam.phi0 = float(parts[6])
                self.dihedrals.append((dihedralKey, dihedralParam))

            elif currentSection == "impropers":
                dihedralKey = DihedralKey()
                dihedralKey.atom0 = parts[0]
                dihedralKey.atom1 = parts[1]
                dihedralKey.atom2 = parts[2]
                dihedralKey.atom3 = parts[3]
                dihedralParam = DihedralParams()
                dihedralParam.k = float(parts[4])
                dihedralParam.n = int(parts[5])
                dihedralParam.phi0 = float(parts[6])
                self.impropers.append((dihedralKey, dihedralParam))

            elif currentSection == "nonbonded":
                currentType = parts[0]
                for (i, atomParameter) in enumerate(self.atomParameters):
                    if atomParameter.atomType == currentType:
                        atomParameter.sigma = float(parts[3])
                        atomParameter.epsilon = float(parts[2])
                        break
                    if i == len(self.atomParameters) - 1:
                        raise Exception(
                            f"PRM: Atom type {currentType} in NONBONDED section not found in ATOMS section")

    def write(self, outputFileName: str):
        with open(outputFileName, "w") as f:
            f.write("ATOMS\n")
            for atomParameter in self.atomParameters:
                f.write(
                    f"MASS    -1     {atomParameter.atomType:<7} {atomParameter.mass:>8f}\n")

            f.write("\n\nBONDS\n")
            for (bondKey, bondParam) in self.bonds:
                f.write(
                    f"{bondKey.atom0:<8}{bondKey.atom1:<8}{bondParam.k:>8f} {bondParam.r0:>8f}\n")

            f.write("\n\nANGLES\n")
            for (angleKey, angleParam) in self.angles:
                f.write(
                    f"{angleKey.atom0:<8}{angleKey.atom1:<8}{angleKey.atom2:<8}{angleParam.k:>8f} {angleParam.theta0:>8f}\n")

            f.write("\n\nDIHEDRALS\n")
            for (dihedralKey, dihedralParam) in self.dihedrals:
                f.write(f"{dihedralKey.atom0:<8}{dihedralKey.atom1:<8}{dihedralKey.atom2:<8}{dihedralKey.atom3:<8}{dihedralParam.k:>8f} {dihedralParam.n:>8d} {dihedralParam.phi0:>8f}\n")

            f.write("\n\nIMPROPERS\n")
            for (dihedralKey, dihedralParam) in self.impropers:
                f.write(f"{dihedralKey.atom0:<8}{dihedralKey.atom1:<8}{dihedralKey.atom2:<8}{dihedralKey.atom3:<8}{dihedralParam.k:>8f} {dihedralParam.n:>8d} {dihedralParam.phi0:>8f}\n")

            f.write("\n\nNONBONDED\n")
            for atomParameter in self.atomParameters:
                f.write(
                    f"{atomParameter.atomType:<8}0.0 {atomParameter.epsilon:>8f} {atomParameter.sigma:>8f}\n")

    def addAtomParameter(self, atomParameter: AtomParameters):
        """
        Adds an AtomParameter (sigma, epsilon, charge...) to the current PRM object
        """
        self.atomParameters.append(atomParameter)

    def append(self, other):
        """
        Appends the content of another PRM object to the current one

        for atomParameter in other.atomParameters:
            if atomParameter not in self.atomParameters:
                self.atomParameters.append(atomParameter)

        """

        for atomParameter in other.atomParameters:
            addAtomParameter = True
            for atomParameter2 in self.atomParameters:
                if atomParameter.atomType == atomParameter2.atomType:
                    addAtomParameter = False
                    break
            if addAtomParameter:
                self.atomParameters.append(atomParameter)

        for (bondKey, bondParam) in other.bonds:
            addBond = True
            for (bondKey2, bondParam2) in self.bonds:
                if bondKey == bondKey2:
                    addBond = False
                    break
            if addBond:
                self.bonds.append((bondKey, bondParam))

        for (angleKey, angleParam) in other.angles:
            addAngle = True
            for (angleKey2, angleParam2) in self.angles:
                if angleKey == angleKey2:
                    addAngle = False
                    break
            if addAngle:
                self.angles.append((angleKey, angleParam))

        for (dihedralKey, dihedralParam) in other.dihedrals:
            addDihedral = True
            for (dihedralKey2, dihedralParam2) in self.dihedrals:
                if dihedralKey == dihedralKey2:
                    addDihedral = False
                    break
            if addDihedral:
                self.dihedrals.append((dihedralKey, dihedralParam))

        for (dihedralKey, dihedralParam) in other.impropers:
            addImproper = True
            for (dihedralKey2, dihedralParam2) in self.impropers:
                if dihedralKey == dihedralKey2:
                    addImproper = False
                    break
            if addImproper:
                self.impropers.append((dihedralKey, dihedralParam))


def extractAlchemicalParameters(prmIn: PRM, alchemicalRegionAtomTypes=[]):
    """
    Given a PRM object and a list of atomtypes (list(str)), returns a new PRM
    object containing only the parameters for these atom types
    """
    newPRM = PRM(None)
    for atomType in prmIn.atomTypes:
        if atomType.atomType in alchemicalRegionAtomTypes:
            newPRM.atomTypes.append(atomType)

    for (bondKey, bondParam) in prmIn.bonds:
        if bondKey.atom0 in alchemicalRegionAtomTypes or bondKey.atom1 in alchemicalRegionAtomTypes:
            newPRM.bonds.append((bondKey, bondParam))

    for (angleKey, angleParam) in prmIn.angles:
        if angleKey.atom0 in alchemicalRegionAtomTypes or angleKey.atom1 in alchemicalRegionAtomTypes or angleKey.atom2 in alchemicalRegionAtomTypes:
            newPRM.angles.append((angleKey, angleParam))

    for (dihedralKey, dihedralParam) in prmIn.dihedrals:
        if dihedralKey.atom0 in alchemicalRegionAtomTypes or dihedralKey.atom1 in alchemicalRegionAtomTypes or dihedralKey.atom2 in alchemicalRegionAtomTypes or dihedralKey.atom3 in alchemicalRegionAtomTypes:
            newPRM.dihedrals.append((dihedralKey, dihedralParam))

    for (dihedralKey, dihedralParam) in prmIn.impropers:
        if dihedralKey.atom0 in alchemicalRegionAtomTypes or dihedralKey.atom1 in alchemicalRegionAtomTypes or dihedralKey.atom2 in alchemicalRegionAtomTypes or dihedralKey.atom3 in alchemicalRegionAtomTypes:
            newPRM.impropers.append((dihedralKey, dihedralParam))

    return newPRM


def renameAtomTypesInPRM(prm: PRM, suffix="_A"):
    """
    Given an input PRM object, renames atom types by adding a suffix "_A" to each atom type
    """
    assert suffix != "", "Suffix cannot be empty"
    for atomType in prm.atomTypes:
        atomType.atomType += suffix
    for (bondKey, bondParam) in prm.bonds:
        bondKey.atom0 += suffix
        bondKey.atom1 += suffix
    for (angleKey, angleParam) in prm.angles:
        angleKey.atom0 += suffix
        angleKey.atom1 += suffix
        angleKey.atom2 += suffix
    for (dihedralKey, dihedralParam) in prm.dihedrals:
        dihedralKey.atom0 += suffix
        dihedralKey.atom1 += suffix
        dihedralKey.atom2 += suffix
        dihedralKey.atom3 += suffix
    for (dihedralKey, dihedralParam) in prm.impropers:
        dihedralKey.atom0 += suffix
        dihedralKey.atom1 += suffix
        dihedralKey.atom2 += suffix
        dihedralKey.atom3 += suffix
    return prm


def getAtomTypeFromIndex(psf: PSF, index: int):
    """
    Given a PSF object and an index i, returns the atomType of the atom at index i
    """
    return psf.atoms[index].atomType


def createNewAtomParameters(atomType: str, referencePrm: PRM, vdwScaling=1.0, newAtomType: str = None):
    """
    Given an atom type (str), and a PRM object containing all the parameters (alch + original), creates a new atom parameters with the given suffix, and returns it
    """
    if not newAtomType:
        raise Exception("newAtomType must be specified")
    print(atomType)
    newAtomParameter = AtomParameters()
    newAtomParameter.atomType = newAtomType
    # in referencePrm find the right AtomParameters object
    referenceAtomParameter = referencePrm.getAtomParameter(atomType)
    # pdb.set_trace()

    newAtomParameter.mass = referenceAtomParameter.mass
    newAtomParameter.sigma = referenceAtomParameter.sigma * vdwScaling
    newAtomParameter.epsilon = referenceAtomParameter.epsilon * vdwScaling
    return newAtomParameter


def scaleOffVdwForSingleAtom(psf: PSF, originalPrm: PRM,  prmToBeUpdated: PRM, atomIndex: int, vdwScaling: float, newAtomType: str):
    """
    Given a PSF, PRM and PDB objects, an atom index, and a scaling factor, 
    returns a PSF and PRM objects modified
    """
    if not newAtomType:
        raise Exception("newAtomType must be specified")
    # 1. Get atom type of the atom
    atomType = getAtomTypeFromIndex(psf, atomIndex)
    # 2. Create new atom type with suffix _A
    newAtomParameter = createNewAtomParameters(
        atomType, originalPrm, vdwScaling, newAtomType)

    workPrm = PRM()
    workPrm.atomParameters.append(newAtomParameter)
    newAtomType = newAtomParameter.atomType

    # 3. Add this atom type to a new prm ?
    # This means adding:
    # - a new atom type
    # - all bonds containing said atom type
    # - all angles containing said atom type
    # (+dih + impdih)
    # This is also true for the prm to be updated itself !

    for bondKey, bondParam in originalPrm.bonds:
        newBondKey = None
        if bondKey.atom0 == atomType:
            newBondKey = BondKey(newAtomType, bondKey.atom1)
        elif bondKey.atom1 == atomType:
            newBondKey = BondKey(bondKey.atom0, newAtomType)
        if newBondKey:
            workPrm.bonds.append((newBondKey, bondParam))

    for (angleKey, angleParam) in originalPrm.angles:
        newAngleKey = None
        if angleKey.atom0 == atomType:
            newAngleKey = AngleKey(newAtomType, angleKey.atom1, angleKey.atom2)
        elif angleKey.atom1 == atomType:
            newAngleKey = AngleKey(angleKey.atom0, newAtomType, angleKey.atom2)
        elif angleKey.atom2 == atomType:
            newAngleKey = AngleKey(angleKey.atom0, angleKey.atom1, newAtomType)
        if newAngleKey:
            workPrm.angles.append((newAngleKey, angleParam))

    for (dihedralKey, dihedralParam) in originalPrm.dihedrals:
        newDihedralKey = None
        if dihedralKey.atom0 == atomType:
            newDihedralKey = DihedralKey(
                newAtomType, dihedralKey.atom1, dihedralKey.atom2, dihedralKey.atom3)
        elif dihedralKey.atom1 == atomType:
            newDihedralKey = DihedralKey(
                dihedralKey.atom0, newAtomType, dihedralKey.atom2, dihedralKey.atom3)
        elif dihedralKey.atom2 == atomType:
            newDihedralKey = DihedralKey(
                dihedralKey.atom0, dihedralKey.atom1, newAtomType, dihedralKey.atom3)
        elif dihedralKey.atom3 == atomType:
            newDihedralKey = DihedralKey(
                dihedralKey.atom0, dihedralKey.atom1, dihedralKey.atom2, newAtomType)
        if newDihedralKey:
            workPrm.dihedrals.append((newDihedralKey, dihedralParam))

    for (dihedralKey, dihedralParam) in originalPrm.impropers:
        newDihedralKey = None
        if dihedralKey.atom0 == atomType:
            newDihedralKey = DihedralKey(
                newAtomType, dihedralKey.atom1, dihedralKey.atom2, dihedralKey.atom3)
        elif dihedralKey.atom1 == atomType:
            newDihedralKey = DihedralKey(
                dihedralKey.atom0, newAtomType, dihedralKey.atom2, dihedralKey.atom3)
        elif dihedralKey.atom2 == atomType:
            newDihedralKey = DihedralKey(
                dihedralKey.atom0, dihedralKey.atom1, newAtomType, dihedralKey.atom3)
        elif dihedralKey.atom3 == newAtomParameter.atomType:
            newDihedralKey = DihedralKey(
                dihedralKey.atom0, dihedralKey.atom1, dihedralKey.atom2, newAtomType)
        if newDihedralKey:
            workPrm.impropers.append((newDihedralKey, dihedralParam))

    prmToBeUpdated.append(workPrm)

    # 4. Change the atom type of the atom in the psf (and pdb?)
    psf.atoms[atomIndex].atomType = newAtomParameter.atomType
    return psf, prmToBeUpdated


class CRD:
    """
    crd file reading, parsing and outputting. 
    """

    def __init__(self, fileName: str = None) -> None:
        """
        Constructor. Given an input file, opens and parses it.
        """
        self.numAtoms = 0
        self.atoms = []

        if fileName is not None:
            self.fileName = fileName
            self.read()

    def read(self):
        with open(self.fileName, "r") as f:
            lines = f.readlines()

        for l in lines:
            if l[0] == "*":
                continue  # title line
            else:
                l = l.strip()
                parts = l.split()
                if len(parts) == 0:
                    continue   # empty line

            if len(parts) <= 6:  # This 6 is a bit arbitrary
                self.numAtoms = int(parts[0])
                continue

            atom = Atom()
            atom.atomNumber = int(parts[0])
            atom.residueNumber = int(parts[1])
            atom.residueName = parts[2]
            atom.atomName = parts[3]
            atom.x = float(parts[4])
            atom.y = float(parts[5])
            atom.z = float(parts[6])
            atom.segName = parts[7]

            self.atoms.append(atom)

    def getNumAtoms(self):
        if not self.numAtoms:
            raise Exception("CRD: numAtoms not set")
        return self.numAtoms

    def getAtoms(self):
        return self.atoms


#######################################################
# Master pseudocode
##########################################################


def testVdwScalingOffForSingleAtom(psf: PSF, atomIndex: int, vdwScaling: float):
    alchPrm = PRM()
    psf, alchPrm = scaleOffVdwForSingleAtom(
        psf, alchPrm, atomIndex, vdwScaling)
    alchPrm.write("testPrmSingleAtom.prm")
    psf.write("testPsfSingleAtom.psf")


def prepareSAIAnnihilationInput(systemPsf: PSF, systemPrm: PRM, alchemicalRegion=[]):
    """
    Given a system defined by a PSF and associated PRM objects, prepares all the
    inputs needed for SAI annihilation.
    Generates input for...
    1. Scaling off ALL electrostatics (psf files)
    2. Scaling off vdw for ALL hydrogen atoms (psf and prm files)
    3. Scaling off vdw for all heavy atoms, in a previously defined order (psf
    and prm files)
    """
    alchemicalPsf = systemPsf.deepcopy()
    alchemicalPrm = PRM()
    # Parameters !
    numberOfElectrostaticsScalingSteps = 4
    # Step 1: scale off ALL electrostatics
    for i in range(numberOfElectrostaticsScalingSteps):
        lambdaElec = 1 - (i+1)*(1./numberOfElectrostaticsScalingSteps)
        alchemicalPsf.scaleElectrostatics(lambdaElec)
        alchemicalPsfOutputFileName = f"step1_{i}.psf"
        alchemicalPsf.print(alchemicalPsfOutputFileName)

    # Step 2 : scale off ALL vdw for hydrogen atoms
    # Actually, as a development step, scale off vdw for a single given atom
    atomIndex = 0
    alchemicalPsf, alchemicalPrm = scaleOffVdwForSingleAtom(
        alchemicalPsf, alchemicalPrm,  atomIndex, 0.5)
    alchemicalPsfOutputFileName = f"step2_{atomIndex}.psf"
    alchemicalPrmOutputFileName = f"step2_{atomIndex}.prm"
    alchemicalPsf.print(alchemicalPsfOutputFileName)
    alchemicalPrm.print(alchemicalPrmOutputFileName)

# Generic procedure  for scaling off vdw for a given atom index :
# 1. Get atom type of the atom
# 2. Create new atom type with suffix _A
# 3. Add this atom type to the prm
# 5. Change the atom type of the atom in the psf (and pdb?)
# 5. Print out the new prm, psf (and pdb ?)
