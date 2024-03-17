import requests


def name():
    print("CHARMM")


pdb_base_url = "https://files.rcsb.org/download/"


class Entry:
    """
    An Entry can be an ATOM or a HETATM
    stores all the information related to the atom
    """

    def __init__(self, entryType, atomSerialNumber, atomName, residueName, chainIdentifier, residueSequenceNumber,
                 x, y, z, segmentIdentifier, elementSymbol):
        self.atomSerialNumber = atomSerialNumber
        self.atomName = atomName
        self.residueName = residueName
        self.chainIdentifier = chainIdentifier
        self.residueSequenceNumber = residueSequenceNumber
        self.x = x
        self.y = y
        self.z = z
        self.segmentIdentifier = segmentIdentifier

    def __str__(self):
        return str(self.atomSerialNumber) + " " + self.atomName


def fromPDB(pdbId: str):
    url = pdb_base_url + pdbId + ".pdb"
    try:
        fileHandle = open(pdbId+".pdb")
        pdb_text = fileHandle.readlines()
    except:
        print("Downloading pdb...")
        response = requests.get(url)
        pdb_text = response.text

    chains = []
    entries = []
    for line in pdb_text:
        line = line.strip("\n")
        recordName = line[:6]
        if recordName == "ATOM  " or recordName == "HETATM":
            atomSerialNumber = int(line[6:11])
            atomName = line[12:16]
            alternateLocationIndicator = line[16]
            residueName = line[17:20]
            chainIdentifier = line[21]
            residueSequenceNumber = int(line[22:26])
            codeForInsertionOfResidues = line[26]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            occupancy = float(line[54:60])
            temperatureFactor = float(line[60:66])
            segmentIdentifier = line[72:76].strip(' ')
            elementSymbol = line[76:78].strip(' ')
            atomCharge = line[78:80]

            if chainIdentifier not in chains:
                chains.append(chainIdentifier)
            # print(line, atomSerialNumber, atomName, residueName, residueSequenceNumber,
            #      x, y, z, segmentIdentifier, elementSymbol)
            entry = Entry(recordName, atomSerialNumber, atomName, residueName, chainIdentifier, residueSequenceNumber,
                          x, y, z, segmentIdentifier, elementSymbol)
            entries.append(entry)
    print(chains)
    chosenChains = ['K', 'A', 'H']

    chainA = [entry for entry in entries if entry.chainIdentifier in chosenChains]
    # for entry in chainA:
    #    print(entry)


def fromCIF(pdbId: str):
    pdbId = pdbId.upper()

    url = pdb_base_url + pdbId + ".cif"
    try:
        fileHandle = open(pdbId+".cif")
        cif_text = fileHandle.readlines()
    except:
        print("Downloading cif...")
        response = requests.get(url)
        with open(pdbId+".cif", 'w') as f:
            f.write(response.text)
        cif_text = response.text
    
    categoryState = ""
    atom_site_attributes = []
    struct_asym_attributes = []
    lineNumber: int = 0
    
    while lineNumber < len(cif_text):
      line = cif_text[lineNumber]
      if line[0] == '#':
        categoryState = ""
      line = cif_text[lineNumber]
      if "loop" in line:
        lineNumber += 1
        line = cif_text[lineNumber]
        if "atom_site." in line :
          categoryState = "atom_site"
        if "_struct_asym." in line :
          categoryState = "struct_asym"


      if categoryState == "atom_site":
        if "atom_site" in line :
          field = line[11:-2]
          atom_site_attributes.append(field)
          print(field)
        else :
          parts = line.split()
          group_PDB_index = atom_site_attributes.index('group_PDB')
          type_symbol_index = atom_site_attributes.index('type_symbol')
          atom_id_index = atom_site_attributes.index('label_atom_id')
          residue_id_index = atom_site_attributes.index('label_comp_id')
          cartn_x_index = atom_site_attributes.index("Cartn_x")
          cartn_y_index = atom_site_attributes.index("Cartn_y")
          cartn_z_index = atom_site_attributes.index("Cartn_z")    

          group_PDB = parts[group_PDB_index]
          type_symbol = parts[type_symbol_index]
          atom_id = parts[atom_id_index]
          residue_id = parts[residue_id_index]
          x = parts[cartn_x_index]
          y = parts[cartn_y_index]
          z = parts[cartn_z_index]
          print(group_PDB, type_symbol, atom_id, residue_id, x, y, z)

      if categoryState == 'struct_asym':
        if "_struct_asym" in line :
          field = line[13:-2]
          struct_asym_attributes.append(field)
        else :
          parts = line.strip('\n').split()
          id_index = struct_asym_attributes.index('id')
          entity_id_index = struct_asym_attributes.index('entity_id')
          
          struct_asym_id = parts[id_index]
          entity_asym_id = parts[entity_id_index]
          print(struct_asym_id, entity_asym_id)


      lineNumber += 1


if __name__ == '__main__':
    # fromPDB("7z2c")
    fromCIF("7z2c")
