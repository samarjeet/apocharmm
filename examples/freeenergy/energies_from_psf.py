# Simlulation of a simple CO2 in water
# Objective : use with hand-modified PSF (scaled charges) to output energies.
# (Later to be used by FEP/MBAR to get free energies)

from charmm import apocharmm as ac
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument("psffile", type=str, 
                  help="Name of psf file to use")

args = parser.parse_args()

#HARD CODED PARAMETERS
nsteps

prm = ac.CharmmParameters(["../../test/data/toppar_water_ions.str", 
                           "../../test/data/par_all36_cgenff.prm"])
psf = ac.CharmmPSF(args.psffile)

fm = ac.ForceManager(
