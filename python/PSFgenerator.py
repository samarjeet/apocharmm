# Generates PSF files by scaling charges of the alchemical region.
# Inp : psf file name, lambda value, line range corresponding to the alchemical
# region indices
import argparse as ap

## Line formatter
# 1. Gets charge
# 2. Divides by two
# 3. float-> string with proper format
# 4. Reinserts in line (without splitting to simplify formatting?)
def formatLine(l, lambdaval):
   charge_string = l.split()[6]
   charge = float(charge_string)
   newcharge = charge * lambdaval
   newcharge_string = f"{newcharge:>7.6f}"
   newl = l.replace(charge_string, newcharge_string)
   return newl

## Output file name generator
# Adds lambda value before the ".psf" extension
def generatePSFName(inputfile, lambdaval):
   lambdastring = "l" + f"{lambdaval}"
   newextension = "." + lambdastring + ".psf"
   newPSFname = inputfile.replace(".psf", newextension)
   return newPSFname



parser = ap.ArgumentParser()
parser.add_argument("PSF", help="PSF file to be copied", type=str)
parser.add_argument("lambdaval", help="Value of lambda (scaling charges)", type=float)
parser.add_argument("range", help='Range of atom indices corresponding to the alchemical region, in quotes. Indexation starts at 1. Ex: "1 5"', type=str)

args = parser.parse_args()

# Get file content 
with open(args.PSF, 'r') as psf:
   content = psf.readlines()
# Find first line of atom lines
for (i,l) in enumerate(content):
   if "!NATOM" in l :
      n = i
      break 
   if i == len(content):
      exit("COULD NOT FIND !NATOM IN PSF FILE\nStopping.")

# Extract alchemical region range
tmp = args.range.split()
alch_region_first, alch_region_last = int(tmp[0]), int(tmp[1]) 
ifirst = alch_region_first+n
ilast = alch_region_last+n

# Scale with lambda
for i in range(ifirst, ilast+1):
   content[i] = formatLine(content[i], args.lambdaval)

# New file name
newpsf = generatePSFName(args.PSF, args.lambdaval)
with open(newpsf, 'w+') as f:
   f.writelines(content)












