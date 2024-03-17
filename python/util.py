import apocharmm as ac
def packageName():
  return "CHARMM"

def TotalMechanicalEnergy(ctx):
   '''
   Adds kuinetic and potential energy.
   Might be Total, might be Mechanical.
   '''

   epot = ctx.calculatePotentialEnergy(True, True)
   ekin = ctx.getKineticEnergy()
   return epot+ekin





def TotalMechanicalEnergy_ContextBound():
   '''
   This should be a member of the Context class
   '''
   pass


if __name__=='__main__':

   print('hi')
   pdb = ac.PDB('../test/data/step3_pbcsetup.pdb')
   psf = ac.CharmmPSF('../test/data/step3_pbcsetup.psf')
   prm = ac.CharmmParameters('../test/data/toppar_water_ions.str')
   fm = ac.ForceManager(psf, prm)
   fm.setFFTGrid(18,18,18)
   fm.setBoxDimensions([20,20,20])
   fm.initialize()
   ctx = ac.CharmmContext(fm)
   ctx.setCoordinates(pdb)
   tot = TotalMechanicalEnergy(ctx)
   print(tot)
