import apocharmm as ac


# arrange 
# act 
# assert
# clean


waterions = '../data/toppar_water_ions.str'
cgenff = '../data/par_all36_cgenff.prm'
prot   = '../data/par_all36_prot.prm'

def test_init():
   prm_waterions = ac.CharmmParameters(waterions)
   prm_cgenff = ac.CharmmParameters(cgenff)
   prm_watprot = ac.CharmmParameters([waterions, prot])

   print(prm_watprot)

   assert type(prm_watprot) == ac.CharmmParameters
