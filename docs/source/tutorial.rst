
======== 
Tutorial
========

.. note::

   The code snippets in this tutorial are available as a python script and a python notebook in the `example/tutorial/` directory. 


Waterbox dynamic example
------------------------

In this section, we describe a basic example of a standard MD simulation on a 50 Angstroms cubic water system.  The input files (``waterbox.crd``, ``waterbox.psf`` and ``toppar_water_ions.str``) can be found in the ``test/data/`` directory.


Assuming we have imported the apocharmm module as ``ac`` (see following code block), let us first import our force field parameters (``.str`` and ``.prm`` files) and system's topology (``.psf``) using the :py:class:`charmm.apocharmm.CharmmParameters` and :py:class:`charmm.apocharmm.CharmmPSF` classes.

.. code-block:: python

   from charmm import apocharmm as ac

   prm = ac.CharmmParameters('data/toppar_water_ions.str')
   psf = ac.CharmmPSF('data/waterbox.psf')
   
Note that CharmmParameters can be initialized with a file name or a list of file names.


We will then prepare a :py:class:`charmm.apocharmm.ForceManager` object using the CharmmParameters and CharmmPSF objects. The ForceManager bundles the system's topology and the force field parameters, in order to handle the forces.  Non-bonded force parameters (cutoff distances, FFT options, PBC) are thus also given to the ForceManager, along with the box size (:py:func:`charmm.apocharmm.ForceManager.setBoxDimensions`).

.. code-block:: python

   fm = ac.ForceManager(psf, prm)
   fm.setBoxDimensions([50., 50., 50.])

   # Default values
   fm.setFFTGrid(48,48,48)     #[default]
   fm.setKappa(0.34)           #[default]
   fm.setPmeSplineOrder(4)     #[default]
   fm.setCutoff(12.0)          #[default]
   fm.setCtonnb(9.0)           #[default]
   fm.setCtofnb(10.0)           #[default]

   # Finish with initialization (creation of all force terms by the ForceManager)
   fm.initialize()

Note that the above lines containing ``#[default]`` are not explicitly needed.


Apocharmm follows a mediator design pattern, where a "mediator" class -- the :py:class:`charmm.apocharmm.CharmmContext` object -- interacts with satellite classes and functions. We will thus create a CharmmContext object
from our ForceManager.

.. code-block:: python 

   ctx = ac.CharmmContext(fm)

Setting up the systems configuration can be done from a CHARMM ``.crd`` file (see :py:class:`charmm.apocharmm.CharmmCrd`). 

.. code-block:: python

   crd = CharmmCrd('data/waterbox.crd')
   ctx.setCoordinates(crd)

A ``pdb`` file can also be used for that purpose, via
``pdb=PDB('data/waterbox.pdb')`` (see :py:class:`charmm.apocharmm.PDB`). 

Initializing the velocities is also handled by the CharmmContext object, through

.. code-block:: python

   ctx.assignVelocitiesAtTemperature(300.0)


Alternatively, one may also use a CHARMM-like restart file to setup both the coordinates and velocities, via
.. code-block:: python

   ctx.readRestart("waterboxRestart.res")


Minimization is done using a :py:class:`charmm.apocharmm.Minimizer` object, linked to our :py:class:`charmm.apocharmm.CharmmContext`.

.. code-block:: python

   mini = ac.minimizer()
   mini.setSimulationContext(ctx)
   mini.minimize(100) # number of minimization steps


We can then initialize a velocity-Verlet integrator (:py:class:`charmm.apocharmm.VelocityVerletIntegrator`), then link it to our mediator CharmmContext as follow:

.. code-block:: python 
   
   integrator = CudaVelocityVerletIntegrator(0.001) # 0.001=time step duration in ps
   integrator.setSimulationContext(ctx) # Linking the integrator and the context.


Apocharmm can run several different integrators, including Langevin thermostat, Langevin piston and Nose-Hoover (see :py:class:`charmm.apocharmm.LangevinThermostatIntegrator`, :py:class:`charmm.apocharmm.LangevinPistonIntegrator`, :py:class:`charmm.apocharmm.NoseHooverThermostatIntegrator` ). 

.. code-block:: python

   integrator.propagate(1000)



Monitoring the simulation
"""""""""""""""""""""""""

:py:class:`charmm.apocharmm.Subscriber` objects produce reports to monitor a simulation. These include 

*   StateSubscribers, reporting the Potential and Kinetic energies and total simulation time (:py:class:`charmm.apocharmm.StateSubscriber`).
*   DcdSubscribers, generating a CHARMM DCD file trajectory (:py:class:`charmm.apocharmm.DcdSubscriber`).
*   NetCDFSubscribers (:py:class:`charmm.apocharmm.NetCDFSubscriber`).
*   MBARSubscribers (:py:class:`charmm.apocharmm.MBARSubscriber`).

To create a subscriber, you need a filename (to which reports will be printed out periodically) and a frequency (number of steps between two printouts). Subscribers are attached to an Integrator (:py:class:`charmm.apocharmm.Integrator`) via the ``subscribe`` function.

In the following block, we create a StateSubscriber called ``sub``, who will print information to the file ``filename.txt`` every ``1000`` steps. 

.. code-block:: python
   
   sub = ac.StateSubscriber('filename.txt', 1000)
   integrator.subscribe(sub)
   





Sphinx cheatsheet (please disregard)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bits of ``code`` in double quotes

*Emphasize between asterisks*

Code blocks:

.. code-block:: python
   
   print("Hello world")


.. code-block:: cpp

   std::cout << "Hello World !" << std::endl

Unformatted things: ::
   
   My unformatted text

Maybe a list 

#.   indent
#.   the 

list.



