APOCHARMM 
=========

Quick links to often used classes/references ?
CharmmContext / ForceManager / Subscriber / CudaIntegrator


## Architecture ideas ##

### User-level features  ###

ForceManager(CharmmPSF, CharmmParameters) linked to CharmmContext.

CharmmContext contains conformation (Coordinates/CharmmCrd), velocities.

Integrator is linked to CharmmContext.

Subscribers are linked to Integrator.


\htmlonly<embed src="ApocharmmSketch.pdf" width="100%" height="100%" \
href="ApocharmmSketch.pdf"></embed> \endhtmlonly

<a href="ApocharmmSketch.pdf" target="_blank"><b>Sketch</b></a>

