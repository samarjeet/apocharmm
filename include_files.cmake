#include_directories(include/Random123)

list(APPEND include_files
  include/CudaSimulationContext.h
  include/CudaContainer.h
  include/Bonded_struct.h
  include/CudaBondedForce.h
  include/CudaPMEDirectForce.h
  include/CudaDirectForceKernels.h
  include/gpu_utils.h
  include/test_utils.cuh
  include/random_utils.h
  include/PrintEnergiesGraph.h
  include/VolumePiston.h
  include/CudaIntegratorGraph.h
  include/SimpleLeapfrogGraph.h
  include/DeviceVector.h
  include/tinyxml2.h
)
