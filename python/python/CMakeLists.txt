pybind11_add_module(apocharmm pywrapper_main.cu) 

set_property(TARGET apocharmm PROPERTY CUDA_ARCHITECTURES 60 70)
target_include_directories(apocharmm PUBLIC ../include)

target_link_libraries(apocharmm PUBLIC chcudadynlib )

target_sources(apocharmm PRIVATE 
  exportCharmmPSF.cu
  exportCharmmContext.cu
  exportCharmmParameters.cu
  exportCoordinates.cu
  exportForceManager.cu
  exportIntegrator.cu
  exportSubscriber.cu
  exportForceManagerGenerator.cu
  exportPBC.cu
)

