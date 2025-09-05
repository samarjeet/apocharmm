cmake_minimum_required(VERSION 3.31)

project(chcuda VERSION 0.1.0
  LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
# set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-g -G -gencode arch=compute_60,code=sm_60" CACHE STRING "Debug flags for cuda" FORCE)
set(CMAKE_CUDA_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-g -G -O0 -gencode arch=compute_70,code=sm_70" CACHE STRING "Debug flags for cuda" FORCE)
#set(CMAKE_CUDA_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-O3 -lineinfo -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80" CACHE STRING "Release flags for cuda" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-O3 -lineinfo  -gencode arch=compute_80,code=sm_80" CACHE STRING "Release flags for cuda" FORCE)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

include(src_files.cmake)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(NetCDF COMPONENTS C REQUIRED)

add_library(chcudadynlib SHARED ${src_files})
target_include_directories(chcudadynlib PUBLIC include)
target_include_directories(chcudadynlib PUBLIC ${NETCDF_INCLUDES})
target_include_directories(chcudadynlib PUBLIC ${Python_SITEARCH}/nvidia/nvtx/include)

target_compile_features(chcudadynlib PUBLIC cxx_std_20 cuda_std_20)
#target_compile_features(chcudadynlib PUBLIC cxx_std_20)
set_property(TARGET chcudadynlib PROPERTY CUDA_ARCHITECTURES 70 80)

target_compile_options(chcudadynlib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:

  # --generate-line-info
  # -lineinfo
  --expt-relaxed-constexpr
  --extended-lambda

  # -O3
  >)

target_compile_definitions(chcudadynlib PUBLIC USE_TEXTURE_OBJECTS)

target_link_libraries(chcudadynlib CUDA::cufft ${NETCDF_LIBRARIES} ${_Python_LIBRARY_RELEASE})
set_target_properties(chcudadynlib PROPERTIES LINKER_LANGUAGE CUDA)

# target_link_directories(chcudadynlib CUDA::cudart)
set_property(TARGET chcudadynlib PROPERTY CUDA_RUNTIME_LIBRARY SHARED) # For CMake 3.23 and later

install(TARGETS chcudadynlib DESTINATION apocharmm)

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

if(PYTHON_FOUND)
  find_package(pybind11 CONFIG REQUIRED)

  # pybind11_add_module(_core python/pywrapper_main.cu)
  python_add_library(_core MODULE python/pywrapper_main.cu WITH_SOABI)
  target_link_libraries(_core PRIVATE pybind11::headers chcudadynlib)

  set_property(TARGET _core PROPERTY CUDA_ARCHITECTURES 70 80)
  target_include_directories(_core PUBLIC include)

  # target_link_libraries(apocharmm PUBLIC chcudadynlib )
  target_sources(_core PRIVATE
    python/exportCharmmPSF.cu
    python/exportCharmmContext.cu
    python/exportCharmmParameters.cu
    python/exportCoordinates.cu
    python/exportForceManager.cu
    python/exportIntegrator.cu
    python/exportSubscriber.cu
    python/exportForceManagerGenerator.cu
    python/exportPBC.cu
  )

  install(TARGETS _core DESTINATION apocharmm)
  set_target_properties(_core PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

# Testing
if(NOT "${SBUILD}" STREQUAL "2")
  enable_testing()
  add_subdirectory(test)
else()
  message(STATUS "SBUILD is  2")
endif()

# Documentation
find_package(Doxygen)
option(BUILD_DOCS "Build documentation" OFF)

if(BUILD_DOCS)
  if(Doxygen_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/config/doxybis)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/build/Doxyfile.out)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    add_custom_target(docs/doxygen/build ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating Doxygen documentation"
      VERBATIM
    )

  else(Doxygen_FOUND)
    message(FATAL_ERROR "Doxygen not found")
  endif(Doxygen_FOUND)
endif(BUILD_DOCS)

# add_subdirectory(fortran_api)

# file(GLOB fortran_api_files fortran_api/*.F90)
# file(GLOB cuda_api_files fortran_api/*.cu)
# enable_language(Fortran)
# add_library(chcudaFortranAPI STATIC ${fortran_api_files} ${cuda_api_files})
# set_property(TARGET chcudaFortranAPI PROPERTY CUDA_ARCHITECTURES 60 70)
# target_link_libraries(chcudaFortranAPI chcudalib)
# add_executable(fortran_exe fortran_src/main.F90)
# target_link_libraries(fortran_exe chcudaFortranAPI)

# # Get Python site-packages path
# execute_process(
# COMMAND python -c "import site; print(site.getsitepackages()[0])"
# OUTPUT_VARIABLE PYTHON_SITEARCH
# OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# # Find NVTX dynamically (avoid hardcoding version)
# file(GLOB NVTX_LIBRARY_CANDIDATES "${Python_SITEARCH}/nvidia/nvtx/lib/libnvToolsExt.so*")
# list(SORT NVTX_LIBRARY_CANDIDATES)
# list(GET NVTX_LIBRARY_CANDIDATES -1 NVTX_LIBRARY)  # Pick the latest version

# if(NOT NVTX_LIBRARY)
# message(FATAL_ERROR "Could not find NVTX library in ${PYTHON_SITEARCH}/nvidia/nvtx/lib/")
# else()
# message(STATUS "Found NVTX: ${NVTX_LIBRARY}")
# endif()

# # Find CUFFT dynamically
# file(GLOB CUFFT_LIBRARY_CANDIDATES "${Python_SITEARCH}/nvidia/cufft/lib/libcufft.so*")
# list(SORT CUFFT_LIBRARY_CANDIDATES)
# list(GET CUFFT_LIBRARY_CANDIDATES -1 CUDA_CUFFT_LIBRARIES)

# if(NOT CUDA_CUFFT_LIBRARIES)
# message(FATAL_ERROR "Could not find CUFFT library in ${PYTHON_SITEARCH}/nvidia/cufft/lib/")
# else()
# message(STATUS "Found CUFFT: ${CUDA_CUFFT_LIBRARIES}")
# endif()

# find_library( NVTX_LIBRARY
# nvToolsExt
# PATHS
# ${Python_SITEARCH}/nvidia/nvtx/lib)
# #        ENV LD_LIBRARY_PATH )
# find_library( CUDA_CUFFT_LIBRARIES
# cufft
# PATHS ENV LD_LIBRARY_PATH )

# message("Samar ${CUDA_CUFFT_LIBRARIES}")

# add_library(chcudalib STATIC ${src_files})
# target_include_directories(chcudalib PUBLIC include)
# set_property(TARGET chcudalib PROPERTY CUDA_ARCHITECTURES 70 80)
# target_compile_definitions(chcudalib PUBLIC USE_TEXTURE_OBJECTS)

# target_include_directories(chcudalib PUBLIC ${NETCDF_INCLUDES})

# target_include_directories(chcudalib SYSTEM PUBLIC external/eigen-3.3.7)

# # The inclusion of _Python_INCLUDE_DIR and pybind11 here seem necessary to
# # properly capture SIGINT by the Python interface. However, this also slows
# # down compilation as it links heavy libraries (?)
# target_include_directories(chcudalib PUBLIC external/LBFGSpp/include external/pybind11/include ${_Python_INCLUDE_DIR} )
# #target_include_directories(chcudalib PUBLIC external/tinyxml2)

# target_compile_options(chcudalib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
# #                       --generate-line-info
# #                     -lineinfo
# --expt-relaxed-constexpr
# #                     -O3
# >)
# target_compile_features(chcudalib PUBLIC cxx_std_20)

# target_link_libraries(chcudalib ${_Python_LIBRARY_RELEASE})
# target_link_libraries(chcudalib ${NVTX_LIBRARY}  ${CUDA_CUFFT_LIBRARIES} ${NETCDF_LIBRARIES})

