cmake_minimum_required(VERSION 3.14)

project(chcuda VERSION 0.1.0
  LANGUAGES CXX  CUDA)

#find_package(CUDA REQUIRED)

add_subdirectory(external/pybind11)
#find_package(Torch REQUIRED)

#add_subdirectory(external/tinyxml2)

#set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
#set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-g -G -gencode arch=compute_60,code=sm_60" CACHE STRING "Debug flags for cuda" FORCE)

set(CMAKE_CUDA_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-g -G -O0 -gencode arch=compute_70,code=sm_70" CACHE STRING "Debug flags for cuda" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-O3 -lineinfo -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80" CACHE STRING "Release flags for cuda" FORCE)

# chcudalib can be STATIC SHARED MODULE
include(src_files.cmake )
include(include_files.cmake )
#add_library(chcudalib STATIC ${include_files} ${src_files})

# Find Python
find_package (Python COMPONENTS Interpreter Development)

add_library(chcudalib STATIC ${src_files})
target_include_directories(chcudalib PUBLIC include)
set_property(TARGET chcudalib PROPERTY CUDA_ARCHITECTURES 70 80)
target_compile_definitions(chcudalib PUBLIC USE_TEXTURE_OBJECTS)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(NetCDF COMPONENTS C REQUIRED)

target_include_directories(chcudalib PUBLIC ${NETCDF_INCLUDES})

target_include_directories(chcudalib SYSTEM PUBLIC external/eigen-3.3.7)

# The inclusion of _Python_INCLUDE_DIR and pybind11 here seem necessary to
# properly capture SIGINT by the Python interface. However, this also slows
# down compilation as it links heavy libraries (?)
target_include_directories(chcudalib PUBLIC external/LBFGSpp/include external/pybind11/include ${_Python_INCLUDE_DIR} )
#target_include_directories(chcudalib PUBLIC external/tinyxml2)

target_compile_options(chcudalib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  #                       --generate-line-info
  #                     -lineinfo 
                       --expt-relaxed-constexpr
  #                     -O3
                      >)
target_compile_features(chcudalib PUBLIC cxx_std_20)

add_library(chcudadynlib SHARED ${src_files})
target_include_directories(chcudadynlib PUBLIC include)
target_include_directories(chcudadynlib SYSTEM PUBLIC external/eigen-3.3.7)
target_include_directories(chcudadynlib PUBLIC external/LBFGSpp/include external/pybind11/include ${_Python_INCLUDE_DIR}) 
#target_include_directories(chcudadynlib PUBLIC ${netCDF_INCLUDE_DIR})
target_include_directories(chcudadynlib PUBLIC ${NETCDF_INCLUDES})
target_compile_features(chcudadynlib PUBLIC cxx_std_20)
set_property(TARGET chcudadynlib PROPERTY CUDA_ARCHITECTURES 70 80)

target_compile_options(chcudadynlib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  #                    --generate-line-info
  #                     -lineinfo
                       --expt-relaxed-constexpr
  #                     -O3
                      >)
set_target_properties(chcudadynlib PROPERTIES LINKER_LANGUAGE CUDA)
target_compile_definitions(chcudadynlib PUBLIC USE_TEXTURE_OBJECTS)

## Add CUDA's NVTX and cufft libraries
find_library( NVTX_LIBRARY
        nvToolsExt
        PATHS ENV LD_LIBRARY_PATH )
find_library( CUDA_CUFFT_LIBRARIES
        cufft
        PATHS ENV LD_LIBRARY_PATH )        
target_link_libraries(chcudalib ${NVTX_LIBRARY}  ${CUDA_CUFFT_LIBRARIES} ${NETCDF_LIBRARIES})
target_link_libraries(chcudalib ${_Python_LIBRARY_RELEASE})
target_link_libraries(chcudadynlib ${NVTX_LIBRARY}  ${CUDA_CUFFT_LIBRARIES} ${NETCDF_LIBRARIES})
target_link_libraries(chcudadynlib ${_Python_LIBRARY_RELEASE})
#target_link_libraries(chcudalib external/tinyxml2)
#target_link_libraries(chcudadynlib external/tinyxml2)

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
#message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
#message(STATUS  " {NVTX_LIBRARY}  {CUDA_CUFFT_LIBRARIES} " ${NVTX_LIBRARY}  ${CUDA_CUFFT_LIBRARIES})

#add_subdirectory(fortran_api)

#file(GLOB fortran_api_files fortran_api/*.F90)
#file(GLOB cuda_api_files fortran_api/*.cu)
#enable_language(Fortran)
#add_library(chcudaFortranAPI STATIC ${fortran_api_files} ${cuda_api_files})
#set_property(TARGET chcudaFortranAPI PROPERTY CUDA_ARCHITECTURES 60 70)

# Find Python
#find_package (Python COMPONENTS Interpreter Development)
if(PYTHON_FOUND)
    set(Python_ENABLED ON)
    #find_package(pybind11 CONFIG)
    #add_subdirectory(external/pybind11)
    add_subdirectory(python)
endif()
#target_link_libraries(chcudaFortranAPI chcudalib)

#add_executable(fortran_exe fortran_src/main.F90)
#target_link_libraries(fortran_exe chcudaFortranAPI)

# Testing
enable_testing()
add_subdirectory(test)

# Documentation
find_package(Doxygen)
option(BUILD_DOCS "Build documentation" OFF)

#if(CMAKE_BUILD_TYPE STREQUAL "Release")
#  message(STATUS "Build is ${CMAKE_BUILD_TYPE}. indeed")
#  
#  set(BUILD_DOCS ON)
#endif(CMAKE_BUILD_TYPE STREQUAL "Release")

if(BUILD_DOCS)
  if (Doxygen_FOUND)
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
