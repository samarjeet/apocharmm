# Add any new tests to this list or the one below!
set(SOURCES_UNITTESTS_TESTS

    # Base tests
    unittest-charmmContext.cpp
    unittest-forceManager.cu
    unittest-subscribers.cpp
    unittest-readCoordinates.cpp
    unittest-integrators.cpp
    unittest-readPRM.cpp

    # unittest-ForceManagerGenerator.cpp
    unittest-restraintForce.cu
    unittest-logger.cpp
    unittest-utils.cpp
    unittest-cudaContainers.cpp
    unittest-restart.cpp
    unittest-checkpoint.cpp

    # Ref tests
    unittest-eneForces.cpp

    # Features tests
    unittest-holonomic.cpp
    unittest-p21.cpp

    # Integrators
    unittest-langevinThermostat.cpp
    unittest-langevinPiston.cpp

    # Systems
    unittest-argon1000.cpp
    unittest-waterbox.cpp
    unittest-benchmarks.cpp
    unittest-insulin1.cpp

    # To be classed/cleaned
    unittest-alchemical.cpp
    unittest-fep.cpp
    unittest-eds.cpp
    unittest-bridgeEDS.cpp
    unittest-benzenePyrrole.cpp

    # unittest-sep_benzene_pyrrole.cpp
    unittest-minimize.cpp

    unittest-pme.cpp
    unittest-direct.cpp

    unittest-replicaExchange.cpp
    unittest-emap.cpp

    # unittest-mpid.cpp

    # unittest-addForce.cu
    # unittest-boltzmannVelocity.cu
    # unittest-integrator.cpp
    # unittest-jac.cpp
    # unittest-psf.cpp
    # unittest-verletIntegrator.cu
    # unittest-testSystem.cu
    # unittest-leapFrogIntegrator.cu
    # unittest-nacl.cu
    # unittest-nonEquil.cu
    # unittest-noseHoover.cpp
    # unittest-rcsb.cu
    # unittest-walp.cpp
    # unittest-argon10.cpp
    # unittest-context_creation.cu
    # unittest-drudeEnergy.cu
    # unittest-forceManager.cu
    # unittest-velocityVerletIntegrator.cu
)

#
# Stuff below shouldn't need to be modified
#
set(SOURCES_UNITTESTS_MAIN unittest-main.cpp)

# A function to check that the given hard-coded list contains all headers + sources in the given folder.
function(CheckFileList LIST_VAR FOLDER)
    set(MESSAGE " should be added to the variable ${LIST_VAR}")
    set(MESSAGE "${MESSAGE} in ${CMAKE_CURRENT_LIST_FILE}\n")
    file(GLOB GLOBBED_LIST "${FOLDER}/*.cpp"
        "${FOLDER}/*.hpp")
    list(REMOVE_ITEM GLOBBED_LIST ${${LIST_VAR}})

    # foreach(EXTRA_ITEM ${GLOBBED_LIST})
    # string(REPLACE "${CATCH_DIR}/" "" RELATIVE_FILE_NAME "${EXTRA_ITEM}")
    # message(AUTHOR_WARNING "The file \"${RELATIVE_FILE_NAME}\"${MESSAGE}")
    # endforeach()
endfunction()

set(UNITTEST_DIR ${PROJECT_SOURCE_DIR}/test/unittests)

# set( HEADER_DIR ${PROJECT_SOURCE_DIR}/external/Catch2/single_include )
set(HEADER_DIR ${UNITTEST_DIR})

# Check if all sources are listed; warn if not.
set(SOURCES_ALL
    ${SOURCES_UNITTESTS_MAIN}
    ${SOURCES_UNITTESTS_TESTS}
)

foreach(name ${SOURCES_ALL})
    list(APPEND SOURCES_ALL_PATH ${UNITTEST_DIR}/${name})
endforeach()

CheckFileList(SOURCES_ALL_PATH ${UNITTEST_DIR})

# Create serial target names.
string(REPLACE ".cpp" "" BASENAMES_UNITTESTS_TESTS "${SOURCES_UNITTESTS_TESTS}")
string(REPLACE ".cu" "" BASENAMES_UNITTESTS_TESTS "${BASENAMES_UNITTESTS_TESTS}")
set(TARGETS_UNITTESTS_TESTS ${BASENAMES_UNITTESTS_TESTS})

set(TARGETS_ALL ${TARGETS_UNITTESTS_TESTS} CatchMain)

# Define program targets:
add_library(CatchMain OBJECT ${UNITTEST_DIR}/${SOURCES_UNITTESTS_MAIN} ${HEADER_DIR}/catch.hpp)

foreach(name ${TARGETS_UNITTESTS_TESTS})
    if(${name} MATCHES "unittest-restraintForce" OR ${name} MATCHES "unittest-forceManager")
        set(filename ${name}.cu)
    else()
        set(filename ${name}.cpp)
    endif()

    add_executable(${name} ${filename} $<TARGET_OBJECTS:CatchMain> ${HEADER_DIR}/catch.hpp)

    target_compile_definitions(${name} PUBLIC DATA_PATH="${PROJECT_SOURCE_DIR}/test/data/")
    target_link_libraries(${name} PRIVATE ${EXTERNAL_LIBRARIES})
    set_property(TARGET ${name} PROPERTY CUDA_RUNTIME_LIBRARY SHARED) # For CMake 3.23 and later

    # target_link_libraries( ${name} PRIVATE chcudalib)
    target_link_libraries(${name} PRIVATE chcudadynlib)

    # target_link_libraries(${name} PRIVATE )
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CUDA)
    add_test(NAME ${name}
        COMMAND ${name}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test/unittests)
endforeach()

foreach(name ${TARGETS_ALL})
    target_include_directories(${name} PRIVATE ${HEADER_DIR})

    set_property(TARGET ${name} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${name} PROPERTY CUDA_ARCHITECTURES 60 70 80)

    # GCC specific warning go here
    if(CMAKE_CXX_COMPILER_ID MATCHES "GCC")
        # Actually keep these
        target_compile_options(${name} PRIVATE -Wweak-vtables -Wexit-time-destructors -Wglobal-constructors -Wmissing-noreturn -Wno-int-in-bool-context)
    endif()

    # Clang specific warning go here
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Actually keep these
        target_compile_options(${name} PRIVATE -Wweak-vtables -Wexit-time-destructors -Wglobal-constructors)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${name} PRIVATE /W4 /w44265 /WX)
    endif()
endforeach()
