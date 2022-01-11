cmake_minimum_required (VERSION 3.10)
project(MParT VERSION 0.0.1)


# Add the cmake folder as a search path and include files
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)
INCLUDE(Submodules)

##############################################################
# Compiler configuration

# Set the C++ version
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported

# Set the build type to Release if it's not explicity set already
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

##############################################################
# Dependencies

# Add Kokkos
find_package(Kokkos)
if(NOT Kokkos_FOUND)
    UpdateGitSubmodule(kokkos)
    add_subdirectory(external/kokkos)
    include_directories(${Kokkos_INCLUDE_DIRS_RET})
endif()

# Add Eigen
find_package(Eigen3 3.3 NO_MODULE)
if(NOT Eigen3_FOUND)
    UpdateGitSubmodule(eigen)
    add_subdirectory(external/eigen)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)
endif()

# Add the current source directory as an include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


##############################################################
# MParT library
add_subdirectory(src)
add_library(mpart ${MPART_SOURCES})
target_link_libraries(mpart Kokkos::kokkos)


##############################################################
# Testing 

# Install Catch2
find_package(Catch2 3)
if(NOT Catch2_FOUND)
    Include(FetchContent)

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.0.0-preview3
    )
  
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()

# Define test sources
add_subdirectory(tests)
add_executable(RunTests ${TEST_SOURCES})
target_link_libraries(RunTests PRIVATE mpart Catch2::Catch2)

##############################################################
# Documentation
find_package(Doxygen)

# if(Doxygen_FOUND)

#     message(STATUS "To build documentation, run 'make doc'")

#     configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/mpart.doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/mpart.doxyfile @ONLY)
#     add_custom_target(doc
#         COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/mpart.doxyfile
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#         COMMENT "Generating API documentation with Doxygen" VERBATIM
#     )

# endif()


find_package(Sphinx)
if(Sphinx_FOUND AND Doxygen_FOUND)

    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/docs)
    set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/mpart.doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/mpart.doxyfile @ONLY)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)

    add_custom_target(doc
                  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/mpart.doxyfile
                  COMMAND ${SPHINX_EXECUTABLE} -b html
                  # Tell Breathe where to find the Doxygen output
                  -Dbreathe_projects.mpart=${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen/xml
                  ${SPHINX_SOURCE} ${SPHINX_BUILD}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating documentation with Sphinx")

endif()
##############################################################