cmake_minimum_required (VERSION 3.13)
project(MParT VERSION 0.0.1)


# Add the cmake folder as a search path and include files
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)
INCLUDE(Submodules)

##############################################################
# Options 

option(MPART_PYTHON "Build python bindings with pybind11" ON)
option(MPART_JULIA "Build julia bindings with CxxWrap.jl" OFF)

##############################################################
# Compiler configuration

# Set the C++ version
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported

# Set the build type to Release if it's not explicity set already
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wno-deprecated-gpu-targets" COMPILER_IS_NVCC)
if(COMPILER_IS_NVCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-gpu-targets --expt-relaxed-constexpr")
    add_compile_definitions(EIGEN_NO_CUDA)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

##############################################################
# Dependencies

# Add Kokkos
find_package(Kokkos QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "Could not find Kokkos. Will build internally.")
    UpdateGitSubmodule(kokkos)
    add_subdirectory(external/kokkos)
else() 
    message(STATUS "Found Kokkos.")
endif()

# Add Eigen
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    UpdateGitSubmodule(eigen)
    add_subdirectory(external/eigen)
endif()

# Add pybind11 if necessary 
if(MPART_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    UpdateGitSubmodule(pybind11)
    add_subdirectory(external/pybind11)
  endif()
endif()

# Add the current source directory as an include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add Julia if necessary
if(MPART_JULIA)
  
  # Set the julie executable and look for the location of cxxwrap
  if(NOT DEFINED JULIA_EXE)
      set(JULIA_EXE "julia")
  endif()
  execute_process(COMMAND ${JULIA_EXE} -e "using CxxWrap; print(CxxWrap.prefix_path())" OUTPUT_VARIABLE cxxwrap_location)
  list(APPEND CMAKE_PREFIX_PATH "${cxxwrap_location}" )
  
  find_package(JlCxx)

  if(NOT JlCxx_FOUND)
    set(MPART_JULIA OFF)
    message(WARNING "Request Julia bindings but CMake could not find JlCxx package.  Setting MPART_JULIA=OFF.")
  else()
    get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
    get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
    get_filename_component(JlCxx_src ${JlCxx_location} DIRECTORY)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
    include_directories(${Julia_INCLUDE_DIRS} ${JlCxx_src}/include)
  endif()

endif()

##############################################################
# MParT library
add_library(mpart "")
add_subdirectory(src)
target_link_libraries(mpart Kokkos::kokkos Eigen3::Eigen)

if(MPART_PYTHON)
    add_subdirectory(bindings/python)
endif()

if(MPART_JULIA)
    add_subdirectory(bindings/julia)
endif()

##############################################################
# Testing 

# Install Catch2
find_package(Catch2 3 QUIET)
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

add_executable(PrintKokkosInfo tests/KokkosInfo.cpp)
target_link_libraries(PrintKokkosInfo Kokkos::kokkos)

##############################################################
# Instalaltion 
install(TARGETS mpart DESTINATION lib)

install(DIRECTORY MParT
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
	      FILES_MATCHING PATTERN "*.h")

##############################################################
# Documentation
find_package(Doxygen)


find_package(Sphinx)
if(Sphinx_FOUND AND Doxygen_FOUND)

    message("SPHINX Executable = ${SPHINX_EXECUTABLE}")
    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/docs)
    set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/mpart.doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/mpart.doxyfile @ONLY)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)

    add_custom_target(sphinx
                  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/mpart.doxyfile
                  COMMAND ${SPHINX_EXECUTABLE} -b html
                  # Tell Breathe where to find the Doxygen output
                  -Dbreathe_projects.mpart=${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen/xml
                  ${SPHINX_SOURCE} ${SPHINX_BUILD}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating documentation with Sphinx")

endif()
##############################################################