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
check_cxx_compiler_flag("-Wno-deprecated-gpu-targets" COMPILER_IS_NVCC1)
check_cxx_compiler_flag("--expt-relaxed-constexpr" COMPILER_IS_NVCC2)
if(COMPILER_IS_NVCC1 AND COMPILER_IS_NVCC2)
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
    message(STATUS "Could not find Kokkos.  Using internal build.")
    UpdateGitSubmodule(kokkos)
    add_subdirectory(external/kokkos)
else()
    message(STATUS "Found Kokkos!")
endif()

# Add Eigen
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Could not find Eigen.  Using internal build.")
    UpdateGitSubmodule(eigen)
    add_subdirectory(external/eigen)
else()
  message(STATUS "Found Eigen!")
endif()

# Add pybind11 if necessary
if(MPART_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    message(STATUS "Could not find pybind11. Using internal build.")
    UpdateGitSubmodule(pybind11)
    add_subdirectory(external/pybind11)
  else()
    message(STATUS "Found pybind11!")
  endif()
endif()


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
    message(WARNING "Request Julia bindings but CMake could not find CxxWrap package.  Setting MPART_JULIA=OFF.")
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
add_library (MParT::mpart ALIAS mpart)

target_link_libraries(mpart PRIVATE Kokkos::kokkos Eigen3::Eigen)

target_include_directories(mpart
                           PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )

add_subdirectory(src)

if(MPART_PYTHON)
    add_subdirectory(bindings/python)
endif()

if(MPART_JULIA)
    add_subdirectory(bindings/julia)
endif()

##############################################################
# Testing

# Install Catch2
find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    message(STATUS "Could not find Catch2.  Using internal build.")
    UpdateGitSubmodule(catch2)
    PinSubmoduleVersion(catch2 v3.0.0-preview3)
    add_subdirectory(external/catch2)
else()
    message(STATUS "Found Catch2!")
endif()

# Define test sources
add_subdirectory(tests)
add_executable(RunTests ${TEST_SOURCES})
target_link_libraries(RunTests PRIVATE mpart Catch2::Catch2 Kokkos::kokkos Eigen3::Eigen)

add_executable(PrintKokkosInfo tests/KokkosInfo.cpp)
target_link_libraries(PrintKokkosInfo Kokkos::kokkos)

##############################################################
# Installation
install(TARGETS mpart
        EXPORT MParTTargets
        DESTINATION lib)

install(DIRECTORY MParT
        DESTINATION include
	    FILES_MATCHING PATTERN "*.h")

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/MParTConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/MParTConfigVersion.cmake
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/MParTConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/MParTConfigVersion.cmake
  DESTINATION lib/cmake/MParT
)


export (TARGETS mpart ${KOKKOS_EXPORTS} NAMESPACE MParT:: FILE MParTTargets.cmake)
export (PACKAGE MParT)

install (EXPORT MParTTargets NAMESPACE MParT:: DESTINATION lib/cmake/MParT)

install (FILES ${CMAKE_CURRENT_BINARY_DIR}/MParTConfig.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/MParTConfigVersion.cmake
         DESTINATION lib/cmake/MParT)



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