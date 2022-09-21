cmake_minimum_required (VERSION 3.13)
project(MParT VERSION 0.0.1)


# Add the cmake folder as a search path and include files
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)
Include(FetchContent)

##############################################################
# Options

option(MPART_PYTHON "Build python bindings with pybind11" ON)
option(MPART_MATLAB "Build matlab bindings with pybind11" ON)
option(MPART_JULIA "Build julia bindings with CxxWrap.jl" ON)
option(MPART_FETCH_DEPS "If CMake should be allowed to fetch and build external dependencies that weren't found." ON)

##############################################################
# Installation path configuration 
Include(SetInstallPaths)

##############################################################
# Compiler configuration

# Set the C++ version
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported

# Set the build type to Release if it's not explicity set already
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(COMPILER_IS_NVCC OFF)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wno-deprecated-gpu-targets" COMPILER_IS_NVCC1)
check_cxx_compiler_flag("--expt-relaxed-constexpr" COMPILER_IS_NVCC2)
if(COMPILER_IS_NVCC1 AND COMPILER_IS_NVCC2)
    add_definitions(-DMPART_ENABLE_GPU)
    message(STATUS "GPU support detected")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-gpu-targets --expt-relaxed-constexpr")
    add_compile_definitions(EIGEN_NO_CUDA)
    set(COMPILER_IS_NVCC ON)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

##############################################################
# RPATH settings

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")



##############################################################
# Dependencies

# Add Kokkos
find_package(Kokkos QUIET)

if(NOT Kokkos_FOUND)
    IF(MPART_FETCH_DEPS)
        message(STATUS "Could not find Kokkos.  Fetching source.")

        FetchContent_Declare(
            kokkos
            GIT_REPOSITORY https://github.com/kokkos/kokkos
            GIT_TAG 3.5.00
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(kokkos)
    else()
        message(FATAL_ERROR "Could not find Kokkos library and MPART_FETCH_DEPS=OFF, so CMake will not attempt to fetch and install Kokkos itself." )
    endif()
else()
    message(STATUS "Found Kokkos!")
endif()


if(${COMPILER_IS_NVCC})
    message(STATUS "Searching for CUBLAS and CUSOLVER linear algebra libraries.")
    find_package(CUDAToolkit COMPONENTS cudart cublas cusolver REQUIRED)
    set(CUDA_LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver)

else()
    message(STATUS "MParT is not compiled with CUDA support, so CUBLAS and CUSOLVER will not be used.")
    set(CUDA_LIBRARIES "")
endif()

# Add Eigen
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    if(MPART_FETCH_DEPS)
        message(STATUS "Could not find Eigen. Fetching source.")

        FetchContent_Declare(
            eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(eigen3)
    else()
        message(FATAL_ERROR "Could not find Eigen3 library and MPART_FETCH_DEPS=OFF, so CMake will not attempt to fetch and install Eigen3 itself." )
    endif()
else()
  message(STATUS "Found Eigen!")
endif()

# Add pybind11 if necessary
if(MPART_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    if(${MPART_FETCH_DEPS})
        message(STATUS "Could not find pybind11. Fetching source.")

        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11
            GIT_TAG        v2.10.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    else()
        message(FATAL_ERROR "Could not find pybind11 library and MPART_FETCH_DEPS=OFF, so CMake will not attempt to fetch and install pybind11 itself." )
    endif()

  else()
    message(STATUS "Found pybind11!")
  endif()
endif()


# Add Julia if necessary
if(MPART_JULIA)

  find_package(Julia)

  if(NOT Julia_FOUND)
    set(MPART_JULIA OFF)
    message(WARNING "Requested Julia bindings but CMake could not find Julia executable.  Setting MPART_JULIA=OFF.")
  else()

    # Get a hint for the location of JlCxx 
    get_filename_component(PARENT_DIR ${Julia_LIBRARY_DIR} DIRECTORY)

    # Use JlCxx_DIR to tell where CxxWrap is located
    if(NOT DEFINED JlCxx_DIR)
        execute_process(COMMAND ${Julia_EXECUTABLE} -e "import CxxWrap; print(CxxWrap.prefix_path())" OUTPUT_VARIABLE cxxwrap_location)
        set(JlCxx_DIR "${cxxwrap_location}/lib/cmake/JlCxx/")
    endif()

    find_package(JlCxx)

    if(NOT JlCxx_FOUND)
        set(MPART_JULIA OFF)
        message(WARNING "Requested Julia bindings but CMake could not find JlCxx package.  Setting MPART_JULIA=OFF.")
    else()
        get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
        message(STATUS "Found JlCxx: ${JlCxx_location}")

        add_definitions(-DJULIA_ENABLE_THREADING)
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${Julia_LIBRARY_DIR}")
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    endif()
  endif()

endif()

##############################################################
# MParT library
add_library(mpart "")
add_library (MParT::mpart ALIAS mpart)

target_link_libraries(mpart PRIVATE Kokkos::kokkos Eigen3::Eigen ${CUDA_LIBRARIES})

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

# Add matlab if necessary
if(MPART_MATLAB)
    find_package(Matlab)
    if(Matlab_FOUND)
        add_subdirectory(bindings/matlab)
    else()
        message(STATUS "Could not find matlab.  Matlab bindings will not be built.")
    endif()
endif()

##############################################################
# Testing

option(MPART_BUILD_TESTS "If ON, unit tests will be built." ON)

if(MPART_BUILD_TESTS)

    # Install Catch2
    find_package(Catch2 QUIET)
    if(NOT Catch2_FOUND)
        IF(MPART_FETCH_DEPS)
            message(STATUS "Could not find Catch2.  Fetching Catch2 source.")

            FetchContent_Declare(
                Catch2
                GIT_REPOSITORY https://github.com/catchorg/Catch2.git
                GIT_TAG        v3.0.0-preview3
            )

            FetchContent_MakeAvailable(Catch2)
        else()
            message(WARNING "Could not find Catch library and MPART_FETCH_DEPS=OFF, so CMake will not attempt to fetch and install Catch2 itself.  Tests will not be built." )
            set(MPART_BUILD_TESTS OFF)
        endif()

    else()
        message(STATUS "Found Catch2!")
    endif()

endif()

if(MPART_BUILD_TESTS)
    # Define test sources
    add_subdirectory(tests)
    add_executable(RunTests ${TEST_SOURCES})
    target_link_libraries(RunTests PRIVATE mpart Catch2::Catch2 Kokkos::kokkos Eigen3::Eigen ${CUDA_LIBRARIES})
endif()

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

Include(BuildDocs)

##############################################################
