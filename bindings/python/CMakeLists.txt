
set(PYTHON_BINDING_SOURCES
  src/Wrapper.cpp
  src/CommonPybindUtilities.cpp
  src/MultiIndex.cpp
  src/MapOptions.cpp
  src/ConditionalMapBase.cpp
  src/ParameterizedFunctionBase.cpp
  src/AffineMap.cpp
  src/TriangularMap.cpp
  src/ComposedMap.cpp
  src/SummarizedMap.cpp
  # src/DebugMap.cpp
  src/MapFactory.cpp
  src/IdentityMap.cpp

  ../common/src/CommonUtilities.cpp
)

if(MPART_ARCHIVE)
  set(PYTHON_BINDING_SOURCES ${PYTHON_BINDING_SOURCES}
    src/Serialization.cpp
  )
endif()

if(MPART_OPT)
  set(PYTHON_BINDING_SOURCES ${PYTHON_BINDING_SOURCES}
    src/MapObjective.cpp
    src/TrainMap.cpp
    src/TrainMapAdaptive.cpp
  )
endif()

include_directories(./include ../common/include)
pybind11_add_module(pympart SHARED NO_EXTRAS ${PYTHON_BINDING_SOURCES})

target_link_libraries(pympart PRIVATE mpart Kokkos::kokkos Eigen3::Eigen ${EXT_LIBRARIES})

# Add an installation target for the python bindings
install(TARGETS pympart DESTINATION "${PYTHON_INSTALL_PREFIX}")
install(DIRECTORY package/ DESTINATION "${PYTHON_INSTALL_PREFIX}")

# See https://gist.github.com/kprussing/db21614ca5b51cedff07dfb70059f280
#set(lib_path "${PYTHON_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
#message(STATUS "LIB_PATH = ${lib_path}")
#list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${lib_path}" is_system)
if(SKBUILD)
    if (SKBUILD_LIB_RPATH)
        set_target_properties(pympart PROPERTIES INSTALL_RPATH "${SKBUILD_LIB_RPATH}")
    endif()
endif()