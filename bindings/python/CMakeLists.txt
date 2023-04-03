
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
    src/AdaptiveTransportMap.cpp
  )
endif()

include_directories(./include ../common/include)
pybind11_add_module(pympart SHARED NO_EXTRAS ${PYTHON_BINDING_SOURCES})

target_link_libraries(pympart PRIVATE mpart Kokkos::kokkos Eigen3::Eigen ${EXT_LIBRARIES})

# Add an installation target for the python bindings
install(TARGETS pympart DESTINATION "${PYTHON_INSTALL_PREFIX}")
install(DIRECTORY package/ DESTINATION "${PYTHON_INSTALL_PREFIX}")