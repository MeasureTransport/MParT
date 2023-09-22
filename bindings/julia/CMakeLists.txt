set(JULIA_BINDING_SOURCES
  src/CommonJuliaUtilities.cpp
  src/ConditionalMapBase.cpp
  src/JlArrayConversions.cpp
  src/MapFactory.cpp
  src/MapOptions.cpp
  src/MultiIndex.cpp
  src/ParameterizedFunctionBase.cpp
  src/TriangularMap.cpp
  src/Wrapper.cpp
  src/AffineMap.cpp
  src/ComposedMap.cpp
  ../common/src/CommonUtilities.cpp
  src/Distributions.cpp
)

if(MPART_OPT)
  set(JULIA_BINDING_SOURCES ${JULIA_BINDING_SOURCES}
      src/MapObjective.cpp
      src/TrainMap.cpp
      src/TrainMapAdaptive.cpp
  )
endif()

include_directories(./include ../common/include)
add_library(mpartjl SHARED ${JULIA_BINDING_SOURCES})
target_link_libraries(mpartjl PRIVATE mpart JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl Kokkos::kokkos Eigen3::Eigen ${EXT_LIBRARIES})

# Add an installation target for the julia bindings
install(TARGETS mpartjl DESTINATION "${JULIA_INSTALL_PREFIX}")