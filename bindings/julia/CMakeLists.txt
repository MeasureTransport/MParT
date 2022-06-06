set(JULIA_BINDING_SOURCES
  src/Wrapper.cpp
  src/CommonJuliaUtilities.cpp
  ../common/src/CommonUtilities.cpp
)

include_directories(./include ../common/include)
add_library(mpartjl SHARED ${JULIA_BINDING_SOURCES})
target_link_libraries(mpartjl PRIVATE mpart JlCxx::cxxwrap_julia Kokkos::kokkos Eigen3::Eigen)

# Add an installation target for the julia bindings
install(TARGETS mpartjl DESTINATION julia/mpart)
install(DIRECTORY package/ DESTINATION julia/mpart)