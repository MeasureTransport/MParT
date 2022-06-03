
set(PYTHON_BINDING_SOURCES
  src/Wrapper.cpp
  src/CommonPybindUtilities.cpp
  src/MultiIndex.cpp
  ../common/src/CommonUtilities.cpp
)

include_directories(./include ../common/include)
pybind11_add_module(pympart SHARED NO_EXTRAS ${PYTHON_BINDING_SOURCES})
target_link_libraries(pympart PRIVATE mpart Kokkos::kokkos Eigen3::Eigen)

# Add an installation target for the python bindings
install(TARGETS pympart DESTINATION python/mpart)
install(DIRECTORY package/ DESTINATION python/mpart)