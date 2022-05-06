set(JULIA_BINDING_SOURCES
  src/Wrapper.cpp
#   src/CommonJuliaUtilities.cpp
)

include_directories(./include)
add_library(mpartjl SHARED ${JULIA_BINDING_SOURCES})
target_link_libraries(mpartjl PRIVATE mpart)

# Add an installation target for the julia bindings
install(TARGETS mpartjl DESTINATION julia/mpart)
install(DIRECTORY package/ DESTINATION julia/mpart)