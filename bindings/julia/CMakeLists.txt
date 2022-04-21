set(JULIA_BINDING_SOURCES
  src/Wrapper.cpp
#   src/CommonJuliaUtilities.cpp
)

include_directories(./include)
add_library(jlmpart SHARED ${JULIA_BINDING_SOURCES})
target_link_libraries(jlmpart PRIVATE mpart)

# Add an installation target for the julia bindings
install(TARGETS jlmpart DESTINATION julia/mpart)
install(DIRECTORY package/ DESTINATION julia/mpart)