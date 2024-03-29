

include_directories(include)
include_directories(external)
include_directories(external/mexplus)

set(MEX_SOURCE
    src/KokkosUtilities_mex.cpp
    src/ConditionalMap_mex.cpp
    src/MultiIndexSet_mex.cpp
    src/MultiIndex_mex.cpp
    src/FixedMultiIndexSet_mex.cpp
    src/ParameterizedFunctionBase_mex.cpp
    src/MexArrayConversions.cpp
    src/MexOptionsConversions.cpp
)

if(MPART_OPT)
    set(MEX_SOURCE ${MEX_SOURCE}
        src/MapObjective_mex.cpp
    )
endif()

matlab_add_mex(NAME MParT_ SRC ${MEX_SOURCE} LINK_TO mpart Kokkos::kokkos Eigen3::Eigen ${EXT_LIBRARIES})

# Add an installation target for the matlab bindings
install(TARGETS MParT_ DESTINATION matlab)
install(DIRECTORY mat/ DESTINATION matlab)