

include_directories(./include)
include_directories(external)

matlab_add_mex(NAME KokkosUtilities_ SRC src/KokkosUtilities_mex.cpp src/MexArrayConversions.cpp  LINK_TO mpart)
# matlab_add_mex(NAME FixedMultiIndexSet_ SRC src/FixedMultiIndexSet_mex.cpp src/MexArrayConversions.cpp LINK_TO mpart)
# matlab_add_mex(NAME TestExp_ SRC src/KokkosTesting.cpp src/MexArrayConversions.cpp LINK_TO mpart)
# matlab_add_mex(NAME BigLibrary_ SRC src/BigLibrary_mex.cpp src/MexArrayConversions.cpp LINK_TO mpart)
matlab_add_mex(NAME MParT_ SRC src/MParT_mex.cpp src/MexArrayConversions.cpp LINK_TO mpart)

# Add an installation target for the python bindings
install(TARGETS KokkosUtilities_ MParT_ DESTINATION matlab)
install(DIRECTORY mat/ DESTINATION matlab)