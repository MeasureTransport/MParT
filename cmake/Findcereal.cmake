# Find the cereal library
# Adapted from https://stackoverflow.com/questions/28629084/how-to-get-a-header-only-library-portably-from-version-control-using-cmake/28631206#28631206

find_package(PkgConfig)

# Allow the user can specify the include directory manually:
if(NOT EXISTS "${cereal_INCLUDE_DIR}")
  find_path(cereal_INCLUDE_DIR
    NAMES cereal/
    DOC "cereal library header files"
  )
endif()

if(EXISTS "${cereal_INCLUDE_DIR}")
  include(FindPackageHandleStandardArgs)
  mark_as_advanced(cereal_INCLUDE_DIR)
endif()

if(EXISTS "${cereal_INCLUDE_DIR}")
  set(cereal_FOUND ON)
else()
  set(cereal_FOUND OFF)
endif()