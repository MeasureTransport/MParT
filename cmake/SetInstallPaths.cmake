

if(PYTHON_INSTALL_PREFIX)
  message(STATUS "PYTHON_INSTALL_PREFIX was set by user to be ${PYTHON_INSTALL_PREFIX}.")
else()
  if(NOT PYTHON_INSTALL_SUFFIX)
    message(STATUS "PYTHON_INSTALL_SUFFIX not specified.  Setting to \"python/mpart\".")
    set(PYTHON_INSTALL_SUFFIX python/mpart)
  endif()

  message(STATUS "PYTHON_INSTALL_PREFIX was not set by user, defaulting to ${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_SUFFIX}")
  set(PYTHON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_SUFFIX})
endif()

message(STATUS "Python packages will be installed to ${PYTHON_INSTALL_PREFIX}")

if(JULIA_INSTALL_PREFIX)
  message(STATUS "JULIA_INSTALL_PREFIX was set by user to be ${JULIA_INSTALL_PREFIX}")
else()
  message(STATUS "JULIA_INSTALL_PREFIX was not set by user, defaulting to ${CMAKE_INSTALL_PREFIX}/julia.")
  set(JULIA_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/julia)
endif()


# if(SKBUILD)
# set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib:\$ORIGIN/../..")
# endif()