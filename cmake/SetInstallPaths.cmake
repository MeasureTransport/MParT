

if(PYTHON_INSTALL_PREFIX)
  message(STATUS "PYTHON_INSTALL_PREFIX was set by user to be ${PYTHON_INSTALL_PREFIX}.")
else()
  message(STATUS "PYTHON_INSTALL_PREFIX was not set by user, defaulting to CMAKE_INSTALL_PREFIX/python.")
  set(PYTHON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/python)
endif()

message(STATUS "Python packages will be installed to ${PYTHON_INSTALL_PREFIX}.")
