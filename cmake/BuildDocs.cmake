
option(MPART_BUILD_EXAMPLES "If the tutorials should be included in the documentation." ON)

find_package(Doxygen)
find_package(Sphinx)


    
if(Sphinx_FOUND AND Doxygen_FOUND)

    message("SPHINX Executable = ${SPHINX_EXECUTABLE}")

    set(SPHINX_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/docs/docs)
    set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/mpart.doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/mpart.doxyfile @ONLY)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/src)

    # Copy the docs folder to the working folder so we can add to it
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/docs DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/docs)

    # Check to make sure we have the necessary tools installed (jupytext or docker)
    set(CAN_BUILD_EXAMPLES ${MPART_BUILD_EXAMPLES})
    if(${MPART_BUILD_EXAMPLES})
        if(${MPART_DOCKER_EXAMPLES})
            find_program(HAS_DOCKER "docker")
            if(HAS_DOCKER)
                message(STATUS "Found docker executable ${HAS_DOCKER}.")
                execute_process(COMMAND "docker" "info" RESULT_VARIABLE docker_res OUTPUT_QUIET ERROR_QUIET)
                
                if("${docker_res}" EQUAL 0)
                    message(STATUS "Can successfully run docker.  Now extracting notebooks from quay.io/measuretransport/mpart_examples to ${SPHINX_SOURCE}/source/tutorials/")

                    # Grab the notebooks from the docker image
                    execute_process(COMMAND "docker" "create" "quay.io/measuretransport/mpart_examples:latest" OUTPUT_VARIABLE container_id ERROR_QUIET)
                    string(STRIP ${container_id} container_id)
                    execute_process(COMMAND "docker" "cp" "${container_id}:/home/bayes/examples/python/" "${SPHINX_SOURCE}/source/tutorials/" OUTPUT_QUIET ERROR_QUIET)
                    execute_process(COMMAND "docker" "rm" "${container_id}" OUTPUT_QUIET ERROR_QUIET)

                    # rename the notebooks to remove .nbconvert.
                    file(GLOB MPART_PYTHON_EXAMPLES "${SPHINX_SOURCE}/source/tutorials/python/*.nbconvert.ipynb" )
                    foreach(file ${MPART_PYTHON_EXAMPLES})
                        string(REPLACE ".nbconvert" "" newfile ${file})
                        file(RENAME ${file} ${newfile})
                    endforeach()

                else()
                    message(WARNING "The docker daemon does not seem to be running. Will not be able to build tutorial documentation with MPART_DOCKER_EXAMPLES=ON.")
                    set(CAN_BUILD_EXAMPLES OFF)
                endif()

            else()
                message(WARNING "The option MPART_DOCKER_EXAMPLES=ON is on, but CMake could not find docker.  Will not be able to build tutorial documentation.")
                set(CAN_BUILD_EXAMPLES OFF)
            endif()
        else()
            find_program(HAS_JUPYTEXT "jupytext")
            if(HAS_JUPYTEXT)
                
                # If a local copy isn't defined, try to download
                if(NOT DEFINED MPART_EXAMPLES_DIR)
                    if(${MPART_FETCH_DEPS})
                        message(STATUS "Fetching examples.")
                        FetchContent_Declare(EXAMPLE_REPO 
                            GIT_REPOSITORY https://github.com/MeasureTransport/MParT-examples.git 
                            GIT_TAG main
                            CONFIGURE_COMMAND ""
                            BUILD_COMMAND ""
                            INSTALL_COMMAND ""
                        )

                        FetchContent_MakeAvailable(EXAMPLE_REPO)
                        set(MPART_EXAMPLES_DIR ${example_repo_SOURCE_DIR})
                        message("MPART_EXAMPLES_DIR = ${example_repo_SOURCE_DIR}")
                    else()
                        set(CAN_BUILD_EXAMPLES OFF)
                    endif()
                endif()

                # If we have the example available, we can process them
                if(${CAN_BUILD_EXAMPLES})
                    file(GLOB MPART_PYTHON_EXAMPLES "${MPART_EXAMPLES_DIR}/examples/python/*.py" )
                    file(COPY ${MPART_PYTHON_EXAMPLES} DESTINATION ${SPHINX_SOURCE}/source/tutorials/python)

                    # Run jupytext to convert py to ipynb
                    file(GLOB MPART_PYTHON_EXAMPLES "${SPHINX_SOURCE}/source/tutorials/python/*.py" )
                    foreach(file ${MPART_PYTHON_EXAMPLES})
                        execute_process(COMMAND "jupytext" "--set-formats" "ipynb,py" "${file}")
                    endforeach()
                endif()
                
            else()
                message(WARNING "Could not find jupytext, but this is needed to build the examples when MPART_DOCKER_EXAMPLES=OFF.  Will not be able to build tutorial documentation.")
                set(CAN_BUILD_EXAMPLES OFF)
            endif()
        endif()
    endif()
    

    
    

    add_custom_target(sphinx
                COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/mpart.doxyfile
                COMMAND ${SPHINX_EXECUTABLE} -b html
                # Tell Breathe where to find the Doxygen output
                -Dbreathe_projects.mpart=${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen/xml
                ${SPHINX_SOURCE} ${SPHINX_BUILD}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating documentation with Sphinx")

endif()