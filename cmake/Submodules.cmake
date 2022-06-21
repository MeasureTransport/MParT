

function(UpdateGitSubmodule subfolder)
    message(STATUS "Updating GIT Submodule ${subfolder}")

    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")

        # Update submodules as needed
        option(GIT_SUBMODULE "Check submodules during build" ON)
        if(GIT_SUBMODULE)
            message(STATUS "Submodule update")
            execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init -- ${CMAKE_CURRENT_SOURCE_DIR}/external/${subfolder})
        endif()
    endif()

endfunction(UpdateGitSubmodule)

function(PinSubmoduleVersion subfolder version)
    message(STATUS "Pinning GIT Submodule ${subfolder} to version ${version}")

    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/external/${subfolder} checkout tags/${version})
    endif()
endfunction(PinSubmoduleTag)