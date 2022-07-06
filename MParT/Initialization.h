#ifndef MPART_INITIALIZATION_H
#define MPART_INITIALIZATION_H


#include <Kokkos_core.hpp>
#include <cstdlib>
#include <string>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace mpart{

    /** @defgroup InitializationHelpers 
    */

    /**
     @brief Call Kokkos::Finalize
     @ingroup InitializationHelpers
     @details the mpart::Initialize function will set add this function to the `atexit` list so that it is called at program termination.
     */
    void Finalize();

    // Simply holds a private variable isInitialized.  A static instance of this function is used in the GetInitializeStatusObject object.
    struct InitializeStatus{

        bool Get(){return isInitialized;};
        void Set(){isInitialized = true;} 

    private:
        bool isInitialized = false;
    };

    // Holds a static InitializeStatus object
    InitializeStatus& GetInitializeStatusObject();

    /**
     @brief Calls Kokkos::initialize if it hasn't been called yet.
     @ingroup InitializationHelpers
     @details 
     This function provides a thin wrapper around the Kokkos::initialize function that can be called multiple times without error.
     This function also adds Kokkos::finalize to `atexit` so that manually calling Kokkos::finalize is not necessary.   
     
     Note that Kokkos::initialize can only be called once and this function will print a warning message if called multiple times.  The warning 
     indicates that any changes to Kokkos parameters, like `--kokkos-threads` in subsequent calls will have no impact.  Only the 
     parameters passed the to the first mpart::Initialize call will be used.

     The warning messages can be silenced by setting the `MPART_WARNINGS` environment variable to `OFF`.

     @tparam Arguments 
     @param args Parameters to be passed on to Kokkos::initialize.
     */
    template<typename... Arguments>
    void Initialize(Arguments... args)
    {
        if(!GetInitializeStatusObject().Get()){
        
            // Initialize kokkos
            Kokkos::initialize(args...);

            // Make sure Kokkos::finalize() is called at program exit.
            std::atexit(&mpart::Finalize);

            // Update the state to remember that we called Initialize
            GetInitializeStatusObject().Set();

        }else{

            const char* warningStr = std::getenv("MPART_WARNINGS");

            bool shouldPrint = true;
            if(warningStr != nullptr){
                std::string warningFlag = warningStr;

                // Convert to lowercase so we catch OFF, off, Off, or any combination of capitalization.
                std::transform(warningFlag.begin(), warningFlag.end(), warningFlag.begin(), [](unsigned char c){ return std::tolower(c); });

                if(warningFlag=="off")
                    shouldPrint = false;
            }

            if(shouldPrint)
                std::cout << "WARNING: mpart::Initialize has already been called.  Any changes to runtime settings (e.g., \"--kokkos-threads\") will not go into effect."  << std::endl;
        }
    }

} // namespace mpart

#endif