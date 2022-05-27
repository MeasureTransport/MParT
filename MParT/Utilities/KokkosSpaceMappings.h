#ifndef MPART_KOKKOSSPACEMAPPINGS_H
#define MPART_KOKKOSSPACEMAPPINGS_H

#include <Kokkos_Core.hpp>

namespace mpart{

    /** Used to convert Kokkos memory space type (e.g., Kokkos::CudaSpace) to an execution space that can access that memory.
        Note that mapping from memory space to device space is not unique.  This metaprogramming technique is only guaranteed 
        to return one of the possible execution spaces.
    */
    template<typename MemorySpace>
    struct MemoryToExecution{};

    template<> struct MemoryToExecution<Kokkos::HostSpace>{using Space = Kokkos::DefaultHostExecutionSpace;};

    #if defined(KOKKOS_ENABLE_CUDA)
    template<> struct MemoryToExecution<Kokkos::CudaSpace>{using Space = Kokkos::Cuda;};
    #endif
    
}

#endif 