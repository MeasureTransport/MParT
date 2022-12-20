#include <Kokkos_Core.hpp>

namespace mpart{
#if defined(MPART_ENABLE_GPU)

// Only enable DeviceSpace if the DefaultExecutionSpace is a GPU space.
using DeviceSpace = std::enable_if<
    !std::is_same<Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace
    >::value, Kokkos::DefaultExecutionSpace::memory_space>::type;

#endif
} // namespace mpart