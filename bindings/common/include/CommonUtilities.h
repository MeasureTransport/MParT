
#include <vector>
#include <string>

namespace mpart{
namespace binding{

    /** Define a wrapper around Kokkos::Initialize that accepts a vector of strings instead of argc and argv. */
    void Initialize(std::vector<std::string> args);

#ifdef MPART_ENABLE_GPU
    template<typename T>
    struct DeviceVector{
        StridedVector<T, Kokkos::DefaultExecutionSpace::memory_space> data;
    };

    template<typename T>
    struct DeviceMatrix{
        StridedMatrix<T, Kokkos::DefaultExecutionSpace::memory_space> data;
    };
#endif

} // namespace mpart
} // namespace binding