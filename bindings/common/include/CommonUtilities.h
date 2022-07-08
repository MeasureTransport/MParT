
#include <vector>
#include <string>

namespace mpart{
namespace binding{
    
    /** Define a wrapper around Kokkos::Initialize that accepts a vector of strings instead of argc and argv. */
    void Initialize(std::vector<std::string> args);

} // namespace mpart
} // namespace binding