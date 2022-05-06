#ifndef MPART_COMMONJULIAUTILITIES_H
#define MPART_COMMONJULIAUTILITIES_H

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"

#include "../../common/include/CommonUtilities.h"

// Note that this macro must be called in the jlcxx namespace, which is why there are two separate namespace blocks in this file
namespace jlcxx{
  template<typename T> struct IsSmartPointerType<mpart::binding::KokkosCustomPointer<T>> : std::true_type { };
  template<typename T> struct ConstructorPointerType<mpart::binding::KokkosCustomPointer<T>> { typedef std::shared_ptr<T> type; };
}

namespace mpart{
namespace binding{
/** Define a wrapper around Kokkos::Initialize that accepts a sequence of Cstrings. */
KokkosRuntime KokkosInit(jlcxx::ArrayRef<char*>);

/**
   @brief Adds the Kokkos bindings to the existing module m. 
   @param m CxxWrap.jl module
 */
void CommonUtilitiesWrapper(jlcxx::Module&);

} // namespace mpart
} // namespace binding



#endif // MPART_COMMONJULIAUTILITIES_H