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
/** Define a wrapper around Kokkos::Initialize that accepts a python dictionary instead of argc and argv. */
KokkosRuntime KokkosInit(jlcxx::ArrayRef<char*>);

/**
   @brief Adds the pybind11 bindings to the existing module pybind11 module m. 
   @param m pybind11 module
 */
void CommonUtilitiesWrapper(jlcxx::Module&);

} // namespace mpart
} // namespace binding



#endif // MPART_COMMONJULIAUTILITIES_H