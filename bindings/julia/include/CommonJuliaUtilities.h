#ifndef MPART_COMMONJULIAUTILITIES_H
#define MPART_COMMONJULIAUTILITIES_H

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"

#include "../../common/include/CommonUtilities.h"

namespace mpart{
namespace binding{
    
/** Define a wrapper around Kokkos::Initialize that accepts a sequence of Cstrings. */
void Initialize(jlcxx::ArrayRef<char*>);

/**
   @brief Adds the Kokkos bindings to the existing module m. 
   @param m CxxWrap.jl module
 */
void CommonUtilitiesWrapper(jlcxx::Module&);

} // namespace mpart
} // namespace binding



#endif // MPART_COMMONJULIAUTILITIES_H