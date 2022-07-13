#ifndef MPART_COMMONJULIAUTILITIES_H
#define MPART_COMMONJULIAUTILITIES_H

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"
#include "jlcxx/stl.hpp"
#include "jlcxx/const_array.hpp"

#include "../../common/include/CommonUtilities.h"

namespace mpart{
namespace binding{

/** Define a wrapper around Kokkos::Initialize that accepts a sequence of Cstrings. */
void Initialize(jlcxx::ArrayRef<char*>);

/**
   @brief Adds Kokkos bindings to the existing module m.
   @param mod CxxWrap.jl module
 */
void CommonUtilitiesWrapper(jlcxx::Module&);

/**
 * @brief Adds MultiIndex bindings to the existing module m.
 * @param mod CxxWrap.jl module
 */
void MultiIndexWrapper(jlcxx::Module&);

/**
 * @brief Adds MapOptions bindings to the existing module m.
 * @param mod CxxWrap.jl module
 */
void MapOptionsWrapper(jlcxx::Module&);

} // namespace mpart
} // namespace binding



#endif // MPART_COMMONJULIAUTILITIES_H