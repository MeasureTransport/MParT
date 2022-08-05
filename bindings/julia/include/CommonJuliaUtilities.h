#ifndef MPART_COMMONJULIAUTILITIES_H
#define MPART_COMMONJULIAUTILITIES_H

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"
#include "jlcxx/stl.hpp"
#include "jlcxx/const_array.hpp"

#include "../../common/include/CommonUtilities.h"

namespace mpart{
namespace binding{

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

/**
 * @brief Adds ParameterizedFunctionBase bindings to the existing module m.
 * @param mod CxxWrap.jl module
 */
void ParameterizedFunctionBaseWrapper(jlcxx::Module&);

/**
 * @brief Adds ConditionalMapBase bindings to the existing module m.
 * @param mod CxxWrap.jl module
 */
void ConditionalMapBaseWrapper(jlcxx::Module&);

/**
 * @brief Adds TriangularMap bindings to the existing module m.
 * @param mod CxxWrap.jl module
 */
void TriangularMapWrapper(jlcxx::Module&);

/**
 * @brief Adds MapFactory bindings to the existing module m.
 * @param mod CxxWrap.jl module
 */
void MapFactoryWrapper(jlcxx::Module&);

#if defined(MPART_ENABLE_GPU)
void ConditionalMapBaseDeviceWrapper(jlcxx::Module&);
#endif

} // namespace binding
} // namespace mpart



#endif // MPART_COMMONJULIAUTILITIES_H