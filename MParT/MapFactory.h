#ifndef MPART_MAPFACTORY_H
#define MPART_MAPFACTORY_H

#include "MParT/MapOptions.h"

#include "MParT/ConditionalMapBase.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

namespace mpart{

    /**
     @brief Namespace containing factory methods for constructing triangular maps and map components.

     Example usage:
     @code{.cpp}
    // First, set options defining the paramterization
    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite; // Optional. Default = BasisTypes::ProbabilistHermite
    options.posFuncType = PosFuncTypes::SoftPlus;       // Optional. Default = PosFuncTypes::SoftPlus
    options.quadType = QuadTypes::AdaptiveSimpson;      // Optional. Default = QuadTypes::AdaptiveSimpson
    options.quadAbsTol = 1e-6;                          // Optional. Default = 1e-6
    options.quadRelTol = 1e-6;                          // Optional. Default = 1e-6
    options.quadMaxSub = 10;                            // Optional. Default = 30
    options.contDeriv = true;                           // Optional. Default = true

    // Create a triangular map with these options
    unsigned int inDim = 4;
    unsigned int outDim = 3;
    unsigned int totalOrder = 3;
    auto map = MapFactory::CreateTriangular<Kokkos::HostSpace>(inDim, outDim, totalOrder, options);

     @endcode
     
     */
    namespace MapFactory{

        /**
        @brief
        @param mset The multiindex set specifying which terms should be used in the multivariate expansion.
        */
        template<typename MemorySpace>
        std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponent(FixedMultiIndexSet<MemorySpace> const& mset,
                                                                         MapOptions options = MapOptions());

        /**
            @brief Constructs a triangular map with MonotoneComponents for each block.  A total order multiindex
                   set is used to define the MonotoneComponent.

            @details For more control over the individual components, consider constructing components with
                     MapFactory::CreateComponent function and then manually constructing and instance of the
                     TriangularMap class.
            @param inputDim The dimension of the input to this map.  Will be the number of inputs passed to
                            the last component of the triangular map.
            @param outputDim The output dimension of the map.  Note that this must be less than or equal to
                             the input dimension.
            @param totalOrder The total order used to define the parameterization of each MonotoneComponent.
            @param options Additional options that will be passed on to CreateComponent to construct each MonotoneComponent.
         */
        template<typename MemorySpace>
        std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateTriangular(unsigned int inputDim,
                                                                          unsigned int outputDim,
                                                                          unsigned int totalOrder,
                                                                          MapOptions options = MapOptions());

        /**
        @brief Constructs a (generally) non-monotone multivariate expansion.
        @param outputDim The output dimension of the expansion.  Each output will be defined by the same multiindex set but will have different coefficients.
        @param mset The multiindex set specifying which terms should be used in the multivariate expansion.
        @param options Options specifying the 1d basis functions used in the parameterization.
        */
        template<typename MemorySpace>
        std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> CreateExpansion(unsigned int outputDim,
                                                                                FixedMultiIndexSet<MemorySpace> const& mset,
                                                                                MapOptions options = MapOptions());

    }
}

#endif