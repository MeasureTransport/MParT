#ifndef MPART_MAPFACTORY_H
#define MPART_MAPFACTORY_H

#include "MParT/MapOptions.h"

#include "MParT/ConditionalMapBase.h"
#include "MParT/TriangularMap.h"
#include "MParT/SummarizedMap.h"
//#include "MParT/DebugMap.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

#include <math.h>

#include <map>

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
            @brief Creates a square triangular map that is an identity in all but one output dimension

            @details Given a dimension \f$d\f$, an active index \f$i\f$, and a real-valued map \f$f:\mathbb{R}^d\rightarrow\mathbb{R}\f$,
            this function creates a map \f$T:\mathbb{R}^d\rightarrow\mathbb{R}^d\f$ such that \f$T_j(x) = x_j\f$ for all \f$j\neq i\f$ and
            \f$T_i(x) = f(x)\f$.

            @param dim The dimension of the map.
            @param activeInd The index of the component to be non-identity.
            @param comp The component placed the activeInd.

         */
        template<typename MemorySpace>
        std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSingleEntryMap(unsigned int dim,
                                                                              unsigned int activeInd,
                                                                              std::shared_ptr<ConditionalMapBase<MemorySpace>> const &comp);


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



        // template<typename MemorySpace>
        // std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateDebugMap(std::shared_ptr<ConditionalMapBase<MemorySpace>> const &comp) { return std::make_shared<DebugMap<MemorySpace>>(comp); }


        /**
        @brief Constructs a (generally) non-monotone multivariate expansion.
        @param outputDim The output dimension of the expansion.  Each output will be defined by the same multiindex set but will have different coefficients.
        @param mset The multiindex set specifying which terms should be used in the multivariate expansion.
        @param options Options specifying the 1d basis functions used in the parameterization.
        */
        template<typename MemorySpace>
        std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSingleEntryMap(unsigned int dim,
                                                                                     unsigned int activeInd,
                                                                                     std::shared_ptr<ConditionalMapBase<MemorySpace>> const &comp);

        /** This struct is used to map the options to functions that can create a map component with types corresponding
            to the options.
        */
        template<typename MemorySpace>
        struct CompFactoryImpl{
            typedef std::tuple<BasisTypes, bool, PosFuncTypes, QuadTypes> OptionsKeyType;
            typedef std::function<std::shared_ptr<ConditionalMapBase<MemorySpace>>(FixedMultiIndexSet<MemorySpace> const&, MapOptions options)> FactoryFunctionType;
            typedef std::map<OptionsKeyType, FactoryFunctionType> FactoryMapType;

            static FactoryFunctionType GetFactoryFunction(MapOptions opts)
            {
                bool isLinearized = (!isinf(opts.basisLB)) ||(!isinf(opts.basisUB));
                OptionsKeyType optionsKey(opts.basisType, isLinearized, opts.posFuncType, opts.quadType);

                auto factoryMap = GetFactoryMap();

                auto iter = factoryMap->find(optionsKey);
                if(iter == factoryMap->end())
                    throw std::runtime_error("Could not find registered factory method for given MapOptions.");

                return iter->second;
            }

            static std::shared_ptr<FactoryMapType> GetFactoryMap()
            {
                static std::shared_ptr<FactoryMapType> map;
                if( !map )
                    map = std::make_shared<FactoryMapType>();
                return map;
            }
        };

    }
}

#endif