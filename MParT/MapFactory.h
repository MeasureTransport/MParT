#ifndef MPART_MAPFACTORY_H
#define MPART_MAPFACTORY_H

#include "MParT/MapOptions.h"

#include "MParT/ConditionalMapBase.h"
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
    options.nugget = 1e-4;                              // Optional. Default = 0.0

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


        /** @brief Constructs a triangular map with MonotoneComponents for each block.  A total order multiindex
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
        @brief Constructs a "Single entry" map, which is an identity for all but one output dimension.  The active output dimension is defined by a given component.
        @param dim The dimension of the mapping
        @param activeInd Which output dimension is "active", i.e., not identity
        @param comp The component that will be used for the active output dimension.
        */
        template<typename MemorySpace>
        std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSingleEntryMap(unsigned int dim,
                                                                              unsigned int activeInd,
                                                                              std::shared_ptr<ConditionalMapBase<MemorySpace>> const &comp);

        /**
         * @brief Create a RectifiedMultivariateExpansion using sigmoids
         * 
         * @details Using the parameters described in \c`opts`, this function creates
         * a "rectified multivariate expansion" object that uses a collection of sigmoid functions
         * to map the input space to the output space. The pertinent options are
         * - \c`opts.basisType` : The type of basis function to use in the expansion for the first \f$d-1\f$ inputs
         * - \c`opts.posFuncType` : The type of positive function to use in the rectified basis @ref MultivariateExpansionWorker
         * - \c`opts.edgeWidth` : The width of the "edge terms" on the last input @ref Sigmoid1d
         * - \c`opts.sigmoidType` : The type of sigmoid function to use in the expansion @ref Sigmoid1d
         * 
         * By default, this constructs total-order multi-index sets. This is only creating a real-valued function.
         * To create a map, use `TriangularMap` function.
         * 
         * @tparam MemorySpace 
         * @param inputDim 
         * @param centers 
         * @param opts 
         * @return std::shared_ptr<ConditionalMapBase<MemorySpace>> 
         */
        template<typename MemorySpace>
        std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSigmoidComponent(
            unsigned int inputDim, StridedVector<const double, MemorySpace> centers,
            MapOptions opts);

        template<typename MemorySpace, std::enable_if_t<std::is_same_v<MemorySpace, Kokkos::HostSpace>, bool> = true>
        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> CreateSigmoidComponent(
            unsigned int inputDim, Eigen::Ref<const Eigen::RowVectorXd> centers,
            MapOptions opts) {
            StridedVector<const double, Kokkos::HostSpace> centersVec = ConstVecToKokkos<double, Kokkos::HostSpace>(centers);
            return CreateSigmoidComponent<Kokkos::HostSpace>(inputDim, centersVec, opts);
        }
        template<typename MemorySpace>
        std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSigmoidComponent(
            FixedMultiIndexSet<MemorySpace> mset_offdiag, FixedMultiIndexSet<MemorySpace> mset_diag,
            StridedVector<const double, MemorySpace> centers, MapOptions opts);

        template<typename MemorySpace>
        std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSigmoidTriangular(
            unsigned int inputDim, unsigned int outputDim,
            std::vector<StridedVector<const double, MemorySpace>> const& centers, MapOptions opts
        );

        template<typename MemorySpace>
        std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSigmoidTriangular(
            unsigned int inputDim, unsigned int outputDim,
            StridedMatrix<const double, MemorySpace> const& centers, MapOptions opts
        ) {
            std::vector<StridedVector<const double, MemorySpace>> centersVecs;
            for(unsigned int i = 0; i < centers.extent(1); i++){
                StridedVector<const double, MemorySpace> center_i = Kokkos::subview(centers, Kokkos::ALL(), i);
                centersVecs.push_back(center_i);
            }
            return CreateSigmoidTriangular<MemorySpace>(inputDim, outputDim, centersVecs, opts);
        }

        template<typename MemorySpace, std::enable_if_t<std::is_same_v<MemorySpace, Kokkos::HostSpace>, bool> = true>
        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> CreateSigmoidTriangular(
            unsigned int inputDim, unsigned int outputDim,
            Eigen::Ref<const Eigen::RowMatrixXd> const& centers, MapOptions opts
        ) {
            StridedMatrix<const double, Kokkos::HostSpace> centersMat = ConstRowMatToKokkos<double,Kokkos::HostSpace>(centers);
            return CreateSigmoidTriangular<Kokkos::HostSpace>(inputDim, outputDim, centersMat, opts);
        }

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