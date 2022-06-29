#ifndef MPART_MAPFACTORY_H
#define MPART_MAPFACTORY_H

#include "MParT/MapOptions.h"

#include "MParT/ConditionalMapBase.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

namespace mpart{

    namespace MapFactory{
        
        /**
        @brief 
        @param mset The multiindex set specifying which terms should be used in the multivariate expansion.
        */
        std::shared_ptr<ConditionalMapBase> CreateComponent(FixedMultiIndexSet<Kokkos::HostSpace> const& mset, 
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
        std::shared_ptr<ConditionalMapBase> CreateTriangular(unsigned int inputDim, 
                                                             unsigned int outputDim,
                                                             unsigned int totalOrder,
                                                             MapOptions options = MapOptions());
    } 
}

#endif 