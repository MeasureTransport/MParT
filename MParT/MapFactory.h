#ifndef MPART_MAPFACTORY_H
#define MPART_MAPFACTORY_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

namespace mpart{

    /**
     @brief 
     @param mset The multiindex set specifying which terms should be used in the multivariate expansion.
     */
    std::shared_ptr<ConditionalMapBase> CreateComponent(FixedMultiIndexSet<Kokkos::HostSpace> const& mset, 
                                                        std::unordered_map<std::string,std::string> options = std::unordered_map<std::string,std::string>());    
}

#endif 