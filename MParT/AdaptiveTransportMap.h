#ifndef MPART_ADAPTIVETRANSPORTMAP_H
#define MPART_ADAPTIVETRANSPORTMAP_H

#include <vector>
#include "Utilities/ArrayConversions.h"
#include "MultiIndices/MultiIndexSet.h"
#include "MapOptions.h"
#include "TriangularMap.h"
#include "MapFactory.h"
#include "MapObjective.h"
#include "TrainMap.h"

namespace mpart {

// Options specifically for ATM algorithm
// 0. indicates that MParT uses the optimization's default value
struct ATMOptions: public MapOptions, public TrainOptions {
    int maxPatience = 10;
    int maxSize = 10;
};

template<typename MemorySpace,typename ObjectiveType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> AdaptiveTransportMap(std::vector<MultiIndexSet> &mset0,
    ObjectiveType &objective,
    ATMOptions options);

// template<typename MemorySpace>
// std::shared_ptr<ConditionalMapBase<MemorySpace>> AdaptiveTransportMap(StridedMatrix<double, MemorySpace> train_x, StridedMatrix<double, MemorySpace> test_x, ATMOptions options = ATMOptions()) {
//     unsigned int dim = train_x.extent(0);
//     MultiIndexSet mset0 = MultiIndexSet::CreateTotalOrder(dim, 0);
//     return AdaptiveTransportMap(mset0, train_x, test_x, options);
// }

}


#endif // MPART_ADAPTIVETRANSPORTMAP_H