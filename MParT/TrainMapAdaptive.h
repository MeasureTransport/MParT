#ifndef MPART_TRAINMAPADAPTIVE_H
#define MPART_TRAINMAPADAPTIVE_H

#include <vector>
#include "Utilities/ArrayConversions.h"
#include "MultiIndices/MultiIndexSet.h"
#include "MapOptions.h"
#include "TriangularMap.h"
#include "MapFactory.h"
#include "MapObjective.h"
#include "TrainMap.h"

namespace mpart {

// Options specifically for ATM algorithm, with map eval opts -> training opts-> ATM specific opts
struct ATMOptions: public MapOptions, public TrainOptions {
    unsigned int maxPatience = 10;
    unsigned int maxSize = 10;
    MultiIndex maxDegrees;
    std::string String() override {
        std::string md_str = maxDegrees.String();
        std::stringstream ss;
        ss << MapOptions::String() << "\n" << TrainOptions::String() << "\n";
        ss << "maxPatience = " << maxPatience << "\n";
        ss << "maxSize = " << maxSize << "\n";
        ss << "maxDegrees = " << maxDegrees.String();
        return ss.str();
    }
};

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> TrainMapAdaptive(std::vector<MultiIndexSet> &mset0,
    std::shared_ptr<MapObjective<MemorySpace>> objective,
    ATMOptions options);

// template<typename MemorySpace>
// std::shared_ptr<ConditionalMapBase<MemorySpace>> TrainMapAdaptive(StridedMatrix<double, MemorySpace> train_x, StridedMatrix<double, MemorySpace> test_x, ATMOptions options = ATMOptions()) {
//     unsigned int dim = train_x.extent(0);
//     MultiIndexSet mset0 = MultiIndexSet::CreateTotalOrder(dim, 0);
//     return TrainMapAdaptive(mset0, train_x, test_x, options);
// }

}


#endif // MPART_TRAINMAPADAPTIVE_H