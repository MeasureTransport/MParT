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

/**
 * @brief Both map and training options combined with special ATM options.
 *
 */
struct ATMOptions: public MapOptions, public TrainOptions {
    /** Maximum number of iterations that do not improve error */
    unsigned int maxPatience = 10;
    /** Maximum number of coefficients in final expansion (including ALL dimensions of map) */
    unsigned int maxSize = std::numeric_limits<unsigned int>::infinity();
    /** Multiindex representing the maximum degree in each input dimension */
    MultiIndex maxDegrees;

    /**
     * @brief Create a string representation of these options.
     *
     * @return std::string
     */
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

/**
 * @brief Adaptively discover new terms in coefficient basis to add to map using the ATM algorithm of Baptista, et al. 2022.
 *
 * @tparam MemorySpace Device or host space to work in
 * @param mset0 vector storing initial (minimal) guess of multiindex sets, corresponding to each dimension. Is changed in-place.
 * @param objective What this map should be adapted to fits
 * @return std::shared_ptr<ConditionalMapBase<MemorySpace>> New map according to specifications.
 */
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