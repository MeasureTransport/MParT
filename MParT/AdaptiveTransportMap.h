#ifndef MPART_ADAPTIVETRANSPORTMAP_H
#define MPART_ADAPTIVETRANSPORTMAP_H

#include <vector>
#include <nlopt/nlopt.hpp>
#include "Utilities/ArrayConversions.h"
#include "Utilities/MathFunctions.h"
#include "MultiIndex/MultiIndexSet.h"
#include "MapOptions.h"
#include "MapFactory.h"

namespace mpart {
double negativeLogLikelihood(const std::vector<double> &coeffs, std::vector<double> &grad, void *data);

enum class DensityTypes {
    StandardGaussian
};

// Options specifically for ATM algorithm
// 0. indicates that MParT uses the optimization's default value
struct ATMOptions: public MapOptions {
    int maxPatience = 10;
    int maxTerms = 10;
    ReferenceTypes densityType = ReferenceTypes::StandardGaussian;
    nlopt::algorithm alg = nlopt::LD_LBFGS;
    double opt_stopval = 0.;
    double opt_ftol_rel = 0.;
    double opt_xtol_rel = 0.;
    int opt_maxeval = 0;
    double opt_maxtime = 0.;
    bool verbose = false;
};

template<typename MemorySpace>
class ATMObjective {
    public:
    ATMObjective() = delete;
    ATMObjective(StridedMatrix<double, MemorySpace> x, StridedMatrix<double, MemorySpace> x_train, std::shared_ptr<ConditionalMapBase<MemorySpace>> map, ATMOptions options)

    double operator()(const std::vector<double> &coeffs, std::vector<double> &grad);
    void Gradient(const std::vector<double> &coeffs, std::vector<double> &grad);
    void SetMap(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {map_ = map;}
    double TestError(const std::vector<double> &coeffs);

    private:
    StridedMatrix<double, MemorySpace> x_;
    StridedMatrix<double, MemorySpace> x_train_
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    ATMOptions options_;
};

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> AdaptiveTransportMap(MultiIndexSet &mset0, StridedMatrix<double, MemorySpace> train_x, StridedMatrix<double, MemorySpace> test_x, ATMOptions options);

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> AdaptiveTransportMap(StridedMatrix<double, MemorySpace> train_x, StridedMatrix<double, MemorySpace> test_x, ATMOptions options = ATMOptions()) {
    unsigned int dim = train_x.extent(0);
    MultiIndexSet mset0 = MultiIndexSet::CreateTotalOrder(dim, 0);
    return AdaptiveTransportMap(mset0, train_x, test_x, options);
}

}


#endif // MPART_ADAPTIVETRANSPORTMAP_H