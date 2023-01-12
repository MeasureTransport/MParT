#ifndef MPART_ADAPTIVETRANSPORTMAP_H
#define MPART_ADAPTIVETRANSPORTMAP_H

#include <vector>
#include <nlopt.hpp>
#include "Utilities/ArrayConversions.h"
#include "Utilities/MathFunctions.h"
#include "MultiIndices/MultiIndexSet.h"
#include "MapOptions.h"
#include "TriangularMap.h"
#include "MapFactory.h"
#include "Distributions/PullbackDensity.h"
#include "Distributions/GaussianSamplerDensity.h"

namespace mpart {
double negativeLogLikelihood(const std::vector<double> &coeffs, std::vector<double> &grad, void *data);

enum class DensityTypes {
    StandardGaussian
};

// Options specifically for ATM algorithm
// 0. indicates that MParT uses the optimization's default value
struct ATMOptions: public MapOptions {
    int maxPatience = 10;
    int maxSize = 10;
    DensityTypes densityType = DensityTypes::StandardGaussian;
    std::string opt_alg = "LD_LBFGS";
    double opt_stopval = -std::numeric_limits<double>::infinity();
    double opt_ftol_rel = 1e-3;
    double opt_ftol_abs = 1e-3;
    double opt_xtol_rel = 1e-4;
    int opt_maxeval = 10;
    double opt_maxtime = 100.;
    bool verbose = false;
};

template<typename MemorySpace>
class ATMObjective {
    public:
    ATMObjective() = delete;
    ATMObjective(StridedMatrix<double, MemorySpace> x,
                 StridedMatrix<double, MemorySpace> x_test,
                 std::shared_ptr<ConditionalMapBase<MemorySpace>> map,
                 ATMOptions options = ATMOptions()):
                 x_(x), x_test_(x_test), map_(map), options_(options) {
        if(options.densityType != DensityTypes::StandardGaussian) {
            throw std::invalid_argument("ATMObjective<Kokkos::HostSpace>::ATMObjective: Currently only accepts Gaussian density");
        }
    }

    double operator()(unsigned int n, const double* coeffs, double* grad);
    void Gradient(unsigned int n, const double* coeffs, double* grad);
    void SetMap(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {map_ = map;}
    double TestError(StridedVector<const double, MemorySpace> coeffView);

    private:
    StridedMatrix<const double, MemorySpace> x_;
    StridedMatrix<const double, MemorySpace> x_test_;
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    ATMOptions options_;
};

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> AdaptiveTransportMap(std::vector<MultiIndexSet> &mset0, StridedMatrix<double, MemorySpace> train_x, StridedMatrix<double, MemorySpace> test_x, ATMOptions options);

// template<typename MemorySpace>
// std::shared_ptr<ConditionalMapBase<MemorySpace>> AdaptiveTransportMap(StridedMatrix<double, MemorySpace> train_x, StridedMatrix<double, MemorySpace> test_x, ATMOptions options = ATMOptions()) {
//     unsigned int dim = train_x.extent(0);
//     MultiIndexSet mset0 = MultiIndexSet::CreateTotalOrder(dim, 0);
//     return AdaptiveTransportMap(mset0, train_x, test_x, options);
// }

}


#endif // MPART_ADAPTIVETRANSPORTMAP_H