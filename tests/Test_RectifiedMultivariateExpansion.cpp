#include <catch2/catch_all.hpp>
#include "MParT/MultiIndex/MultiIndexSet.h"
#include "MParT/MultiIndex/FixedMultiIndexSet.h"

using namespace mpart;
using namespace Catch;
using namespace Catch::Matchers;
using MemorySpace = Kokkos::HostSpace;

TEMPLATE_TEST_CASE("RectifiedMultivariateExpansion","[rmve]", SigmoidTypes::Logistic) {
    unsigned int maxOrder = 4;
    unsigned int dim = 3;
    FixedMultiIndexSet<MemorySpace> fmset_offdiag(dim-1, maxOrder);
    auto limiter = NonzeroDiagTotalOrderLimiter(maxOrder);
    MultiIndexSet mset_diag = MultiIndexSet::CreateTotalOrder(dim, maxOrder, limiter);
    FixedMultiIndexSet<MemorySpace> fmset_diag(mset_diag);
    
    const int num_sigmoids = 3;
    const int order = num_sigmoids+1+2;
    const int param_length = 2 + num_sigmoids*(num_sigmoids+1)/2;
    Kokkos::View<double*, MemorySpace> center("Sigmoid Center", param_length);
    Kokkos::View<double*, MemorySpace> width("Sigmoid Width", param_length);
    Kokkos::View<double*, MemorySpace> weight("Sigmoid Weight", param_length);
    double edge_bound = 3.;
    center(0) = -edge_bound; width(0) = 2*edge_bound/10; weight(0) = 1.;
    center(1) =  edge_bound; width(1) = 2*edge_bound/10; weight(1) = 1.;
    int param_idx = 2;
    for(int curr_order = 1; curr_order <= num_sigmoids; curr_order++) {
        for(int i = 0; i < curr_order; i++) {
            center(param_idx) = 4*(-(curr_order-1)/2 + i);
            width(param_idx) = 1/((double)i+1);
            weight(param_idx) = 1./curr_order;
            param_idx++;
        }
    }
    Sigmoid1d<MemorySpace,TestType> Sigmoids (center, width, weight);

    auto MVE_offdiag = MultivariateExpansion()
    SECTION("Initialization") {

    }
}