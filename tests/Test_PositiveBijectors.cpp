#include <catch2/catch_all.hpp>

#include <Kokkos_Core.hpp>

#include "MParT/PositiveBijectors.h"
#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing soft plus function.", "[SofPlus]" ) {

    // Test values near origin
    std::vector<double> xs{-1.0, -0.5, 0.0, 0.1, 1.0};

    for(auto& x : xs){
        double eval = SoftPlus::Evaluate(x);
        CHECK( eval == Approx(std::log(1.0+std::exp(x))) );
        CHECK( SoftPlus::Derivative(x) == Approx(std::exp(x) / (std::exp(x) + 1.0)) );
        CHECK( SoftPlus::SecondDerivative(x) == Approx(std::exp(x) / std::pow(std::exp(x) + 1.0, 2.0)) );
        CHECK( SoftPlus::Inverse(eval) == Approx(x) );
    }

    // Test extreme positive values
    std::vector<double> xes{50.0, 100.0};

    for(auto& x : xes){
        double eval = SoftPlus::Evaluate(x);
        CHECK( eval == Approx(x) );
        CHECK( SoftPlus::Derivative(x) == Approx(1.0) );
        CHECK( SoftPlus::Inverse(eval) == Approx(x) );
    }

}


#if defined(MPART_ENABLE_GPU)

TEST_CASE( "Testing soft plus function on device.", "[SofPlusDevice]" ) {
    const double floatTol = 1e-15;

    // Test values near origin
    Kokkos::View<double*,Kokkos::HostSpace> xs_host("host xs", 5);
    xs_host(0) = -1.0;
    xs_host(1) = -0.5;
    xs_host(2) = 0.0;
    xs_host(3) = 0.1;
    xs_host(4) = 1.0;

    auto xs_device = ToDevice<mpart::DeviceSpace>(xs_host);

    Kokkos::View<double*,mpart::DeviceSpace> ys_device("ys_device", xs_host.extent(0));
    Kokkos::View<double*,mpart::DeviceSpace> deriv_device("deriv_device", xs_host.extent(0));

    Kokkos::parallel_for(xs_host.size(), KOKKOS_LAMBDA(const size_t ind){
        ys_device(ind) = SoftPlus::Evaluate(xs_device(ind));
        deriv_device(ind) = SoftPlus::Derivative(xs_device(ind));
    });

    auto ys_host = ToHost(ys_device);
    auto deriv_host = ToHost<>(deriv_device);

    for(unsigned int i=0; i<xs_host.extent(0); ++i){
        CHECK(ys_host(i) == Approx(SoftPlus::Evaluate(xs_host(i))).epsilon(floatTol));
        CHECK(deriv_host(i) == Approx(SoftPlus::Derivative(xs_host(i))).epsilon(floatTol) );
    }
}
#endif