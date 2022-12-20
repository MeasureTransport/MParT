/**
virtual void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> &output) = 0;

// LogDensity function with conversion from Eigen to Kokkos (and possibly copy to/from device).
Eigen::VectorXd LogDensity(Eigen::Ref<const Eigen::RowMatrixXd> const &pts);

template<typename AnyMemorySpace>
StridedVector<double, AnyMemorySpace> LogDensity(StridedMatrix<const double, AnyMemorySpace> const &X);

virtual void GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> &output) = 0;

Eigen::RowMatrixXd GradLogDensity(Eigen::Ref<const Eigen::RowMatrixXd> const &pts);

template<typename AnyMemorySpace>
StridedMatrix<double, AnyMemorySpace> GradLogDensity(StridedMatrix<const double, AnyMemorySpace> const &X);
*/
#include <catch2/catch_all.hpp>

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/DensityBase.h"

using namespace mpart;
using namespace Catch;

// Uniform density on [0,e]^2
class UniformDensity : public DensityBase<Kokkos::HostSpace> {

public:
void LogDensityImpl(StridedMatrix<const double, Kokkos::HostSpace> const &pts, StridedVector<double, Kokkos::HostSpace> output) override {
    double euler = std::exp(1.0);
    unsigned int N = pts.extent(1);
    Kokkos::parallel_for( "uniform log density", N, KOKKOS_LAMBDA (const int& j) {
        bool in_bounds1 = (pts(0, j) >= 0.0) && (pts(0, j) <= euler);
        bool in_bounds2 = (pts(1, j) >= 0.0) && (pts(1, j) <= euler);
        output(j) = in_bounds1 && in_bounds2 ? -2 : -std::numeric_limits<double>::infinity();
    });
}

void GradLogDensityImpl(StridedMatrix<const double, Kokkos::HostSpace> const &pts, StridedMatrix<double, Kokkos::HostSpace> output) override {
    unsigned int N = pts.extent(1);
    Kokkos::parallel_for( "uniform grad log density", N, KOKKOS_LAMBDA (const int& j) {
        output(0,j) = 0.;
        output(1,j) = 0.;
    });
}

};

TEST_CASE( "Testing Custom Uniform Density", "[UniformDensity]" ) {
    auto density = std::make_shared<UniformDensity>();
    unsigned int N_pts = 20;
    int a = -5; int b = 5;
    Kokkos::View<double**, Kokkos::HostSpace> pts ("pts", 2, N_pts);
    for(unsigned int j = 0; j < N_pts; ++j) {
        double lambda = ((double)j)/((double)N_pts-1);
        pts(0, j) = a + lambda*(b-a);
        pts(1, j) = a + lambda*(b-a);
    }

    SECTION("LogDensityImpl"){
        Kokkos::View<double*, Kokkos::HostSpace> output ("output", N_pts);
        density->LogDensityImpl(pts, output);
        for(unsigned int j = 0; j < N_pts; ++j) {
            if(pts(0,j) >= 0. && pts(0,j) <= std::exp(1.)) {
                REQUIRE(output(j) == Approx(-2.0));
            } else {
                REQUIRE(output(j) == Approx(-std::numeric_limits<double>::infinity()));
            }
        }
    }
}
