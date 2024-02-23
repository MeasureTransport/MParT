#include "Test_Distributions_Common.h"
TEST_CASE( "Testing Custom Uniform Density", "[UniformDensity]" ) {
    auto density = std::make_shared<UniformDensity<Kokkos::HostSpace>>(2);
    unsigned int N_pts = 20;
    int a = -5; int b = 5;
    Kokkos::View<double**, Kokkos::HostSpace> pts ("pts", 2, N_pts);
    for(unsigned int j = 0; j < N_pts; ++j) {
        double lambda = ((double)j)/((double)N_pts-1);
        pts(0, j) = a + lambda*(b-a);
        pts(1, j) = a + lambda*(b-a);
    }

    auto pts_eigen = KokkosToMat(pts);
    StridedMatrix<const double, Kokkos::HostSpace> ptsConst = pts;

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

    SECTION("LogDensityInputGradImpl") {
        Kokkos::View<double**, Kokkos::HostSpace> output ("output", 2, N_pts);
        Kokkos::deep_copy(output, -3.);
        density->LogDensityInputGradImpl(pts, output);
        for(unsigned int j = 0; j < N_pts; ++j) {
            REQUIRE(output(0,j) == Approx(0.));
            REQUIRE(output(1,j) == Approx(0.));
        }
    }

    SECTION("LogDensityKokkos") {
        auto output = density->LogDensity(ptsConst);
        for(unsigned int j = 0; j < N_pts; ++j) {
            if(pts(0,j) >= 0. && pts(0,j) <= std::exp(1.)) {
                REQUIRE(output(j) == Approx(-2.0));
            } else {
                REQUIRE(output(j) == Approx(-std::numeric_limits<double>::infinity()));
            }
        }
    }

    SECTION("LogDensityEigen") {
        auto output_eigen = density->LogDensity(pts_eigen);
        for(unsigned int j = 0; j < N_pts; ++j) {
            if(pts(0,j) >= 0. && pts(0,j) <= std::exp(1.)) {
                REQUIRE(output_eigen(j) == Approx(-2.0));
            } else {
                REQUIRE(output_eigen(j) == Approx(-std::numeric_limits<double>::infinity()));
            }
        }
    }

    SECTION("LogDensityInputGradKokkos") {
        auto output = density->LogDensityInputGrad(ptsConst);
        for(unsigned int j = 0; j < N_pts; ++j) {
            REQUIRE(output(0,j) == Approx(0.));
            REQUIRE(output(1,j) == Approx(0.));
        }
    }

    SECTION("LogDensityInputGradEigen") {
        auto output_eigen = density->LogDensityInputGrad(pts_eigen);
        for(unsigned int j = 0; j < N_pts; ++j) {
            REQUIRE(output_eigen(0,j) == Approx(0.));
            REQUIRE(output_eigen(1,j) == Approx(0.));
        }
    }
}
