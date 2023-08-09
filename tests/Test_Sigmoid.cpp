#include <catch2/catch_all.hpp>
#include "MParT/Sigmoid.h"

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;


TEMPLATE_TEST_CASE("Sigmoid","[sigmoid]", Logistic) {
    SECTION("Initialization") {
        Kokkos::View<double*, MemorySpace> centers("Sigmoid Centers", 2);
        Kokkos::View<double*, MemorySpace> widths("Sigmoid Centers", 1);

        CHECK_THROWS_AS((Sigmoid<MemorySpace,TestType>(centers, widths)), std::invalid_argument);
    }
    SECTION("Single Sigmoid") {
        Kokkos::View<double*, MemorySpace> center("Sigmoid Centers", 1);
        Kokkos::View<double*, MemorySpace> width("Sigmoid Centers", 1);
        Kokkos::View<double*, MemorySpace> coeff("Sigmoid coeff", 1);
        center(0) = 0; width(0) = 1; coeff(0) = 1.0;
        Sigmoid<MemorySpace,TestType> Sigmoid (center, width);
        Sigmoid.SetCoeffs(coeff);
        Kokkos::View<double**, MemorySpace> evalPts("Input point", 1, 3);
        evalPts(0,0) = -100; evalPts(0,1) = 0.0; evalPts(0,2) = 100;
        StridedMatrix<double,MemorySpace> out = Sigmoid.Evaluate(evalPts);
        double approxTol = 1e-5;
        REQUIRE_THAT(out(0,0), Matchers::WithinAbs(0.0, approxTol));
        REQUIRE_THAT(out(0,1), Matchers::WithinAbs(0.5, approxTol));
        REQUIRE_THAT(out(0,2), Matchers::WithinAbs(1.0, approxTol));
    }
    SECTION("Multiple Sigmoids") {
        int N_Sigmoid = 3;
        Kokkos::View<double*, MemorySpace> centers("Sigmoid Centers", N_Sigmoid);
        Kokkos::View<double*, MemorySpace> widths("Sigmoid Centers", N_Sigmoid);
        Sigmoid<MemorySpace,TestType> Sigmoid (centers, widths);
        Kokkos::View<double*, MemorySpace> coeffs("Sigmoid Coefficients", N_Sigmoid);
        for(int j = 0; j < N_Sigmoid; j++) coeffs(j) = 0.5*(j+1);
        Sigmoid.WrapCoeffs(coeffs);
    }
}