#include "MParT/UnivariateExpansion.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/Sigmoid.h"
#include "catch2/catch_all.hpp"

using namespace mpart;
using namespace Catch::Matchers;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE("UnivariateExpansion") {
    using Basis_T = ProbabilistHermite;
    unsigned int maxOrder = 5;
    UnivariateExpansion<MemorySpace, Basis_T> expansion(maxOrder);
    Basis_T basis;
    REQUIRE(expansion.numCoeffs == maxOrder + 1);
    REQUIRE(expansion.inputDim == 1);
    REQUIRE(expansion.outputDim == 1);
    Kokkos::View<double*, MemorySpace> coeffs("coeffs", maxOrder + 1);
    Kokkos::deep_copy(coeffs, 0.0);
    expansion.WrapCoeffs(coeffs);
    unsigned int numPts = 20;
    Kokkos::View<double**, MemorySpace> points("points", 1, numPts);
    Kokkos::View<double**, MemorySpace> sens("sens", 1, numPts);
    double grid = numPts/2;
    for(int i = 0; i < numPts; i++) {
        points(0, i) = 3.*double(i-grid)/double(grid);
        sens(0, i) = i % 3 ? -0.5 : 0.5;
    }

    // Test Evaluation
    SECTION("Evaluation") {
        for(int order = 0; order <= maxOrder; order++) {
            coeffs(order) = 1.0;
            StridedMatrix<double, MemorySpace> eval = expansion.Evaluate(points);
            REQUIRE(eval.extent(0) == 1);
            REQUIRE(eval.extent(1) == numPts);
            for(int i = 0; i < numPts; i++) {
                double eval_ref = basis.Evaluate(order, points(0, i));
                CHECK_THAT(eval(0, i), WithinRel(eval_ref, 1e-14));
            }
            coeffs(order) = 0.0;
        }
    }
    SECTION("Gradient") {
        for(int order = 0; order <= maxOrder; order++) {
            coeffs(order) = 1.0;
            StridedMatrix<double, MemorySpace> grad = expansion.Gradient(points, sens);
            REQUIRE(grad.extent(0) == 1);
            REQUIRE(grad.extent(1) == numPts);
            for(int i = 0; i < numPts; i++) {
                double grad_ref = basis.Derivative(order, points(0, i)) * sens(0, i);
                CHECK_THAT(grad(0, i), WithinRel(grad_ref, 1e-14));
            }
            coeffs(order) = 0.0;
        }
    }
    SECTION("CoeffGrad") {
        StridedMatrix<double, MemorySpace> grad = expansion.CoeffGrad(points, sens);
        REQUIRE(grad.extent(0) == maxOrder+1);
        REQUIRE(grad.extent(1) == numPts);
        for(int order = 0; order <= maxOrder; order++) {
            for(int i = 0; i < numPts; i++) {
                double grad_ref = basis.Evaluate(order, points(0, i)) * sens(0, i);
                CHECK_THAT(grad(order, i), WithinRel(grad_ref, 1e-14));
            }
        }
    }
    SECTION("LogDeterminant") {
        Kokkos::deep_copy(coeffs, 1.0);
        StridedVector<double, MemorySpace> logDet = expansion.LogDeterminant(points);
        REQUIRE(logDet.extent(0) == numPts);
        unsigned int counter = 0;
        for(int i = 0; i < numPts; i++) {
            double det_ref = 0.;
            for(int order = 0; order <= maxOrder; order++)
                det_ref += basis.Derivative(order, points(0, i));
            if(det_ref > 0.) {
                double logDet_ref = std::log(det_ref);
                CHECK_THAT(logDet(i), WithinRel(logDet_ref, 1e-14));
                counter++;
            }
        }
        // Make sure logdet gets checked for a few points for good measure
        REQUIRE(counter > 5);
    }
    SECTION("LogDeterminantInputGrad") {
        Kokkos::deep_copy(coeffs, 1.0);
        StridedMatrix<double, MemorySpace> logDetGrad = expansion.LogDeterminantInputGrad(points);
        REQUIRE(logDetGrad.extent(0) == 1);
        REQUIRE(logDetGrad.extent(1) == numPts);
        for(int i = 0; i < numPts; i++) {
            double df = 0.;
            double d2f = 0.;
            for(int order = 0; order <= maxOrder; order++) {
                df += basis.Derivative(order, points(0, i));
                d2f += basis.SecondDerivative(order, points(0, i));
            }
            if(df > 0.) {
                double logDetGrad_ref = d2f/df;
                CHECK_THAT(logDetGrad(0, i), WithinRel(logDetGrad_ref, 1e-14));
            }
        }
    }
    SECTION("LogDeterminantCoeffGrad") {
        Kokkos::deep_copy(coeffs, 1.0);
        StridedMatrix<double, MemorySpace> logDetCoeffGrad = expansion.LogDeterminantCoeffGrad(points);
        REQUIRE(logDetCoeffGrad.extent(0) == maxOrder+1);
        REQUIRE(logDetCoeffGrad.extent(1) == numPts);
        for(int i = 0; i < numPts; i++) {
            double df = 0.;
            for(int order = 0; order <= maxOrder; order++) df += basis.Derivative(order, points(0, i));
            if(df <= 0.) continue;
            for(int order = 0; order <= maxOrder; order++) {
                double df_j = basis.Derivative(order, points(0, i));
                double logDetCoeffGrad_ref = df_j / df;
                CHECK_THAT(logDetCoeffGrad(order, i), WithinRel(logDetCoeffGrad_ref, 1e-14));
            }
        }
    }
}

TEST_CASE("UnivariateExpansion Inverse") {
    // Set up Sigmoid basis
    using Basis_T = Sigmoid1d<MemorySpace>;
    unsigned int num_sigmoid = 3;
    unsigned int maxOrder = num_sigmoid + 1 + 2; // 3 sigmoids + line + 2 edges
    unsigned int numParams = 2 + num_sigmoid*(num_sigmoid+1)/2;
    Kokkos::View<double*, MemorySpace> centers ("centers", numParams);
    Kokkos::View<double*, MemorySpace> widths ("widths", numParams);
    Kokkos::View<double*, MemorySpace> coeffs ("coeffs", maxOrder+1);
    Kokkos::deep_copy(widths, 1.0);
    double bound = 3;
    centers(0) = -bound; centers(1) = bound;
    unsigned int param_idx = 2;
    for(int order = 0; order <= num_sigmoid; order++) {
        double scaling = double(order)/2.;
        for(int j = 0; j < order; j++) {
            centers(param_idx++) = bound*(j-scaling)/scaling;
        }
    }
    Basis_T basis(centers, widths);
    UnivariateExpansion<MemorySpace, Basis_T> expansion(maxOrder, basis);
    Kokkos::deep_copy(coeffs, 1.0);
    expansion.WrapCoeffs(coeffs);
    unsigned int numPts = 100;
    Kokkos::View<double**, MemorySpace> points("points", 1, numPts);
    double grid = double(numPts)/2;
    for(int i = 0; i < numPts; i++) {
        points(0, i) = (bound+0.05)*double(i-grid)/double(grid);
    }
    StridedMatrix<double, MemorySpace> eval = expansion.Evaluate(points);
    Kokkos::View<double**, MemorySpace> prefix ("inverse prefix", 0, numPts);
    StridedMatrix<double, MemorySpace> inv = expansion.Inverse(prefix, eval);
    REQUIRE(inv.extent(0) == 1);
    REQUIRE(inv.extent(1) == numPts);
    double tol = 5e-6; // Comes from rootfinding
    for(int i = 0; i < numPts; i++) {
        if(std::abs(points(0,i)) > tol*1e-2) CHECK_THAT(inv(0, i), WithinRel(points(0, i), tol));
        else CHECK_THAT(inv(0, i), WithinAbs(points(0, i), 1e-10));
    }
}