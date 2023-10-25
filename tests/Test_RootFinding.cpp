#include <catch2/catch_all.hpp>

#include "MParT/Utilities/RootFinding.h"

#include <Kokkos_Core.hpp>

using namespace mpart::RootFinding;
using namespace Catch;
using HostSpace = Kokkos::HostSpace;

void CheckFoundBounds(std::function<double(double)> f, double xlb, double xd, double xub, double ylb, double yd, double yub) {
    REQUIRE((xlb < xd && xd < xub));
    REQUIRE((ylb < yd && yd < yub));
    REQUIRE((ylb == f(xlb) && yub == f(xub)));
}

TEST_CASE( "RootFindingUtils", "[RootFindingUtils]") {
    SECTION("swapPair") {
        double x1 = 1., x2 = 2., y1 = 3., y2 = 4.;
        swapPair(x1, x2, y1, y2);
        REQUIRE((x1 == 2. && x2 == 1. && y1 == 4. && y2 == 3.));
    }
    auto identity = [](double x){return x;};
    auto sigmoid = [](double x){return 1./(std::exp(-x)+1);};
    auto sigmoid_combo = [sigmoid](double x){return sigmoid(2*(x-1)) + 3*sigmoid(x*0.5) + 0.5*sigmoid(1.5*(x+1));};
    SECTION("FindBound lower linear") {
        double xd = -1.1;
        double yd = identity(xd);
        double xub = 2., yub = 2.;
        double xlb = 0., ylb = 0.;
        FindBound<HostSpace>(false, yd, identity, xub, yub, xlb, ylb, 10'000);
        CheckFoundBounds(identity, xlb, xd, xub, ylb, yd, yub);
    }
    SECTION("FindBound upper linear") {
        double xd = 1.1;
        double yd = identity(xd);
        double xub = 0., yub = 0.;
        double xlb = -2., ylb = -2.;
        FindBound<HostSpace>(true, yd, identity, xlb, ylb, xub, yub, 10'000);
        CheckFoundBounds(identity, xlb, xd, xub, ylb, yd, yub);
    }
    SECTION("FindBound lower sigmoid") {
        double xd = -0.5;
        double yd = sigmoid(xd);
        double xub =  2., yub = sigmoid(xub);
        double xlb =  0., ylb = sigmoid(xlb);
        FindBound<HostSpace>(false, yd, sigmoid, xub, yub, xlb, ylb, 10'000);
        CheckFoundBounds(sigmoid, xlb, xd, xub, ylb, yd, yub);
    }
    SECTION("FindBound upper sigmoid") {
        double xd = 0.5;
        double yd = sigmoid(xd);
        double xub =  0., yub = sigmoid(xub);
        double xlb = -2., ylb = sigmoid(xlb);
        FindBound<HostSpace>(true, yd, sigmoid, xlb, ylb, xub, yub, 10'000);
        CheckFoundBounds(sigmoid, xlb, xd, xub, ylb, yd, yub);
    }
    SECTION("Test Inverse Linear, low x0") {
        double xd = 0.5, yd = identity(xd);
        double x0 = 0.0, xtol = 1e-5, ftol = 1e-5;
        double xd_found = InverseSingleBracket<HostSpace>(yd, identity, x0, xtol, ftol);
        CHECK( xd_found == Approx(xd).epsilon(2*xtol));
    }
    SECTION("Test Inverse Linear, high x0") {
        double xd = 0.5, yd = identity(xd);
        double x0 = 1.0, xtol = 1e-5, ftol = 1e-5;
        double xd_found = InverseSingleBracket<HostSpace>(yd, identity, x0, xtol, ftol);
        CHECK( xd_found == Approx(xd).epsilon(2*xtol));
    }
    SECTION("Test Inverse Sigmoid, low x0") {
        double xd = 0.5, yd = sigmoid(xd);
        double x0 = 0.0, xtol = 1e-5, ftol = 1e-5;
        double xd_found = InverseSingleBracket<HostSpace>(yd, sigmoid, x0, xtol, ftol);
        CHECK( xd_found == Approx(xd).epsilon(2*xtol));
    }
    SECTION("Test Inverse Sigmoid, high x0") {
        double xd = 0.5, yd = sigmoid(xd);
        double x0 = 1.0, xtol = 1e-5, ftol = 1e-5;
        double xd_found = InverseSingleBracket<HostSpace>(yd, sigmoid, x0, xtol, ftol);
        CHECK( xd_found == Approx(xd).epsilon(2*xtol));
    }
    SECTION("Test Inverse Sigmoid Combo, low x0") {
        double xd = 0.5, yd = sigmoid_combo(xd);
        double x0 = -5.0, xtol = 1e-5, ftol = 1e-5;
        double xd_found = InverseSingleBracket<HostSpace>(yd, sigmoid_combo, x0, xtol, ftol);
        CHECK( xd_found == Approx(xd).epsilon(2*xtol));
    }
    SECTION("Test Inverse Sigmoid Combo, high x0") {
        double xd = 0.5, yd = sigmoid_combo(xd);
        double x0 = 5.0, xtol = 1e-5, ftol = 1e-5;
        double xd_found = InverseSingleBracket<HostSpace>(yd, sigmoid_combo, x0, xtol, ftol);
        CHECK( xd_found == Approx(xd).epsilon(2*xtol));
    }

}