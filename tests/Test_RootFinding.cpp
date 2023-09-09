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
    SECTION("FindBound lower linear") {
        double xd = -1.;
        double yd = identity(xd);
        double xub = 2., yub = 2.;
        double xlb = 0., ylb = 0.;
        FindBound<HostSpace>(false, yd, identity, xlb, ylb, xub, yub, 10'000);
        CheckFoundBounds(identity, xlb, xd, xub, ylb, yd, yub);
    }
    SECTION("FindBound upper linear") {
        double xd = 1.;
        double yd = identity(xd);
        double xub = 0., yub = 0.;
        double xlb = -2., ylb = -2.;
        FindBound<HostSpace>(true, yd, identity, xub, yub, xlb, ylb, 10'000);
        CheckFoundBounds(identity, xlb, xd, xub, ylb, yd, yub);
    }
    SECTION("FindBound lower sigmoid") {
        double xd = -0.5;
        double yd = sigmoid(xd);
        double xub = 2., yub = 2.;
        double xlb = 0., ylb = 0.;
        FindBound<HostSpace>(false, yd, sigmoid, xlb, ylb, xub, yub, 10'000);
        CheckFoundBounds(sigmoid, xlb, xd, xub, ylb, yd, yub);
    }
    SECTION("FindBound upper sigmoid") {
        double xd = 0.5;
        double yd = sigmoid(xd);
        double xub = 0., yub = 0.;
        double xlb = -2., ylb = -2.;
        FindBound<HostSpace>(true, yd, sigmoid, xub, yub, xlb, ylb, 10'000);
        std::cout << "ylb = " << ylb << ", f(xlb) = " << sigmoid(xlb) << std::endl;
        CheckFoundBounds(sigmoid, xlb, xd, xub, ylb, yd, yub);
    }


}