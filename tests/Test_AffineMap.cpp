#include <catch2/catch_all.hpp>

#include "MParT/AffineMap.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing Shifte AffineMap", "[ShiftMap]" ) {

    Kokkos::View<double*, Kokkos::HostSpace> b("b", 2);
    
    b(0) = 1.0;
    b(1) = 2.0;

    auto map = std::make_shared<AffineMap<Kokkos::HostSpace>>(b);

    unsigned int numPts = 10;
    Kokkos::View<double**, Kokkos::HostSpace> pts("Point", 2, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        pts(0,i) = double(i)/double(numPts-1);
        pts(1,i) = double(i)/double(numPts-1);
    }

    // Check to make sure the log determinant is constant and what it should be
    Kokkos::View<double*, Kokkos::HostSpace> logDet = map->LogDeterminant(pts);
    for(unsigned int i=0; i<numPts; ++i){
        CHECK(logDet(i) == Approx(0.0).epsilon(1e-14));
    }

    // Test the forward evaluation
    Kokkos::View<double**, Kokkos::HostSpace> evals = map->Evaluate(pts);

    Kokkos::View<double**, Kokkos::HostSpace> pts2 = map->Inverse(pts, evals);
    for(unsigned int i=0; i<numPts; ++i){
        double trueOut1 = pts(0,i) + b(0);
        double trueOut2 = pts(0,i) + b(1);
        CHECK(evals(0,i)==Approx(trueOut1).epsilon(1e-14));
        CHECK(evals(1,i)==Approx(trueOut2).epsilon(1e-14));
        CHECK(pts2(0,i)==Approx(pts(0,i)).epsilon(1e-14));
        CHECK(pts2(1,i)==Approx(pts(1,i)).epsilon(1e-14));
    }

}

TEST_CASE( "Testing Linear AffineMap", "[LinearMap]" ) {

    Kokkos::View<double**, Kokkos::HostSpace> A("A", 2,2);
    
    A(0,0) = 2.0;
    A(1,0) = 1.0;
    A(0,1) = 1.0;
    A(1,1) = 4.0;

    auto map = std::make_shared<AffineMap<Kokkos::HostSpace>>(A);

    unsigned int numPts = 3;
    Kokkos::View<double**, Kokkos::HostSpace> pts("Point", 2, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        pts(0,i) = double(i)/double(numPts-1);
        pts(1,i) = double(i)/double(numPts-1);
    }

    // Check to make sure the log determinant is constant and what it should be
    Kokkos::View<double*, Kokkos::HostSpace> logDet = map->LogDeterminant(pts);
    for(unsigned int i=0; i<numPts; ++i){
        CHECK(logDet(i) == Approx(std::log(A(0,0)*A(1,1)-A(0,1)*A(1,0))).epsilon(1e-14));
    }

    // Test the forward evaluation
    Kokkos::View<double**, Kokkos::HostSpace> evals = map->Evaluate(pts);

    Kokkos::View<double**, Kokkos::HostSpace> pts2 = map->Inverse(pts, evals);
    for(unsigned int i=0; i<numPts; ++i){
        double trueOut1 = A(0,0)*pts(0,i) + A(0,1)*pts(1,i);
        double trueOut2 = A(1,0)*pts(0,i) + A(1,1)*pts(1,i);
        CHECK(evals(0,i)==Approx(trueOut1).epsilon(1e-14));
        CHECK(evals(1,i)==Approx(trueOut2).epsilon(1e-14));
        CHECK(pts2(0,i)==Approx(pts(0,i)).epsilon(1e-14));
        CHECK(pts2(1,i)==Approx(pts(1,i)).epsilon(1e-14));
    }

}

TEST_CASE( "Testing Full AffineMap", "[FullAffineMap]" ) {

    Kokkos::View<double**, Kokkos::HostSpace> A("A", 2,2);
    Kokkos::View<double*, Kokkos::HostSpace> b("b", 2);
    
    A(0,0) = 2.0;
    A(1,0) = 1.0;
    A(0,1) = 1.0;
    A(1,1) = 4.0;

    b(0) = 1.0;
    b(1) = 2.0;

    auto map = std::make_shared<AffineMap<Kokkos::HostSpace>>(A,b);

    unsigned int numPts = 10;
    Kokkos::View<double**, Kokkos::HostSpace> pts("Point", 2, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        pts(0,i) = double(i)/double(numPts-1);
        pts(1,i) = double(i)/double(numPts-1);
    }

    // Check to make sure the log determinant is constant and what it should be
    Kokkos::View<double*, Kokkos::HostSpace> logDet = map->LogDeterminant(pts);
    for(unsigned int i=0; i<numPts; ++i){
        CHECK(logDet(i) == Approx(std::log(A(0,0)*A(1,1)-A(0,1)*A(1,0))).epsilon(1e-14));
    }

    // Test the forward evaluation
    Kokkos::View<double**, Kokkos::HostSpace> evals = map->Evaluate(pts);

    Kokkos::View<double**, Kokkos::HostSpace> pts2 = map->Inverse(pts, evals);
    for(unsigned int i=0; i<numPts; ++i){
        double trueOut1 = A(0,0)*pts(0,i) + A(0,1)*pts(1,i) + b(0);
        double trueOut2 = A(1,0)*pts(0,i) + A(1,1)*pts(1,i) + b(1);
        CHECK(evals(0,i)==Approx(trueOut1).epsilon(1e-14));
        CHECK(evals(1,i)==Approx(trueOut2).epsilon(1e-14));
        CHECK(pts2(0,i)==Approx(pts(0,i)).epsilon(1e-14));
        CHECK(pts2(1,i)==Approx(pts(1,i)).epsilon(1e-14));
    }

}
