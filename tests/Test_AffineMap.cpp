#include <catch2/catch_all.hpp>

#include "MParT/AffineMap.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing Shift-only AffineMap", "[ShiftMap]" ) {

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

TEST_CASE( "Testing Linear-only AffineMap", "[LinearMap]" ) {

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


TEST_CASE( "Testing Rectangular AffineMap", "[Rectangular LinearMap]" ) {

    Kokkos::View<double**, Kokkos::HostSpace> A("A", 2,3);
    
    /* 
     A = [[2.0, 3.0, 1.0],
          [1.0, 1.0, 4.0]]
    */
    A(0,0) = 2.0;
    A(1,0) = 1.0;
    A(0,1) = 3.0;
    A(1,1) = 1.0;
    A(0,2) = 1.0;
    A(1,2) = 4.0;
    

    auto map = std::make_shared<AffineMap<Kokkos::HostSpace>>(A);
    REQUIRE(map->inputDim==3);
    REQUIRE(map->outputDim==2);

    unsigned int numPts = 3;
    Kokkos::View<double**, Kokkos::HostSpace> pts("Point", A.extent(1), numPts);
    for(unsigned int i=0; i<numPts; ++i){
        for(unsigned int d=0; d<A.extent(0);++d)
            pts(d,i) = double(i)/double(numPts-1);
    }

    // Check to make sure the log determinant is constant and what it should be
    Kokkos::View<double*, Kokkos::HostSpace> logDet = map->LogDeterminant(pts);
    for(unsigned int i=0; i<numPts; ++i){
        CHECK(logDet(i) == Approx(std::log(A(0,1)*A(1,2)-A(0,2)*A(1,1))).epsilon(1e-14));
    }

    // Test the forward evaluation
    Kokkos::View<double**, Kokkos::HostSpace> evals = map->Evaluate(pts);
    REQUIRE(evals.extent(0)==2);
    REQUIRE(evals.extent(1)==pts.extent(1));

    Kokkos::View<double**, Kokkos::HostSpace> pts2 = map->Inverse(pts, evals);
    REQUIRE(pts2.extent(0)==2);
    REQUIRE(pts2.extent(1)==pts.extent(1));

    for(unsigned int i=0; i<numPts; ++i){
        double trueOut1 = A(0,0)*pts(0,i) + A(0,1)*pts(1,i) + A(0,2)*pts(2,i);
        double trueOut2 = A(1,0)*pts(0,i) + A(1,1)*pts(1,i) + A(1,2)*pts(2,i);
        CHECK(evals(0,i)==Approx(trueOut1).epsilon(1e-14));
        CHECK(evals(1,i)==Approx(trueOut2).epsilon(1e-14));
        CHECK(pts2(0,i)==Approx(pts(1,i)).epsilon(1e-14));
        CHECK(pts2(1,i)==Approx(pts(2,i)).epsilon(1e-14));
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


#if defined(MPART_ENABLE_GPU)

TEST_CASE( "Testing Shift-only AffineMap on Device", "[DeviceShiftMap]" ) {

    Kokkos::View<double*, Kokkos::HostSpace> hb("b", 2);
    
    hb(0) = 1.0;
    hb(1) = 2.0;

    Kokkos::View<double*, mpart::DeviceSpace> db = ToDevice<mpart::DeviceSpace>(hb);


    auto map = std::make_shared<AffineMap<mpart::DeviceSpace>>(db);

    
    unsigned int numPts = 10;
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> hpts("Point", 2, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        hpts(0,i) = double(i)/double(numPts-1);
        hpts(1,i) = double(i)/double(numPts-1);
    }
    Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace> dpts = ToDevice<mpart::DeviceSpace>(hpts);

    // Check to make sure the log determinant is constant and what it should be
    Kokkos::View<double*, mpart::DeviceSpace> dlogDet = map->LogDeterminant(dpts);
    Kokkos::View<double*, Kokkos::HostSpace> hlogDet = ToHost(dlogDet);
    for(unsigned int i=0; i<numPts; ++i){
        CHECK(hlogDet(i) == Approx(0.0).epsilon(1e-14));
    }

    // Test the forward evaluation
    auto devals = map->Evaluate(dpts);
    auto hevals = ToHost(devals);

    auto dpts2 = map->Inverse(dpts, devals);
    auto hpts2 = ToHost(dpts2);

    for(unsigned int i=0; i<numPts; ++i){
        double trueOut1 = hpts(0,i) + hb(0);
        double trueOut2 = hpts(0,i) + hb(1);
        CHECK(hevals(0,i)==Approx(trueOut1).epsilon(1e-14));
        CHECK(hevals(1,i)==Approx(trueOut2).epsilon(1e-14));
        CHECK(hpts2(0,i)==Approx(hpts(0,i)).epsilon(1e-14));
        CHECK(hpts2(1,i)==Approx(hpts(1,i)).epsilon(1e-14));
    }

}

TEST_CASE( "Testing Linear-only AffineMap on Device", "[DeviceLinearMap]" ) {

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> hA("A", 2,2);
    
    hA(0,0) = 2.0;
    hA(1,0) = 1.0;
    hA(0,1) = 1.0;
    hA(1,1) = 4.0;

    auto dA = ToDevice<mpart::DeviceSpace>(hA);

    auto map = std::make_shared<AffineMap<mpart::DeviceSpace>>(dA);

    unsigned int numPts = 3;
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> hpts("Point", 2, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        hpts(0,i) = double(i)/double(numPts-1);
        hpts(1,i) = double(i)/double(numPts-1);
    }
    auto dpts = ToDevice<mpart::DeviceSpace>(hpts);


    //Check to make sure the log determinant is constant and what it should be
    auto dlogDet = map->LogDeterminant(dpts);
    auto hlogDet = ToHost(dlogDet);

    for(unsigned int i=0; i<numPts; ++i){
        CHECK(hlogDet(i) == Approx(std::log(hA(0,0)*hA(1,1)-hA(0,1)*hA(1,0))).epsilon(1e-14));
    }

    // Test the forward evaluation
    auto devals = map->Evaluate(dpts);
    auto hevals = ToHost(devals);

    // auto dpts2 = map->Inverse(dpts, devals);
    // auto hpts2 = ToHost(dpts2);
    
    // for(unsigned int i=0; i<numPts; ++i){
    //     double trueOut1 = hA(0,0)*hpts(0,i) + hA(0,1)*hpts(1,i);
    //     double trueOut2 = hA(1,0)*hpts(0,i) + hA(1,1)*hpts(1,i);
    //     CHECK(hevals(0,i)==Approx(trueOut1).epsilon(1e-14));
    //     CHECK(hevals(1,i)==Approx(trueOut2).epsilon(1e-14));
    //     CHECK(hpts2(0,i)==Approx(hpts(0,i)).epsilon(1e-14));
    //     CHECK(hpts2(1,i)==Approx(hpts(1,i)).epsilon(1e-14));
    // }

}


TEST_CASE( "Testing Rectangular AffineMap on Device", "[DeviceRectangularLinearMap]" ) {

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> hA("A", 2,3);
    
    /* 
     A = [[2.0, 3.0, 1.0],
          [1.0, 1.0, 4.0]]
    */
    hA(0,0) = 2.0;
    hA(1,0) = 1.0;
    hA(0,1) = 3.0;
    hA(1,1) = 1.0;
    hA(0,2) = 1.0;
    hA(1,2) = 4.0;
    
    auto dA = ToDevice<mpart::DeviceSpace>(hA);
    auto map = std::make_shared<AffineMap<mpart::DeviceSpace>>(dA);

    REQUIRE(map->inputDim==3);
    REQUIRE(map->outputDim==2);

    unsigned int numPts = 3;
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> hpts("Point", hA.extent(1), numPts);
    for(unsigned int i=0; i<numPts; ++i){
        for(unsigned int d=0; d<hA.extent(0);++d)
            hpts(d,i) = double(i)/double(numPts-1);
    }
    auto dpts = ToDevice<mpart::DeviceSpace>(hpts);

    // Check to make sure the log determinant is constant and what it should be
    Kokkos::View<double*, mpart::DeviceSpace> dlogDet = map->LogDeterminant(dpts);
    Kokkos::View<double*, Kokkos::HostSpace> hlogDet = ToHost(dlogDet);

    for(unsigned int i=0; i<numPts; ++i){
        CHECK(hlogDet(i) == Approx(std::log(hA(0,1)*hA(1,2)-hA(0,2)*hA(1,1))).epsilon(1e-14));
    }

    // Test the forward evaluation
    auto devals = map->Evaluate(dpts);
    auto hevals = ToHost(devals);
    REQUIRE(hevals.extent(0)==2);
    REQUIRE(hevals.extent(1)==hpts.extent(1));

    auto dpts2 = map->Inverse(dpts, devals);
    auto hpts2 = ToHost(dpts2);
    REQUIRE(hpts2.extent(0)==2);
    REQUIRE(hpts2.extent(1)==hpts.extent(1));

    for(unsigned int i=0; i<numPts; ++i){
        double trueOut1 = hA(0,0)*hpts(0,i) + hA(0,1)*hpts(1,i) + hA(0,2)*hpts(2,i);
        double trueOut2 = hA(1,0)*hpts(0,i) + hA(1,1)*hpts(1,i) + hA(1,2)*hpts(2,i);
        CHECK(hevals(0,i)==Approx(trueOut1).epsilon(1e-14));
        CHECK(hevals(1,i)==Approx(trueOut2).epsilon(1e-14));
        CHECK(hpts2(0,i)==Approx(hpts(1,i)).epsilon(1e-14));
        CHECK(hpts2(1,i)==Approx(hpts(2,i)).epsilon(1e-14));
    }

}
TEST_CASE( "Testing Full AffineMap on Device", "[DeviceFullAffineMap]" ) {

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> hA("A", 2,2);
    Kokkos::View<double*, Kokkos::HostSpace> hb("b", 2);
    
    hA(0,0) = 2.0;
    hA(1,0) = 1.0;
    hA(0,1) = 1.0;
    hA(1,1) = 4.0;

    hb(0) = 1.0;
    hb(1) = 2.0;

    auto dA = ToDevice<mpart::DeviceSpace>(hA);
    auto db = ToDevice<mpart::DeviceSpace>(hb);
    
    auto map = std::make_shared<AffineMap<mpart::DeviceSpace>>(dA,db);

    unsigned int numPts = 10;
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> hpts("Point", 2, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        hpts(0,i) = double(i)/double(numPts-1);
        hpts(1,i) = double(i)/double(numPts-1);
    }
    auto dpts = ToDevice<mpart::DeviceSpace>(hpts);

    // Check to make sure the log determinant is constant and what it should be
    Kokkos::View<double*, mpart::DeviceSpace> dlogDet = map->LogDeterminant(dpts);
    Kokkos::View<double*, Kokkos::HostSpace> hlogDet = ToHost(dlogDet);

    for(unsigned int i=0; i<numPts; ++i){
        CHECK(hlogDet(i) == Approx(std::log(hA(0,0)*hA(1,1)-hA(0,1)*hA(1,0))).epsilon(1e-14));
    }

    // Test the forward evaluation
    auto devals = map->Evaluate(dpts);
    auto hevals = ToHost(devals);

    auto dpts2 = map->Inverse(dpts, devals);
    auto hpts2 = ToHost(dpts2);
    
    for(unsigned int i=0; i<numPts; ++i){
        double trueOut1 = hA(0,0)*hpts(0,i) + hA(0,1)*hpts(1,i);
        double trueOut2 = hA(1,0)*hpts(0,i) + hA(1,1)*hpts(1,i);
        CHECK(hevals(0,i)==Approx(trueOut1).epsilon(1e-14));
        CHECK(hevals(1,i)==Approx(trueOut2).epsilon(1e-14));
        CHECK(hpts2(0,i)==Approx(hpts(0,i)).epsilon(1e-14));
        CHECK(hpts2(1,i)==Approx(hpts(1,i)).epsilon(1e-14));
    }

}


#endif 
