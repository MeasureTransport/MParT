#include <catch2/catch_all.hpp>

#include "MParT/Utilities/LinearAlgebra.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing Matrix Sum", "[Mat+Mat]" ) {

    Kokkos::View<double**, Kokkos::HostSpace> A("A", 2, 2);
    Kokkos::View<double** ,Kokkos::HostSpace> B("B", 2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 0.5; A(1,1) = 1.0;

    B(0,0) = 1.0; B(0,1) = 2.0;
    B(1,0) = 0.125; B(1,1) = 0.5;

    Kokkos::View<double**, Kokkos::HostSpace> C = A + B;
    for(unsigned int i=0; i<A.extent(0); ++i){
        for(unsigned int j=0; j<A.extent(1); ++j){
            CHECK(C(i,j) == Approx(A(i,j)+B(i,j)).epsilon(1e-15).margin(1e-15));
        }
    }
}


TEST_CASE( "Testing Vecotr Sum", "[Vec+vec]" ) {

    Kokkos::View<double*, Kokkos::HostSpace> A("A", 2);
    Kokkos::View<double* ,Kokkos::HostSpace> B("B", 2);
    A(0) = 1.0;
    A(1) = 0.5;

    B(0) = 1.0; 
    B(1) = 0.125;

    Kokkos::View<double*, Kokkos::HostSpace> C = A + B;
    for(unsigned int i=0; i<A.extent(0); ++i){
        CHECK(C(i) == Approx(A(i)+B(i)).epsilon(1e-15).margin(1e-15));
    }
}


TEST_CASE( "Testing Linear Mat-Mat product", "[LinearAlgebra_MatMat]" ) {

    Kokkos::View<double**, Kokkos::HostSpace> A("A", 2, 2);
    Kokkos::View<double** ,Kokkos::HostSpace> B("B", 2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 0.5; A(1,1) = 1.0;

    B(0,0) = 1.0; B(0,1) = 2.0;
    B(1,0) = 0.125; B(1,1) = 0.5;

    auto eigA = KokkosToMat(A);
    auto eigB = KokkosToMat(B);
    Eigen::MatrixXd trueC;
    StridedMatrix<double,Kokkos::HostSpace> C;

    SECTION("No Transpose") {
        C = A*B;
        trueC = eigA*eigB;
        
        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());
        
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("A Transpose") {
        C = transpose(A)*B;
        trueC = eigA.transpose()*eigB;
        
        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());
        
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("B Transpose") {
        C = A*transpose(B);
        trueC = eigA*eigB.transpose();
        
        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());
        
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("Both Transpose") {
        C = transpose(A)*transpose(B);
        trueC = eigA.transpose() * eigB.transpose();
        
        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());
        
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("Extra Transpose") {
        C = transpose(A)*transpose(transpose(B));
        trueC = eigA.transpose() * eigB;
        
        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());
        
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
}


#if defined(MPART_ENABLE_GPU)

TEST_CASE( "Testing Linear Mat-Mat product on Device", "[LinearAlgebra_MatMatDevice]" ) {

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> A("A", 2, 2);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> B("B", 2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 0.5; A(1,1) = 1.0;

    B(0,0) = 1.0; B(0,1) = 2.0;
    B(1,0) = 0.125; B(1,1) = 0.5;

    auto dA = ToDevice<mpart::DeviceSpace>(A);
    auto dB = ToDevice<mpart::DeviceSpace>(B);
    
    auto eigA = KokkosToMat(A);
    auto eigB = KokkosToMat(B);
    Eigen::MatrixXd trueC;
    StridedMatrix<double,Kokkos::HostSpace> C;
    StridedMatrix<double,mpart::DeviceSpace> dC;

    SECTION("No Transpose") {
        dC = dA*dB;
        C = ToHost(dC);

        trueC = eigA*eigB;
        
        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());
        
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("A Transpose") {
        dC = transpose(dA)*dB;
        C = ToHost(dC);
        trueC = eigA.transpose()*eigB;
        
        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());
        
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("B Transpose") {
        dC = dA*transpose(dB);
        C = ToHost(dC);
        trueC = eigA*eigB.transpose();
        
        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());
        
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("Both Transpose") {
        dC = transpose(dA)*transpose(dB);
        C = ToHost(dC);
        trueC = eigA.transpose() * eigB.transpose();
        
        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());
        
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("Extra Transpose") {
        dC = transpose(dA)*transpose(transpose(dB));
        C = ToHost(dC);
        trueC = eigA.transpose() * eigB;
        
        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());
        
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
}

#endif