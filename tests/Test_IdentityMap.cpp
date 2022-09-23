#include <catch2/catch_all.hpp>

#include "MParT/IdentityMap.h"

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "Testing identity map", "[IdentityMap]" ) {

    unsigned int inDim = 4;
    unsigned int outDim = 3;

    unsigned int numPts = 100;
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map = std::make_shared<IdentityMap<MemorySpace>>(inDim, outDim);
    CHECK(map->inputDim == inDim);
    CHECK(map->outputDim == outDim);


    Kokkos::View<double**, Kokkos::HostSpace> pts("pts", inDim, numPts);

    for(unsigned int i=0; i<inDim; ++i){
        for(unsigned int j=0; j<numPts; ++j){
            pts(i,j) = j;
        }
    }

    SECTION("Evaluate"){

        StridedMatrix<const double, Kokkos::HostSpace> ptsConst = pts;

        Kokkos::View<double**, Kokkos::HostSpace> output = map->Evaluate(ptsConst);

        REQUIRE(output.extent(0)==outDim);
        REQUIRE(output.extent(1)==numPts);

        for(unsigned int i=0; i<outDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(output(i,j) == j);
            }
        }
    }


    SECTION("Inverse"){

        StridedMatrix<const double, Kokkos::HostSpace> ptsConst = pts;


        Kokkos::View<double**, Kokkos::HostSpace> x1("x1", inDim-outDim, numPts);
        Kokkos::View<double**, Kokkos::HostSpace> r("r", outDim, numPts);

        for(unsigned int i=0; i<inDim-outDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                x1(i,j) = j;
            }
        }

        for(unsigned int i=0; i<outDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                r(i,j) = j;
            }
        }

        Kokkos::View<double**, Kokkos::HostSpace> inverse = map->Inverse(x1, r);
    
        
        REQUIRE(inverse.extent(0)==outDim);
        REQUIRE(inverse.extent(1)==numPts);

        for(unsigned int i=0; i<outDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(inverse(i,j) == j);
            }
        }
    }

    SECTION("LogDet"){
        StridedMatrix<const double, Kokkos::HostSpace> ptsConst = pts;
        Kokkos::View<double*, Kokkos::HostSpace> output = map->LogDeterminant(ptsConst);
    
        REQUIRE(output.size()==numPts);
        for(unsigned int j=0; j<numPts; ++j){
            CHECK(output(j) == 0);
        }
        
    }


    SECTION("Input Gradient"){

        StridedMatrix<const double, Kokkos::HostSpace> ptsConst = pts;
        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivities", map->outputDim, numPts);

        for(unsigned int j=0; j<numPts; ++j){
            for(unsigned int i=0; i<map->outputDim; ++i){
                sens(i,j) = 1.0 + 0.1*i + j;
            }
        }

        Kokkos::View<double**,Kokkos::HostSpace> evals = map->Evaluate(pts);
        Kokkos::View<double**,Kokkos::HostSpace> evals2;

        Kokkos::View<double**,Kokkos::HostSpace> inputGrad = map->Gradient(pts, sens);

        REQUIRE(inputGrad.extent(0)==map->inputDim);
        REQUIRE(inputGrad.extent(1)==numPts);

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<map->inputDim; ++i){
            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                pts(i,ptInd) += fdstep;

            evals2 = map->Evaluate(pts);

            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd){
                
                double fdDeriv = 0.0;
                for(unsigned int j=0; j<map->outputDim; ++j)
                    fdDeriv += sens(j,ptInd) * (evals2(j,ptInd)-evals(j,ptInd))/fdstep;

                CHECK( inputGrad(i,ptInd) == Approx(fdDeriv).margin(1e-3)); 
            }

            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                pts(i,ptInd) -= fdstep;
        }
        
    }



    SECTION("LogDeterminantInputGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> detGrad = map->LogDeterminantInputGrad(pts);
        REQUIRE(detGrad.extent(0)==map->inputDim);
        REQUIRE(detGrad.extent(1)==numPts);
        
        
        Kokkos::View<double*,Kokkos::HostSpace> logDet = map->LogDeterminant(pts);
        Kokkos::View<double*,Kokkos::HostSpace> logDet2;

        // Compare with finite differences
        double fdstep = 1e-6;
        for(unsigned int i=0; i<map->inputDim; ++i){

            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                pts(i,ptInd) += fdstep;

            logDet2 = map->LogDeterminant(pts);

            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd){
                CHECK( detGrad(i,ptInd) == Approx((logDet2(ptInd)-logDet(ptInd))/fdstep).margin(1e-3)); 
            }

            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                pts(i,ptInd) -= fdstep;
        }

    }



}

