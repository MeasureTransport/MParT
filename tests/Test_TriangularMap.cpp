#include <catch2/catch_all.hpp>

#include "MParT/TriangularMap.h"
#include "MParT/MapFactory.h"

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "Testing 3d triangular map from MonotoneComponents", "[TriangularMap_MonotoneComponents]" ) {

    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;
    options.basisNorm = false;

    unsigned int numBlocks = 3;
    unsigned int maxDegree = 2;
    unsigned int extraInputs = 1;

    unsigned int coeffSize = 0;

    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(numBlocks);
    for(unsigned int i=0;i<numBlocks;++i){
        FixedMultiIndexSet<MemorySpace> mset(i+extraInputs+1,maxDegree);
        coeffSize += mset.Size();

        blocks.at(i) = MapFactory::CreateComponent<MemorySpace>(mset, options);
    }

    std::shared_ptr<TriangularMap<MemorySpace>> triMap = std::make_shared<TriangularMap<MemorySpace>>(blocks);

    CHECK(triMap->outputDim == numBlocks);
    CHECK(triMap->inputDim == numBlocks+extraInputs);
    CHECK(triMap->numCoeffs == coeffSize);


    Kokkos::View<double*,Kokkos::HostSpace> coeffs("Coefficients", triMap->numCoeffs);
    for(unsigned int i=0; i<triMap->numCoeffs; ++i)
        coeffs(i) = 0.1*(i+1);

    SECTION("Coefficients"){

        // Set the coefficients of the triangular map
        triMap->SetCoeffs(coeffs);

        // Now make sure that the coefficients of each block were set
        unsigned int cumCoeffInd = 0;
        for(unsigned int i=0; i<numBlocks; ++i){
            for(unsigned int j=0; j<blocks.at(i)->numCoeffs; ++j){
                CHECK(blocks.at(i)->Coeffs()(j) == coeffs(cumCoeffInd)); // Values of coefficients should be equal
                CHECK(&blocks.at(i)->Coeffs()(j) == &triMap->Coeffs()(cumCoeffInd)); // Memory location should also be the same (no copy)
                cumCoeffInd++;
            }
        }
    }


    unsigned int numSamps = 10;
    Kokkos::View<double**, Kokkos::HostSpace> in("Map Input", numBlocks+extraInputs, numSamps);
    for(unsigned int i=0; i<numBlocks+extraInputs; ++i){
        for(unsigned int j=0; j<numSamps; ++j){
            in(i,j) = double(i)/(numBlocks+extraInputs) + double(j)/numSamps;
        }
    }

    triMap->SetCoeffs(coeffs);
    auto out = triMap->Evaluate(in);

    SECTION("Evaluation"){

        for(unsigned int i=0; i<numBlocks; ++i){

            auto outBlock = blocks.at(i)->Evaluate(Kokkos::subview(in, std::make_pair(0,int(i+1+extraInputs)), Kokkos::ALL()));

            REQUIRE(outBlock.extent(1)==numSamps);
            REQUIRE(outBlock.extent(0)==1);
            for(unsigned int j=0; j<numSamps; ++j)
                CHECK( out(i,j) == Approx(outBlock(0,j)).epsilon(1e-6));
        }
    }


    SECTION("Inverse"){

        auto inv = triMap->Inverse(in,out);

        for(unsigned int i=0; i<numBlocks; ++i){
            for(unsigned int j=0; j<numSamps; ++j)
                CHECK( inv(i,j) == Approx(in(i+extraInputs,j)).epsilon(1e-6));
        }
    }

    SECTION("LogDeterminant"){
        auto logDet = triMap->LogDeterminant(in);

        REQUIRE(logDet.extent(0)==numSamps);
        Kokkos::View<double*, Kokkos::HostSpace> truth("True Log Det", numSamps);

        for(unsigned int i=0; i<numBlocks; ++i){
            auto blockLogDet = blocks.at(i)->LogDeterminant(Kokkos::subview(in, std::make_pair(0,int(i+1+extraInputs)), Kokkos::ALL()));

            for(unsigned int j=0; j<numSamps; ++j)
                truth(j) += blockLogDet(j);
        }

        for(unsigned int j=0; j<numSamps; ++j)
            CHECK(logDet(j) == Approx(truth(j)).epsilon(1e-10));

    }

    SECTION("CoeffGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivities", triMap->outputDim, numSamps);
        for(unsigned int j=0; j<numSamps; ++j){
            for(unsigned int i=0; i<triMap->outputDim; ++i){
                sens(i,j) = 1.0 + 0.1*i + j;
            }
        }

        Kokkos::View<double**,Kokkos::HostSpace> evals = triMap->Evaluate(in);
        Kokkos::View<double**,Kokkos::HostSpace> evals2;

        Kokkos::View<double**,Kokkos::HostSpace> coeffGrad = triMap->CoeffGrad(in, sens);

        REQUIRE(coeffGrad.extent(0)==triMap->numCoeffs);
        REQUIRE(coeffGrad.extent(1)==numSamps);

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<triMap->numCoeffs; ++i){
            coeffs(i) += fdstep;

            triMap->SetCoeffs(coeffs);
            evals2 = triMap->Evaluate(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){

                double fdDeriv = 0.0;
                for(unsigned int j=0; j<triMap->outputDim; ++j)
                    fdDeriv += sens(j,ptInd) * (evals2(j,ptInd)-evals(j,ptInd))/fdstep;

                CHECK( coeffGrad(i,ptInd) == Approx(fdDeriv).epsilon(1e-3));
            }
            coeffs(i) -= fdstep;
        }

    }


    SECTION("Input Gradient"){

        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivities", triMap->outputDim, numSamps);
        for(unsigned int j=0; j<numSamps; ++j){
            for(unsigned int i=0; i<triMap->outputDim; ++i){
                sens(i,j) = 1.0 + 0.1*i + j;
            }
        }

        Kokkos::View<double**,Kokkos::HostSpace> evals = triMap->Evaluate(in);
        Kokkos::View<double**,Kokkos::HostSpace> evals2;

        Kokkos::View<double**,Kokkos::HostSpace> inputGrad = triMap->Gradient(in, sens);

        REQUIRE(inputGrad.extent(0)==triMap->inputDim);
        REQUIRE(inputGrad.extent(1)==numSamps);

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<triMap->inputDim; ++i){
            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) += fdstep;

            evals2 = triMap->Evaluate(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){

                double fdDeriv = 0.0;
                for(unsigned int j=0; j<triMap->outputDim; ++j)
                    fdDeriv += sens(j,ptInd) * (evals2(j,ptInd)-evals(j,ptInd))/fdstep;

                CHECK( inputGrad(i,ptInd) == Approx(fdDeriv).epsilon(1e-3));
            }

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) -= fdstep;
        }

    }

    SECTION("LogDeterminantCoeffGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivities", triMap->outputDim, numSamps);
        for(unsigned int j=0; j<numSamps; ++j){
            for(unsigned int i=0; i<triMap->outputDim; ++i){
                sens(i,j) = 1.0 + 0.1*i + j;
            }
        }

        Kokkos::View<double**,Kokkos::HostSpace> detGrad = triMap->LogDeterminantCoeffGrad(in);
        REQUIRE(detGrad.extent(0)==triMap->numCoeffs);
        REQUIRE(detGrad.extent(1)==numSamps);

        Kokkos::View<double*,Kokkos::HostSpace> logDet = triMap->LogDeterminant(in);
        Kokkos::View<double*,Kokkos::HostSpace> logDet2;

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<triMap->numCoeffs; ++i){
            coeffs(i) += fdstep;

            triMap->SetCoeffs(coeffs);
            logDet2 = triMap->LogDeterminant(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                CHECK( detGrad(i,ptInd) == Approx((logDet2(ptInd)-logDet(ptInd))/fdstep).epsilon(1e-4));

            coeffs(i) -= fdstep;
        }

    }

    SECTION("Slice"){
        int sliceBegin = 1;
        int sliceEnd = 3;
        auto slice = triMap->Slice(sliceBegin, sliceEnd);
        REQUIRE(slice->inputDim == triMap->inputDim);
        REQUIRE(slice->outputDim == sliceEnd-sliceBegin);
        // Just test the evaluation
        for(unsigned int i=sliceBegin; i<sliceEnd; ++i){

            auto outBlock = blocks.at(i)->Evaluate(Kokkos::subview(in, std::make_pair(0,int(i+1+extraInputs)), Kokkos::ALL()));

            REQUIRE(outBlock.extent(1)==numSamps);
            REQUIRE(outBlock.extent(0)==1);
            for(unsigned int j=0; j<numSamps; ++j)
                CHECK( out(i,j) == Approx(outBlock(0,j)).epsilon(1e-6));
        }
    }

}
