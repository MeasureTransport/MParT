#include <catch2/catch_all.hpp>

#include "MParT/TriangularMap.h"
#include "MParT/MapFactory.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing 3d triangular map from MonotoneComponents", "[TriangularMap_MonotoneComponents]" ) {


    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;

    unsigned int numBlocks = 3;
    unsigned int maxDegree = 2;
    unsigned int extraInputs = 1;

    unsigned int coeffSize = 0;

    std::vector<std::shared_ptr<ConditionalMapBase>> blocks(numBlocks);
    for(unsigned int i=0;i<numBlocks;++i){
        FixedMultiIndexSet mset(i+extraInputs+1,maxDegree);
        coeffSize += mset.Size();

        blocks.at(i) = MapFactory::CreateComponent(mset, options);
    }

    std::shared_ptr<ConditionalMapBase> triMap = std::make_shared<TriangularMap>(blocks);

    CHECK(triMap->outputDim == numBlocks);
    CHECK(triMap->inputDim == numBlocks+extraInputs);
    CHECK(triMap->numCoeffs == coeffSize);

    SECTION("Coefficients"){
        Kokkos::View<double*,Kokkos::HostSpace> coeffs("Coefficients", triMap->numCoeffs);
        for(unsigned int i=0; i<triMap->numCoeffs; ++i)
            coeffs(i) = 0.1*(i+1);
        
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
    auto out = triMap->Evaluate(in);
    
    SECTION("Evaluation"){
        
        for(unsigned int i=0; i<numBlocks; ++i){
            auto outBlock = blocks.at(i)->Evaluate(Kokkos::subview(in, std::pair(0,int(i+1+extraInputs)), Kokkos::ALL()));
            
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
            auto blockLogDet = blocks.at(i)->LogDeterminant(Kokkos::subview(in, std::pair(0,int(i+1+extraInputs)), Kokkos::ALL()));
            
            for(unsigned int j=0; j<numSamps; ++j)
                truth(j) += blockLogDet(j);
        }

        for(unsigned int j=0; j<numSamps; ++j)
            CHECK(logDet(j) == Approx(truth(j)).epsilon(1e-10));

    }
}
