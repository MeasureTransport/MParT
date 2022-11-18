#include <catch2/catch_all.hpp>

#include "MParT/TriangularMap.h"
#include "MParT/MapFactory.h"

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "Test serializing 3d triangular map from MonotoneComponents", "[Serialization_TriangularMap]" ) {

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

    std::shared_ptr<ConditionalMapBase<MemorySpace>> triMap = std::make_shared<TriangularMap<MemorySpace>>(blocks);

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
}