#include <sstream>
#include <catch2/catch_all.hpp>

#include "MParT/TriangularMap.h"
#include "MParT/MapFactory.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Test serializing 3d triangular map from MonotoneComponents", "[Serialization_TriangularMap]" ) {

    MapOptions options1;
    options1.basisType = BasisTypes::ProbabilistHermite;
    options1.basisNorm = false;

    unsigned int numBlocks = 3;
    unsigned int maxDegree = 2;
    unsigned int extraInputs = 1;

    unsigned int coeffSize = 0;

    std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> blocks1(numBlocks);
    for(unsigned int i=0;i<numBlocks;++i){
        FixedMultiIndexSet<Kokkos::HostSpace> mset(i+extraInputs+1,maxDegree);
        coeffSize += mset.Size();

        blocks1.at(i) = MapFactory::CreateComponent<Kokkos::HostSpace>(mset, options1);
    }

    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> triMap1 = std::make_shared<TriangularMap<Kokkos::HostSpace>>(blocks1);

    Kokkos::View<double*,Kokkos::HostSpace> coeffs1("Coefficients", triMap1->numCoeffs);
    for(unsigned int i=0; i<triMap1->numCoeffs; ++i)
        coeffs(i) = 0.1*(i+1);
    triMap1->SetCoeffs(coeffs1);

    std::stringstream ss;

    SECTION("Check Serialization"){
        cereal::BinaryOutputArchive oarchive(ss);
        oarchive(MapOptions);
        oarchive(triMap1.inputDim, triMap1.outputDim, maxDegree);
        save(oarchive, triMap1.Coeffs())
        for(int i=0; i<numBlocks; ++i) {
            save(oarchive, triMap1.GetComponent(i)->GetMultiIndexSet());
        }
    }
    SECTION("Check Deserialization"){
        cereal::BinaryInputArchive iarchive(ss);
        MapOptions options2;
        iarchive(options2);
        REQUIRE(options1.basisType == options2.basisType);
        REQUIRE(options1.basisNorm == options2.basisNorm);
        unsigned int inputDim2, outputDim2, maxDegree2;
        iarchive(inputDim2, outputDim2, maxDegree2);
        REQUIRE(inputDim2 == triMap1.inputDim);
        REQUIRE(outputDim2 == triMap1.outputDim);
        REQUIRE(maxDegree2 == maxDegree);
        Kokkos::View<double*,Kokkos::HostSpace> coeffs2;
        load(iarchive, coeffs2);
        REQUIRE(coeffs1.extent(0) == coeffs2.extent(0));
        for(unsigned int i=0; i<coeffs1.extent(0); ++i)
            REQUIRE(coeffs1(i) == coeffs2(i));
        for(int i=0; i<numBlocks; ++i) {
            auto mset1 = triMap1.GetComponent(i)->GetMultiIndexSet();
            FixedMultiIndexSet<Kokkos::HostSpace> mset2;
            load(iarchive, mset2);
            REQUIRE(mset2.Size() == mset1.Size());
            auto maxDegrees1 = mset1.GetMaxDegrees();
            auto maxDegrees2 = mset2.GetMaxDegrees();
            REQUIRE(maxDegrees1.extent(0) == maxDegrees2.extent(0));
            for(unsigned int j=0; j<maxDegrees1.extent(0); ++j)
                REQUIRE(maxDegrees1(j) == maxDegrees2(j));
        }
    }
}