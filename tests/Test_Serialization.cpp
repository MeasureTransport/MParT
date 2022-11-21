#include <sstream>
#include <catch2/catch_all.hpp>

#include "MParT/MonotoneComponent.h"
#include "MParT/PositiveBijectors.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/Quadrature.h"
#include "MParT/TriangularMap.h"
#include "MParT/MapFactory.h"
#include "MParT/Utilities/Serialization.h"

using namespace mpart;
using namespace Catch;
using DefaultMonotoneComponent = MonotoneComponent<MultivariateExpansionWorker<ProbabilistHermite,Kokkos::HostSpace>, SoftPlus, AdaptiveSimpson<Kokkos::HostSpace>, Kokkos::HostSpace>;

TEST_CASE( "Test serializing Kokkos Views", "[Serialization]" ) {
    std::stringstream ss;
    SECTION("Check Serialization vectors") {
        Kokkos::View<double*, Kokkos::HostSpace> vec("vec", 10);
        Kokkos::View<double*, Kokkos::HostSpace> vec2("vec2", 10);
        Kokkos::deep_copy(vec, 1.0);
        Kokkos::deep_copy(vec2, 2.0);
        {
            cereal::BinaryOutputArchive archive(ss);
            archive(vec);
        }
        {
            cereal::BinaryInputArchive archive(ss);
            archive(vec2);
        }
        for (int i = 0; i < 10; i++) {
            REQUIRE(vec2(i) == 1.0);
        }
    }

    SECTION("Check Serialization Matrices: LayoutLeft") {
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> mat("mat", 10, 10);
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> mat2("mat2", 10, 10);
        Kokkos::deep_copy(mat, 1.0);
        Kokkos::deep_copy(mat2, 2.0);
        {
            cereal::BinaryOutputArchive archive(ss);
            archive(mat);
        }
        {
            cereal::BinaryInputArchive archive(ss);
            archive(mat2);
        }
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                REQUIRE(mat2(i, j) == 1.0);
            }
        }
    }

    SECTION("Check Serialization Matrices: LayoutRight") {
        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> mat("mat", 10, 10);
        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> mat2("mat2", 10, 10);
        Kokkos::deep_copy(mat, 1.0);
        Kokkos::deep_copy(mat2, 2.0);
        {
            cereal::BinaryOutputArchive archive(ss);
            archive(mat);
        }
        {
            cereal::BinaryInputArchive archive(ss);
            archive(mat2);
        }
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                REQUIRE(mat2(i, j) == 1.0);
            }
        }
    }
}

TEST_CASE( "Test serializing 3d triangular map from MonotoneComponents", "[Serialization]" ) {

    MapOptions options1;
    options1.basisType = BasisTypes::ProbabilistHermite;
    options1.basisNorm = false;

    unsigned int numBlocks = 3;
    unsigned int maxDegree = 2;
    unsigned int extraInputs = 1;

    unsigned int coeffSize = 0;

    std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> blocks1(numBlocks);
    std::vector<std::shared_ptr<FixedMultiIndexSet<Kokkos::HostSpace>>> msets1(numBlocks);
    for(unsigned int i=0;i<numBlocks;++i){
        msets1[i] = std::make_shared<FixedMultiIndexSet<Kokkos::HostSpace>>(i+extraInputs+1,maxDegree);
        coeffSize += msets1[i]->Size();

        blocks1.at(i) = MapFactory::CreateComponent<Kokkos::HostSpace>(*msets1[i], options1);
    }

    auto triMap1 = std::make_shared<TriangularMap<Kokkos::HostSpace>>(blocks1);

    Kokkos::View<double*,Kokkos::HostSpace> coeffs1("Coefficients", triMap1->numCoeffs);
    for(unsigned int i=0; i<triMap1->numCoeffs; ++i)
        coeffs1(i) = 0.1*(i+1);
    triMap1->SetCoeffs(coeffs1);

    std::stringstream ss;

    SECTION("Check Serialization"){
        cereal::BinaryOutputArchive oarchive(ss);
        oarchive(options1);
        oarchive(triMap1->inputDim, triMap1->outputDim, maxDegree);
        save(oarchive, triMap1->Coeffs());
        for(int i=0; i<numBlocks; ++i) {
            oarchive(msets1[i]);
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
        REQUIRE(inputDim2 == triMap1->inputDim);
        REQUIRE(outputDim2 == triMap1->outputDim);
        REQUIRE(maxDegree2 == maxDegree);
        Kokkos::View<double*,Kokkos::HostSpace> coeffs2;
        load(iarchive, coeffs2);
        REQUIRE(coeffs1.extent(0) == coeffs2.extent(0));
        for(unsigned int i=0; i<coeffs1.extent(0); ++i)
            REQUIRE(coeffs1(i) == coeffs2(i));
        for(int i=0; i<numBlocks; ++i) {
            auto mset1 = *msets1[i];
            std::shared_ptr<FixedMultiIndexSet<Kokkos::HostSpace>> mset2_ptr {nullptr};
            iarchive(mset2_ptr);
            auto mset2 = *mset2_ptr;
            REQUIRE(mset2.Size() == mset1.Size());
            auto maxDegrees1 = mset1.MaxDegrees();
            auto maxDegrees2 = mset2.MaxDegrees();
            REQUIRE(maxDegrees1.extent(0) == maxDegrees2.extent(0));
            for(unsigned int j=0; j<maxDegrees1.extent(0); ++j)
                REQUIRE(maxDegrees1(j) == maxDegrees2(j));
        }
    }
}