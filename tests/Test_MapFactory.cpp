#include <catch2/catch_all.hpp>

#include "MParT/MapFactory.h"
#include "MParT/ConditionalMapBase.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

#include <unordered_map>
#include <string>

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "Testing map component factory", "[MapFactoryComponent]" ) {

    
    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;

    unsigned int dim = 3;
    unsigned int maxDegree = 5;
    FixedMultiIndexSet<MemorySpace> mset(dim,maxDegree);

    SECTION("AdaptiveSimpson"){
        options.quadType = QuadTypes::AdaptiveSimpson;

        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateComponent<MemorySpace>(mset, options);
        REQUIRE(map!=nullptr);

        unsigned int numPts = 100;
        Kokkos::View<double**,MemorySpace> pts("Points", dim, numPts);
        for(unsigned int i=0; i<numPts; ++i)
            pts(dim-1,i) = double(i)/double(numPts-1);

        Kokkos::View<double**, MemorySpace> eval = map->Evaluate(pts);
    }

    SECTION("ClenshawCurtis"){
        options.quadType = QuadTypes::ClenshawCurtis;
        
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateComponent<MemorySpace>(mset, options);
        REQUIRE(map!=nullptr);

        unsigned int numPts = 100;
        Kokkos::View<double**,MemorySpace> pts("Points", dim, numPts);
        for(unsigned int i=0; i<numPts; ++i)
            pts(dim-1,i) = double(i)/double(numPts-1);

        Kokkos::View<double**, MemorySpace> eval = map->Evaluate(pts);
    }

    SECTION("AdaptiveClenshawCurtis"){
        options.quadType = QuadTypes::AdaptiveClenshawCurtis;
        
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateComponent<MemorySpace>(mset, options);
        REQUIRE(map!=nullptr);

        unsigned int numPts = 100;
        Kokkos::View<double**,MemorySpace> pts("Points", dim, numPts);
        for(unsigned int i=0; i<numPts; ++i)
            pts(dim-1,i) = double(i)/double(numPts-1);

        Kokkos::View<double**, MemorySpace> eval = map->Evaluate(pts);
    }
}


TEST_CASE( "Testing map component factory with linearized basis", "[MapFactoryLinearizedComponent]" ) {

    
    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;
    options.basisLB = -3;
    options.basisUB = 3;
    
    unsigned int dim = 3;
    unsigned int maxDegree = 5;
    FixedMultiIndexSet<MemorySpace> mset(dim,maxDegree);

    SECTION("AdaptiveSimpson"){
        options.quadType = QuadTypes::AdaptiveSimpson;

        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateComponent<MemorySpace>(mset, options);
        REQUIRE(map!=nullptr);

        unsigned int numPts = 100;
        Kokkos::View<double**,MemorySpace> pts("Points", dim, numPts);
        for(unsigned int i=0; i<numPts; ++i)
            pts(dim-1,i) = 5.0*double(i)/double(numPts-1);

        Kokkos::View<double**, MemorySpace> eval = map->Evaluate(pts);
    }

    SECTION("ClenshawCurtis"){
        options.quadType = QuadTypes::ClenshawCurtis;
        
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateComponent<MemorySpace>(mset, options);
        REQUIRE(map!=nullptr);

        unsigned int numPts = 100;
        Kokkos::View<double**,MemorySpace> pts("Points", dim, numPts);
        for(unsigned int i=0; i<numPts; ++i)
            pts(dim-1,i) = double(i)/double(numPts-1);

        Kokkos::View<double**, MemorySpace> eval = map->Evaluate(pts);
    }

    SECTION("AdaptiveClenshawCurtis"){
        options.quadType = QuadTypes::AdaptiveClenshawCurtis;
        
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateComponent<MemorySpace>(mset, options);
        REQUIRE(map!=nullptr);

        unsigned int numPts = 100;
        Kokkos::View<double**,MemorySpace> pts("Points", dim, numPts);
        for(unsigned int i=0; i<numPts; ++i)
            pts(dim-1,i) = double(i)/double(numPts-1);

        Kokkos::View<double**, MemorySpace> eval = map->Evaluate(pts);
    }
}


TEST_CASE( "Testing multivariate expansion factory", "[MapFactoryExpansion]" ) {

    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;

    unsigned int outDim = 5;
    unsigned int inDim = 3;
    unsigned int maxDegree = 5;
    FixedMultiIndexSet<MemorySpace> mset(inDim,maxDegree);

    std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> func = MapFactory::CreateExpansion<MemorySpace>(outDim, mset, options);
    REQUIRE(func!=nullptr);

    unsigned int numPts = 100;
    Kokkos::View<double**,MemorySpace> pts("Points", inDim, numPts);
    for(unsigned int i=0; i<numPts; ++i)
        pts(inDim-1,i) = double(i)/double(numPts-1);

    Kokkos::View<double**, MemorySpace> eval = func->Evaluate(pts);
    CHECK(eval.extent(0)==outDim);
    CHECK(eval.extent(1)==numPts);
}


TEST_CASE( "Testing factory method for triangular map", "[MapFactoryTriangular]" ) {

    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;


    std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateTriangular<MemorySpace>(4,3,5, options);

    REQUIRE(map != nullptr);
}

