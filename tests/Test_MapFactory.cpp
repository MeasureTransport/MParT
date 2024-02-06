#include <catch2/catch_all.hpp>

#include "MParT/MapFactory.h"
#include "MParT/ConditionalMapBase.h"
#include "MParT/TriangularMap.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

#include <unordered_map>
#include <string>

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "Testing map component factory", "[MapFactoryComponent]" ) {

    
    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;
    options.basisNorm = false;

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
    options.basisLB = -5;
    options.basisUB = 4;
    options.basisNorm = false;
    
    
    MapOptions options2;
    options2.basisType = BasisTypes::ProbabilistHermite;
    options2.basisNorm = false;
    
    unsigned int dim = 1;
    unsigned int maxDegree = 7;
    FixedMultiIndexSet<MemorySpace> mset(dim,maxDegree);

    SECTION("AdaptiveSimpson"){
        options.quadType = QuadTypes::AdaptiveSimpson;

        std::shared_ptr<ConditionalMapBase<MemorySpace>> linearized_map = MapFactory::CreateComponent<MemorySpace>(mset, options);
        
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateComponent<MemorySpace>(mset, options2);
        REQUIRE(linearized_map!=nullptr);
        REQUIRE(map!=nullptr);

        Kokkos::View<double*,MemorySpace> coeffs("Coefficients", map->numCoeffs);
        for(unsigned int i=0; i<map->numCoeffs; ++i)
            coeffs(i) = 1.0;
        map->SetCoeffs(coeffs);
        linearized_map->SetCoeffs(coeffs);

        unsigned int numPts = 5;
        Kokkos::View<double**,MemorySpace> pts("Points", dim, numPts);
        pts(0,0) = -6;
        pts(0,1) = -4.5;
        pts(0,2) = 0;
        pts(0,3) = 3.5;
        pts(0,4) = 4.5;
        
        
        Kokkos::View<double**, MemorySpace> linearized_evals = linearized_map->Evaluate(pts);
        Kokkos::View<double**, MemorySpace> evals = map->Evaluate(pts);

        for(unsigned int i=0; i<numPts; ++i){
            if((pts(0,i)<options.basisLB)||(pts(0,i)>options.basisUB)){
                CHECK( std::abs(linearized_evals(0,i) - evals(0,i))>1e-13);
            }else{
                CHECK( linearized_evals(0,i) == Approx(evals(0,i)).epsilon(1e-15));
            }
        }
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

TEST_CASE( "Testing factory method for single entry map, activeInd = 1", "[MapFactorySingleEntryMap 1]" ) {

    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;
    unsigned int dim = 7;
    unsigned int activeInd = 1;
    unsigned int maxDegree = 5;
    FixedMultiIndexSet<MemorySpace> mset(activeInd, maxDegree);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> comp = MapFactory::CreateComponent<MemorySpace>(mset, options);

    std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateSingleEntryMap<MemorySpace>(dim, activeInd, comp);

    REQUIRE(map != nullptr);
}

TEST_CASE( "Testing factory method for single entry map, activeInd = dim", "[MapFactorySingleEntryMap 2]" ) {

    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;
    unsigned int dim = 7;
    unsigned int activeInd = dim;
    unsigned int maxDegree = 5;
    FixedMultiIndexSet<MemorySpace> mset(activeInd, maxDegree);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> comp = MapFactory::CreateComponent<MemorySpace>(mset, options);

    std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateSingleEntryMap<MemorySpace>(dim, activeInd, comp);

    REQUIRE(map != nullptr);
}

TEST_CASE( "Testing factory method for single entry map, 1 < activeInd < dim", "[MapFactorySingleEntryMap 3]" ) {

    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;
    unsigned int dim = 7;
    unsigned int activeInd = 3;
    unsigned int maxDegree = 5;
    FixedMultiIndexSet<MemorySpace> mset(activeInd, maxDegree);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> comp = MapFactory::CreateComponent<MemorySpace>(mset, options);

    std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateSingleEntryMap<MemorySpace>(dim, activeInd, comp);

    REQUIRE(map != nullptr);
}

TEST_CASE( "Testing factory method for Sigmoid Component", "[MapFactorySigmoidComponent]" ) {

    MapOptions options;
    unsigned int inputDim = 7;
    unsigned int maxDegree = 5;
    unsigned int numCenters = 2 + maxDegree*(maxDegree+1)/2;
    Kokkos::View<double*, MemorySpace> centers("Centers", numCenters);
    double bound = 3.;
    centers(0) = -bound; centers(1) = bound;
    unsigned int center_idx = 2;
    for(int j = 0; j < maxDegree; j++){
        for(int i = 0; i <= j; i++){
            centers(center_idx) = -bound + (2*bound)*(i+1)/(j+2);
            center_idx++;
        }
    }
    options.basisType = BasisTypes::HermiteFunctions;
    std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> func = MapFactory::CreateSigmoidComponent<MemorySpace>(inputDim, centers, options);
    REQUIRE(func != nullptr);

    unsigned int numPts = 100;
    Kokkos::View<double**,MemorySpace> pts("Points", inputDim, numPts);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0,0}, {inputDim, numPts});
    Kokkos::parallel_for("Fill points", policy, KOKKOS_LAMBDA(const int i, const int j){
        pts(i,j) = double(j*i)/double((inputDim)*(numPts-1));
    });
    Kokkos::fence();
    // Checking example of Gradient
    StridedVector<double, MemorySpace> coeffs = func->Coeffs();
    Kokkos::deep_copy(coeffs, 1.0);
    Kokkos::View<double**, MemorySpace> eval = func->Evaluate(pts);
    CHECK(eval.extent(0)==1);
    Kokkos::View<double**, MemorySpace> sens ( "Sensitivities", 1, numPts);
    Kokkos::parallel_for("fill sensitivities", numPts, KOKKOS_LAMBDA(const int i){
        sens(0,i) = 1.0;
    });
    Kokkos::View<double**, MemorySpace> grad = func->Gradient(pts, sens);
    CHECK(grad.extent(0)==inputDim);
    CHECK(grad.extent(1)==numPts);
    double fd_step = 1e-6;
    for(int i = 0; i < inputDim; i++){
        Kokkos::parallel_for(numPts, KOKKOS_LAMBDA(const int j){
            if(i > 0) pts(i-1,j) -= fd_step;
            pts(i,j) += fd_step;
        });
        Kokkos::fence();
        Kokkos::View<double**, MemorySpace> eval2 = func->Evaluate(pts);
        for(int j = 0; j < numPts; j++){
            double fd = (eval2(0,j) - eval(0,j))/(fd_step);
            CHECK(grad(i,j) == Approx(fd).epsilon(20*fd_step));
        }
    }
    SECTION("Create Triangular Sigmoid Map") {
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> maps;
        for(int i = 1; i <= inputDim; i++){
            maps.push_back(MapFactory::CreateSigmoidComponent<MemorySpace>(i, centers, options));
        }
        auto map = std::make_shared<TriangularMap<MemorySpace>>(maps);
        REQUIRE(map != nullptr);
    }
}
