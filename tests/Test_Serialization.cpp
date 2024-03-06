#include <cereal/types/memory.hpp>

#include <sstream>
#include <catch2/catch_all.hpp>

#include "MParT/MonotoneComponent.h"
#include "MParT/PositiveBijectors.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/HermiteFunction.h"
#include "MParT/LinearizedBasis.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/Quadrature.h"
#include "MParT/TriangularMap.h"
#include "MParT/MapFactory.h"
#include "MParT/Utilities/Serialization.h"



using namespace mpart;
using namespace Catch;
using DefaultMonotoneComponent = MonotoneComponent<MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,Kokkos::HostSpace>, SoftPlus, AdaptiveSimpson<Kokkos::HostSpace>, Kokkos::HostSpace>;

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

TEST_CASE( "Test serializing polynomials.", "[Serialization]" ) {

    std::stringstream ss;
    double val1, val2;

    SECTION("Unnormalized"){
        // Save to string stream
        {
        ProbabilistHermite poly;
        val1 = poly.Evaluate(7, 0.5);

        cereal::BinaryOutputArchive archive(ss);
        archive(poly);
        }

        // Load from string stream
        {

        cereal::BinaryInputArchive archive(ss);

        ProbabilistHermite poly;
        archive(poly);

        val2 = poly.Evaluate(7, 0.5);
        CHECK(val1==val2);
        }
    }

    SECTION("Normalized"){
        // Save to string stream
        {
        ProbabilistHermite poly(true);
        val1 = poly.Evaluate(7, 0.5);

        cereal::BinaryOutputArchive archive(ss);
        archive(poly);
        }

        // Load from string stream
        {
        cereal::BinaryInputArchive archive(ss);

        ProbabilistHermite poly;
        archive(poly);

        val2 = poly.Evaluate(7, 0.5);
        CHECK(val1==val2);
        }
    }
}

TEST_CASE( "Test serializing quadrature rules.", "[Serialization]" ) {

    double val1, val2;
    auto f = [](double x, double* f){f[0]=exp(x);};

    SECTION("Clenshaw Curtis"){
        std::stringstream ss;
        
        {
            
        ClenshawCurtisQuadrature quad(4,1);
        val1 = 0.0;
        quad.Integrate(f, 0, 2.0, &val1);

        cereal::BinaryOutputArchive archive(ss);
        archive(quad);
        }

        {
        cereal::BinaryInputArchive archive(ss);

        ClenshawCurtisQuadrature quad;
        archive(quad);

        val2 = 0.0;
        quad.Integrate(f, 0, 2.0, &val2);
        CHECK(val1==val2);
        }
    }

    SECTION("Adaptive Simpson"){
        std::stringstream ss;
        val1 = val2 = 0.0;
        {
        AdaptiveSimpson quad(10,1, 1e-4, 1e-3, QuadError::First, 2);
        val1 = 0.0;
        quad.Integrate(f, 0, 2.0, &val1);

        cereal::BinaryOutputArchive archive(ss);
        archive(quad);
        }

        {
        cereal::BinaryInputArchive archive(ss);

        AdaptiveSimpson quad;
        archive(quad);

        val2 = 0.0;
        quad.Integrate(f, 0, 2.0, &val2);
        CHECK(val1==val2);
        }
    }

    SECTION("Adaptive Clenshaw Curtis"){
        std::stringstream ss;
        val1 = val2 = 0.0;
        {
        AdaptiveClenshawCurtis quad(2, 10, 1, 1e-4, 1e-3, QuadError::First);

        val1 = 0.0;
        quad.Integrate(f, 0, 2.0, &val1);

        cereal::BinaryOutputArchive archive(ss);
        archive(quad);
        }

        {
        cereal::BinaryInputArchive archive(ss);

        AdaptiveClenshawCurtis quad;
        archive(quad);
        
        val2 = 0.0;
        quad.Integrate(f, 0, 2.0, &val2);
        CHECK(val1==val2);
        }
    }
}

TEST_CASE("Test serialization of fixed multiindex set.", "[Serialization]"){

    const unsigned int dim = 2;
    const unsigned int maxOrder = 5;
    unsigned int length,size;
    std::stringstream ss;

    {
    FixedMultiIndexSet<Kokkos::HostSpace> mset(dim,maxOrder);
    length = mset.Length();
    size = mset.Size();

    cereal::BinaryOutputArchive archive(ss);
    archive(mset);
    }

    {
    FixedMultiIndexSet<Kokkos::HostSpace> mset;
    cereal::BinaryInputArchive archive(ss);
    archive(mset);

    CHECK(mset.Length()==length);
    CHECK(mset.Size()==size);
    }
}

TEST_CASE("Test serialization of multivariate expansion worker.", "[Serialization]"){


    unsigned int dim = 3;
    unsigned int maxDegree = 3;
    std::stringstream ss;

    Kokkos::View<double*,Kokkos::HostSpace> pt("Point", dim);
    pt(0) = 0.2;
    pt(1) = 0.1;
    pt(2) = 0.345;
    
    unsigned int cacheSize1, cacheSize2;
    {
    FixedMultiIndexSet<Kokkos::HostSpace> mset(dim, maxDegree); // Create a total order limited fixed multindex set
    MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,Kokkos::HostSpace> expansion(mset);

    cacheSize1 = expansion.CacheSize();
    CHECK(cacheSize1 == (maxDegree+1)*(2*dim+1));

    cereal::BinaryOutputArchive archive(ss);
    archive(expansion);

    }

    {
    MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,Kokkos::HostSpace> expansion;

    cereal::BinaryInputArchive archive(ss);
    archive(expansion);

    cacheSize2 = expansion.CacheSize();
    REQUIRE(cacheSize2 == (maxDegree+1)*(2*dim+1));

    // Build a cache
    std::vector<double> cache(cacheSize2);
    expansion.FillCache1(&cache[0], pt, DerivativeFlags::None);
    
    ProbabilistHermite poly1d;
    for(unsigned int d=0; d<dim-1;++d){
        for(unsigned int i=0; i<maxDegree+1; ++i){
            CHECK(cache[i + d*(maxDegree+1)] == Approx( poly1d.Evaluate(i,pt(d))).epsilon(1e-15) );
        }
    }
    }

}


TEST_CASE("Test serialization of monotone component.", "[Serialization]"){
    
    unsigned int maxDegree = 2;
    unsigned int dim = 1;

    // Create points evently space on [lb,ub]
    unsigned int numPts = 20;
    double lb = -5.0;
    double ub = 5.0;

    Kokkos::View<double**, Kokkos::HostSpace> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i)
        evalPts(0,i) = (i/double(numPts-1))*(ub-lb) - lb;

    Kokkos::View<double**,Kokkos::HostSpace> output1("Output1", dim,numPts);
    Kokkos::View<double**,Kokkos::HostSpace> output2("Output2", dim, numPts);


    std::stringstream ss;
    {
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);

        Kokkos::View<double*,Kokkos::HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(1) = 1.0; // Linear term = x ^1
        coeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
        coeffs(0) = 1.0 + coeffs(2); // Constant term = x^0

        MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,LinearizedBasis<HermiteFunction>>,Kokkos::HostSpace> expansion(mset);

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol,QuadError::First);

        //std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>>
        std::shared_ptr<mpart::ParameterizedFunctionBase<Kokkos::HostSpace>> comp = std::make_shared<MonotoneComponent<decltype(expansion), Exp, AdaptiveSimpson<Kokkos::HostSpace>, Kokkos::HostSpace>>(expansion, quad);
        comp->SetCoeffs(coeffs);

        output1 = comp->Evaluate(evalPts);

        cereal::BinaryOutputArchive archive(ss);
        archive(comp);
    }

    {   
        std::shared_ptr<mpart::ParameterizedFunctionBase<Kokkos::HostSpace>> comp;

        cereal::BinaryInputArchive archive(ss);
        archive(comp);

        output2 = comp->Evaluate(evalPts);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK(output1(0,i) == output2(0,i));
        }
    }
}


TEST_CASE("Test serialization of triangular map.", "[Serialization]"){

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

    unsigned int numSamps = 10;
    Kokkos::View<double**, Kokkos::HostSpace> in("Map Input", numBlocks+extraInputs, numSamps);
    for(unsigned int i=0; i<numBlocks+extraInputs; ++i){
        for(unsigned int j=0; j<numSamps; ++j){
            in(i,j) = double(i)/(numBlocks+extraInputs) + double(j)/numSamps;
        }
    }

    auto out1 = triMap1->Evaluate(in);

    SECTION("Check direct triangularMap Serialization"){
        {
            cereal::BinaryOutputArchive oarchive(ss);
            oarchive(triMap1);
        }
        {
            cereal::BinaryInputArchive iarchive(ss);
            std::shared_ptr<TriangularMap<Kokkos::HostSpace>> triMap2;
            iarchive(triMap2);

            CHECK(triMap1->inputDim == triMap2->inputDim);
            CHECK(triMap1->outputDim == triMap2->outputDim);
            CHECK(triMap1->numCoeffs == triMap2->numCoeffs);
            
            // Make sure the coefficients are the same 
            auto coeffs1 = triMap1->Coeffs();
            auto coeffs2 = triMap2->Coeffs();
            
            REQUIRE(coeffs1.size() == coeffs2.size());
            for(unsigned int i=0; i<coeffs1.size(); ++i){
                CHECK(coeffs1(i)==coeffs2(i));
            }

            // Test evaluation
            auto out2 = triMap1->Evaluate(in);
            REQUIRE(out1.extent(0)==out2.extent(0));
            REQUIRE(out1.extent(1)==out2.extent(1));
            for(unsigned int i=0; i<out1.extent(0); ++i){
                for(unsigned int j=0; j<out1.extent(1); ++j){
                    CHECK(fabs(out1(i,j)-out2(i,j))<1e-10);
                }
            }   
        }
    }

    SECTION("Check save/load functions."){
        ss.str("");
        triMap1->Save(ss);
        auto triMap2 = ParameterizedFunctionBase<Kokkos::HostSpace>::Load(ss);
    
        CHECK(triMap1->inputDim == triMap2->inputDim);
        CHECK(triMap1->outputDim == triMap2->outputDim);
        CHECK(triMap1->numCoeffs == triMap2->numCoeffs);
    
        // Make sure the coefficients are the same 
        auto coeffs1 = triMap1->Coeffs();
        auto coeffs2 = triMap2->Coeffs();
        
        REQUIRE(coeffs1.size() == coeffs2.size());
        for(unsigned int i=0; i<coeffs1.size(); ++i){
            CHECK(coeffs1(i)==coeffs2(i));
        }

        // Test evaluation
        auto out2 = triMap1->Evaluate(in);
        REQUIRE(out1.extent(0)==out2.extent(0));
        REQUIRE(out1.extent(1)==out2.extent(1));
        for(unsigned int i=0; i<out1.extent(0); ++i){
            for(unsigned int j=0; j<out1.extent(1); ++j){
                CHECK(fabs(out1(i,j)-out2(i,j))<1e-10);
            }
        }   
    }
}