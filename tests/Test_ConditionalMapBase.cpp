#include <catch2/catch_all.hpp>

#include "MParT/ConditionalMapBase.h"

using namespace mpart;
using namespace Catch;


 class MyIdentityMap : public ConditionalMapBase{
public:
    MyIdentityMap(unsigned int dim) : ConditionalMapBase(dim,dim){};
    
    virtual ~MyIdentityMap() = default;

    virtual void EvaluateImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts, 
                            Kokkos::View<double**, Kokkos::HostSpace>      &output) override{Kokkos::deep_copy(output,pts);};
    
    virtual void LogDeterminantImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts,
                                    Kokkos::View<double*, Kokkos::HostSpace>             &output) override{ 
        for(unsigned int i=0; i<output.size(); ++i)
            output(i)=0.0;
    }

    virtual void InverseImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& x1, 
                            Kokkos::View<const double**, Kokkos::HostSpace> const& r,
                            Kokkos::View<double**, Kokkos::HostSpace>      & output) override{Kokkos::deep_copy(output,r);};
};


TEST_CASE( "Testing coefficient functions of conditional map base class", "[ConditionalMapBaseCoeffs]" ) {

    MyIdentityMap map(4);
    CHECK(map.inputDim == 4);
    CHECK(map.outputDim == 4);

    SECTION("Using Kokkos"){
    
        unsigned int numCoeffs = 10;
        Kokkos::View<double*, Kokkos::HostSpace> coeffs("New Coeffs", numCoeffs);
        for(unsigned int i=0; i<numCoeffs; ++i)
            coeffs(i) = i;
        
        map.SetCoeffs(coeffs);
        CHECK(map.Coeffs().extent(0) == numCoeffs);

        for(unsigned int i=0; i<numCoeffs; ++i)
            CHECK(map.Coeffs()(i) == coeffs(i));

        coeffs(0) = 100;
        CHECK(map.Coeffs()(0) != coeffs(0));

        // Now check using a slice of the coefficients
        unsigned int start = 2;
        unsigned int end = 4;
        auto coeffSlice = Kokkos::subview(coeffs, std::make_pair(start, end));

        map.Coeffs() = coeffSlice;
        CHECK(coeffs.extent(0) == numCoeffs);
        CHECK(map.Coeffs().extent(0)==(end-start));

        for(unsigned int i=0; i<end-start; ++i)
            CHECK(map.Coeffs()(i)==coeffs(i+start));

        coeffs(start) = 1024;
        for(unsigned int i=0; i<end-start; ++i)
            CHECK(map.Coeffs()(i)==coeffs(i+start));

    }

    SECTION("Using Eigen"){

        unsigned int numCoeffs = 10;
        Eigen::VectorXd coeffs(numCoeffs);
        for(unsigned int i=0; i<numCoeffs; ++i)
            coeffs(i) = i;
        
        Kokkos::resize(map.Coeffs(), numCoeffs);
        map.CoeffMap() = coeffs;
        CHECK(map.Coeffs().extent(0) == numCoeffs);

        for(unsigned int i=0; i<numCoeffs; ++i){
            CHECK(map.Coeffs()(i) == coeffs(i));
            coeffs(i)++;
            CHECK(map.Coeffs()(i) != coeffs(i));
        }   

        map.SetCoeffs(coeffs);
        for(unsigned int i=0; i<numCoeffs; ++i){
            CHECK(map.Coeffs()(i) == coeffs(i));   
            coeffs(i)++;
            CHECK(map.Coeffs()(i) != coeffs(i));   
        }  

        map.SetCoeffs(coeffs);
        for(unsigned int i=0; i<numCoeffs; ++i){
            CHECK(map.Coeffs()(i) == coeffs(i)); 
            coeffs(i)++;  
            map.CoeffMap()(i)++;
            CHECK(map.Coeffs()(i) == coeffs(i));   
        }        
    }

}

TEST_CASE( "Testing evaluation of an identity conditional map", "[ConditionalMapBaseEvaluation]" ) {

    unsigned int dim = 4;
    unsigned int numPts = 100;
    MyIdentityMap map(dim);
    CHECK(map.inputDim == dim);
    CHECK(map.outputDim == dim);

    

    SECTION("Using Kokkos"){

        Kokkos::View<double**, Kokkos::HostSpace> pts("pts", dim, numPts);

        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = j;
            }
        }

        Kokkos::View<const double**, Kokkos::HostSpace> ptsConst = pts;
        
        Kokkos::View<double**, Kokkos::HostSpace> output = map.Evaluate(ptsConst);

        REQUIRE(output.extent(0)==dim);
        REQUIRE(output.extent(1)==numPts);
        
        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(output(i,j) == j);
            }
        }
    }

    SECTION("Using Eigen"){

        Eigen::RowMatrixXd pts(dim,numPts);
        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = j;
            }
        }

        Eigen::RowMatrixXd output;
        output = map.Evaluate(pts);

        REQUIRE(output.rows()==dim);
        REQUIRE(output.cols()==numPts);
        
        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(output(i,j) == j);
            }
        }        
    }

}


TEST_CASE( "Testing inverse evaluation of an identity conditional map", "[ConditionalMapBaseInverse]" ) {

    unsigned int dim = 4;
    unsigned int numPts = 100;
    MyIdentityMap map(dim);
    CHECK(map.inputDim == dim);
    CHECK(map.outputDim == dim);

    

    SECTION("Using Kokkos"){

        Kokkos::View<double**, Kokkos::HostSpace> pts("pts", dim, numPts);

        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = j;
            }
        }

        Kokkos::View<double**, Kokkos::HostSpace> output = map.Inverse(pts, pts);

        REQUIRE(output.extent(0)==dim);
        REQUIRE(output.extent(1)==numPts);
        
        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(output(i,j) == j);
            }
        }
    }

    SECTION("Using Eigen"){

        Eigen::RowMatrixXd pts(dim,numPts);
        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = j;
            }
        }

        Eigen::RowMatrixXd output;
        output = map.Inverse(pts,pts);

        REQUIRE(output.rows()==dim);
        REQUIRE(output.cols()==numPts);
        
        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(output(i,j) == j);
            }
        }        
    }

}