#include <catch2/catch_all.hpp>

#include "MParT/ConditionalMapBase.h"

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

class MyIdentityMap : public ConditionalMapBase<MemorySpace>{
public:
    MyIdentityMap(unsigned int dim, unsigned int numCoeffs, unsigned int dimOut = 0) : ConditionalMapBase(dim,!dimOut? dim : dimOut,numCoeffs){};

    virtual ~MyIdentityMap() = default;

    virtual void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                              StridedMatrix<double, MemorySpace>              output) override{
                                int a = this->inputDim - this->outputDim;
                                auto pts_view = Kokkos::subview(pts, std::make_pair(a, int(this->inputDim)), Kokkos::ALL());
                                Kokkos::deep_copy(output,pts_view);
                            };

    virtual void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                               StridedMatrix<const double, MemorySpace> const& sens,
                               StridedMatrix<double, MemorySpace>              output) override
    {
        assert(false);
    }

    virtual void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const&,
                                    StridedVector<double, MemorySpace>        output) override{
        for(unsigned int i=0; i<output.size(); ++i)
            output(i)=0.0;
    }

    virtual void InverseImpl(StridedMatrix<const double, MemorySpace> const&,
                             StridedMatrix<const double, MemorySpace> const& r,
                             StridedMatrix<double, MemorySpace>              output) override{Kokkos::deep_copy(output,r);};

    virtual void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                               StridedMatrix<const double, MemorySpace> const& sens,
                               StridedMatrix<double, MemorySpace>              output) override
    {
        assert(false);
    }


    virtual void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                             StridedMatrix<double, MemorySpace>              output) override
    {
        assert(false);
    }

    virtual void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                             StridedMatrix<double, MemorySpace>              output) override
    {
        assert(false);
    }

    // Only creates a slice of the tail of the input
    std::shared_ptr<ConditionalMapBase<MemorySpace>> Slice(int a, int b) override{
        return std::make_shared<MyIdentityMap>(this->inputDim, 0, b-a);
    }
};


TEST_CASE( "Testing coefficient functions of conditional map base class", "[ConditionalMapBaseCoeffs]" ) {

    unsigned int numCoeffs = 10;
    MyIdentityMap map(4,numCoeffs);

    CHECK(map.inputDim == 4);
    CHECK(map.outputDim == 4);

    SECTION("Using Kokkos"){

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
    MyIdentityMap map(dim,0);
    CHECK(map.inputDim == dim);
    CHECK(map.outputDim == dim);



    SECTION("Using Kokkos"){

        Kokkos::View<double**, Kokkos::HostSpace> pts("pts", dim, numPts);

        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = j;
            }
        }

        StridedMatrix<const double, Kokkos::HostSpace> ptsConst = pts;

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
    MyIdentityMap map(dim,0);
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


TEST_CASE( "Testing slicing evaluation of an identity conditional map", "[ConditionalMapBaseSlice]" ) {
    unsigned int dim = 7;
    unsigned int numPts = 5;
    MyIdentityMap map(dim,0);

    int a = 2;
    int b = 5;
    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mapSlice = map.Slice(a,b);

    int sliceOutdim = b-a;

    CHECK(map.inputDim == dim);
    CHECK(map.outputDim == dim);

    CHECK(mapSlice->inputDim == dim);
    CHECK(mapSlice->outputDim == sliceOutdim);

    SECTION("Using Kokkos"){

        Kokkos::View<double**, Kokkos::HostSpace> pts("pts", dim, numPts);

        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = i;
            }
        }

        Kokkos::View<double**, Kokkos::HostSpace> output = mapSlice->Evaluate(pts);

        REQUIRE(output.extent(0)==sliceOutdim);
        REQUIRE(output.extent(1)==numPts);

        int idx_start = map.inputDim - (b-a);

        for(unsigned int i=0; i<sliceOutdim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(output(i,j) == idx_start+i);
            }
        }
    }
}