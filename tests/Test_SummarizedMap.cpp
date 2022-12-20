#include <catch2/catch_all.hpp>

#include "MParT/SummarizedMap.h"
#include "MParT/MapFactory.h"
#include "MParT/AffineFunction.h"
#include <MParT/Utilities/ArrayConversions.h>
using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "SummarizedMap", "[SummarizedMap_MonotoneComponent]" ) {

    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;
    options.basisNorm = false;

    unsigned int maxDegree = 2;
    unsigned int dim = 7;
    unsigned int lrcRank = 2;
    FixedMultiIndexSet<MemorySpace> mset(lrcRank+1, maxDegree);

    // make an affine function
    Eigen::RowMatrixXd A = Eigen::RowMatrixXd::Random(lrcRank, dim-1);
    Kokkos::View<double**, MemorySpace> viewA(A.data(), lrcRank, dim-1);
    std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> affineFunc = std::make_shared<AffineFunction<Kokkos::HostSpace>>(viewA);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> comp = MapFactory::CreateComponent<Kokkos::HostSpace>(mset, options);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> sumMap = std::make_shared<SummarizedMap<MemorySpace>>(affineFunc, comp);

    CHECK(sumMap->outputDim == 1);
    CHECK(sumMap->inputDim == dim);
    CHECK(sumMap->numCoeffs == comp->numCoeffs);


    Kokkos::View<double*,Kokkos::HostSpace> coeffs("Coefficients", sumMap->numCoeffs);
    for(unsigned int i=0; i<sumMap->numCoeffs; ++i)
        coeffs(i) = 0.1*(i+1);
        
    SECTION("Coefficients"){
        
        // Set the coefficients of the triangular map
        sumMap->SetCoeffs(coeffs);

        // Now make sure that the coefficients of each block were set
        unsigned int cumCoeffInd = 0;
        for(unsigned int i=0; i<sumMap->numCoeffs; ++i){
            CHECK(sumMap->Coeffs()(i) == 0.1*(i+1)); // Values of coefficients should be correct
            CHECK(sumMap->Coeffs()(i) == comp->Coeffs()(i)); // Values of coefficients should be equal to those of comp
            // CHECK(&sumMap->Coeffs()(i) == &comp->Coeffs()(i)); //
        
        }
    }


    unsigned int numSamps = 10;
    Kokkos::View<double**, Kokkos::HostSpace> in("Map Input", dim, numSamps);
    for(unsigned int i=0; i<dim; ++i){
        for(unsigned int j=0; j<numSamps; ++j){
            in(i,j) = double(i)/dim + double(j)/numSamps;
        }
    }

    Kokkos::View<double**, Kokkos::HostSpace> inToSummarize("Map Input until dim - 1", dim - 1, numSamps);
    for(unsigned int i=0; i<dim - 1; ++i){
        for(unsigned int j=0; j<numSamps; ++j){
            inToSummarize(i,j) = double(i)/dim + double(j)/numSamps;
        }
    }

    sumMap->SetCoeffs(coeffs);
    Kokkos::View<double**, Kokkos::HostSpace> out = sumMap->Evaluate(in);
    
    SECTION("Evaluation"){

        
        Kokkos::View<double**, Kokkos::HostSpace> summary = affineFunc->Evaluate(inToSummarize);
        Kokkos::View<double**, Kokkos::HostSpace> summaryAndLastDimOfIn("summaryAndLastDimOfIn",lrcRank + 1,numSamps);

        for(unsigned int i=0; i<lrcRank; ++i){
            for(unsigned int j=0; j<numSamps; ++j){
                summaryAndLastDimOfIn(i,j) = summary(i,j);
            }
        }

        for(unsigned int j=0; j<numSamps; ++j){
            summaryAndLastDimOfIn(lrcRank, j) = in(dim-1,j);
        }

        Kokkos::View<double**, Kokkos::HostSpace> out_ = comp->Evaluate(summaryAndLastDimOfIn);

        for(unsigned int i=0; i<out.extent(0); ++i){
            for(unsigned int j=0; j<numSamps; ++j)
                CHECK( out(i,j) == Approx(out_(i,j)).margin(1e-4));
        }
    }


    SECTION("Inverse"){

        Kokkos::View<double**, Kokkos::HostSpace> inv = sumMap->Inverse(in,out);

        for(unsigned int i=0; i<inv.extent(0); ++i){
            for(unsigned int j=0; j<numSamps; ++j)
                CHECK( inv(i,j) == Approx(in(dim-1,j)).margin(1e-4));
        }
    }

    SECTION("LogDeterminant"){

        Kokkos::View<double*, Kokkos::HostSpace> logDet = sumMap->LogDeterminant(in);
        Kokkos::View<double**, Kokkos::HostSpace> summary = affineFunc->Evaluate(inToSummarize);
        Kokkos::View<double**, Kokkos::HostSpace> summaryAndLastDimOfIn("summaryAndLastDimOfIn",lrcRank + 1,numSamps);

        for(unsigned int i=0; i<lrcRank; ++i){
            for(unsigned int j=0; j<numSamps; ++j)
                summaryAndLastDimOfIn(i,j) = summary(i,j);
        }

        for(unsigned int j=0; j<numSamps; ++j)
            summaryAndLastDimOfIn(lrcRank, j) = in(dim-1,j);
        

        Kokkos::View<double*, Kokkos::HostSpace> logDet_ = comp->LogDeterminant(summaryAndLastDimOfIn);

        for(unsigned int j=0; j<numSamps; ++j)
            CHECK( logDet(j) == Approx(logDet_(j)).margin(1e-4));

    }

    SECTION("CoeffGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivities", sumMap->outputDim, numSamps);
        for(unsigned int j=0; j<numSamps; ++j){
            for(unsigned int i=0; i<sumMap->outputDim; ++i){
                sens(i,j) = 1.0 + 0.1*i + j;
            }
        }

        Kokkos::View<double**,Kokkos::HostSpace> evals = sumMap->Evaluate(in);
        Kokkos::View<double**,Kokkos::HostSpace> evals2;

        Kokkos::View<double**,Kokkos::HostSpace> coeffGrad = sumMap->CoeffGrad(in, sens);

        REQUIRE(coeffGrad.extent(0)==sumMap->numCoeffs);
        REQUIRE(coeffGrad.extent(1)==numSamps);

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<sumMap->numCoeffs; ++i){
            coeffs(i) += fdstep;

            sumMap->SetCoeffs(coeffs);
            evals2 = sumMap->Evaluate(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                
                double fdDeriv = 0.0;
                for(unsigned int j=0; j<sumMap->outputDim; ++j)
                    fdDeriv += sens(j,ptInd) * (evals2(j,ptInd)-evals(j,ptInd))/fdstep;

                CHECK( coeffGrad(i,ptInd) == Approx(fdDeriv).margin(1e-4)); 
            }
            coeffs(i) -= fdstep;
        }
        
    }


    SECTION("Input Gradient"){

        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivities", sumMap->outputDim, numSamps);
        for(unsigned int j=0; j<numSamps; ++j){
            for(unsigned int i=0; i<sumMap->outputDim; ++i){
                sens(i,j) = 1.0 + 0.1*i + j;
            }
        }

        Kokkos::View<double**,Kokkos::HostSpace> evals = sumMap->Evaluate(in);
        Kokkos::View<double**,Kokkos::HostSpace> evals2;

        Kokkos::View<double**,Kokkos::HostSpace> inputGrad = sumMap->Gradient(in, sens);

        REQUIRE(inputGrad.extent(0)==sumMap->inputDim);
        REQUIRE(inputGrad.extent(1)==numSamps);

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<sumMap->inputDim; ++i){
            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) += fdstep;

            evals2 = sumMap->Evaluate(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                
                double fdDeriv = 0.0;
                for(unsigned int j=0; j<sumMap->outputDim; ++j)
                    fdDeriv += sens(j,ptInd) * (evals2(j,ptInd)-evals(j,ptInd))/fdstep;

                CHECK( inputGrad(i,ptInd) == Approx(fdDeriv).epsilon(1e-3)); 
            }

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) -= fdstep;
        }
        
    }

    SECTION("LogDeterminantCoeffGrad"){


        Kokkos::View<double**,Kokkos::HostSpace> detGrad = sumMap->LogDeterminantCoeffGrad(in);
        REQUIRE(detGrad.extent(0)==sumMap->numCoeffs);
        REQUIRE(detGrad.extent(1)==numSamps);
        
        Kokkos::View<double*,Kokkos::HostSpace> logDet = sumMap->LogDeterminant(in);
        Kokkos::View<double*,Kokkos::HostSpace> logDet2;

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<sumMap->numCoeffs; ++i){
            coeffs(i) += fdstep;

            sumMap->SetCoeffs(coeffs);
            logDet2 = sumMap->LogDeterminant(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                CHECK( detGrad(i,ptInd) == Approx((logDet2(ptInd)-logDet(ptInd))/fdstep).margin(1e-4)); 
            
            coeffs(i) -= fdstep;
        }

    }

        SECTION("LogDeterminantInputGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> detGrad = sumMap->LogDeterminantInputGrad(in);
        REQUIRE(detGrad.extent(0)==sumMap->inputDim);
        REQUIRE(detGrad.extent(1)==numSamps);
        
        
        Kokkos::View<double*,Kokkos::HostSpace> logDet = sumMap->LogDeterminant(in);
        Kokkos::View<double*,Kokkos::HostSpace> logDet2;

        // Compare with finite differences
        double fdstep = 1e-6;
        for(unsigned int i=0; i<sumMap->inputDim; ++i){

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) += fdstep;

            logDet2 = sumMap->LogDeterminant(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                CHECK( detGrad(i,ptInd) == Approx((logDet2(ptInd)-logDet(ptInd))/fdstep).margin(1e-3)); 
            }

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) -= fdstep;
        }

    }

}

