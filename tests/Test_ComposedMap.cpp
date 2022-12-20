#include <catch2/catch_all.hpp>

#include "MParT/ComposedMap.h"
#include "MParT/MapFactory.h"

#include "MParT/Utilities/LinearAlgebra.h"

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "Testing 2 layer composed map", "[ShallowComposedMap]" ) {

    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;
    options.basisNorm = false;
    
    unsigned int dim = 2;
    unsigned int numMaps = 2;
    unsigned int order = 2;
    unsigned int coeffSize = 0;
    

    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> maps(numMaps);
    for(unsigned int i=0;i<numMaps;++i){

        maps.at(i) = MapFactory::CreateTriangular<Kokkos::HostSpace>(dim, dim, order, options);
        coeffSize += maps.at(i)->numCoeffs;
    }

    std::shared_ptr<ConditionalMapBase<MemorySpace>> composedMap = std::make_shared<ComposedMap<MemorySpace>>(maps);

    CHECK(composedMap->outputDim == dim);
    CHECK(composedMap->inputDim == dim);
    CHECK(composedMap->numCoeffs == coeffSize);


    Eigen::RowVectorXd coeffs(composedMap->numCoeffs);
    for(unsigned int i=0; i<composedMap->numCoeffs; ++i)
        coeffs(i) = 0.1*(i+1);
    
    SECTION("Coefficients"){
        
        // Set the coefficients of the triangular map
        composedMap->SetCoeffs(coeffs);

        // Now make sure that the coefficients of each block were set
        unsigned int cumCoeffInd = 0;
        for(unsigned int i=0; i<numMaps; ++i){
            for(unsigned int j=0; j<maps.at(i)->numCoeffs; ++j){
                CHECK(maps.at(i)->Coeffs()(j) == coeffs(cumCoeffInd)); // Values of coefficients should be equal
                CHECK(&maps.at(i)->Coeffs()(j) == &composedMap->Coeffs()(cumCoeffInd)); // Memory location should also be the same (no copy)
                cumCoeffInd++;
            }
        }
    }


    unsigned int numSamps = 10;
    Kokkos::View<double**, Kokkos::HostSpace> in("Map Input", dim, numSamps);
    //Eigen::RowMatrixXd in(dim, numSamps);
    for(unsigned int i=0; i<dim; ++i){
        for(unsigned int j=0; j<numSamps; ++j){
            in(i,j) = double(i)/(dim) + double(j)/numSamps;
        }
    }

    composedMap->SetCoeffs(coeffs);
    auto out = composedMap->Evaluate(in);
    
    SECTION("Evaluate"){
        Kokkos::View<double**, Kokkos::HostSpace> trueOut("True output", dim, numSamps);
        Kokkos::deep_copy(trueOut, in);
        for(auto& comp : maps)
            trueOut = comp->Evaluate(trueOut);

        for(unsigned int i=0; i<in.extent(0); ++i){
            for(unsigned int j=0; j<numSamps; ++j)
                CHECK( out(i,j) == Approx(trueOut(i,j)).epsilon(1e-7).margin(1e-7));
        }
    }

    SECTION("Inverse"){

        auto inv = composedMap->Inverse(in,out);

        for(unsigned int i=0; i<in.extent(0); ++i){
            for(unsigned int j=0; j<numSamps; ++j)
                CHECK( inv(i,j) == Approx(in(i,j)).epsilon(1e-5).margin(1e-5));
        }
    }

    SECTION("LogDeterminant"){
        auto logDet = composedMap->LogDeterminant(in);

        REQUIRE(logDet.extent(0)==numSamps);
        Kokkos::View<double*, Kokkos::HostSpace> truth("True Log Det", numSamps);
        Kokkos::View<double*, Kokkos::HostSpace> partialDet("Partial log det", numSamps);
        Kokkos::View<double**, Kokkos::HostSpace> currPts("intermediate points", in.extent(0), in.extent(1));
        Kokkos::deep_copy(currPts, in);

        for(auto& map : maps){
            partialDet = map->LogDeterminant(currPts);
            currPts = map->Evaluate(currPts);
            truth += partialDet;
        }

        for(unsigned int j=0; j<numSamps; ++j)
            CHECK(logDet(j) == Approx(truth(j)).epsilon(1e-10));

    }

    SECTION("CoeffGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivities", composedMap->outputDim, numSamps);
        
        for(unsigned int j=0; j<numSamps; ++j){
            for(unsigned int i=0; i<composedMap->outputDim; ++i){
                sens(i,j) = 1.0 + 0.1*i + j;
            }
        }

        Kokkos::View<double**,Kokkos::HostSpace> evals = composedMap->Evaluate(in);
        Kokkos::View<double**,Kokkos::HostSpace> evals2;

        Kokkos::View<double**,Kokkos::HostSpace> coeffGrad = composedMap->CoeffGrad(in, sens);

        REQUIRE(coeffGrad.extent(0)==composedMap->numCoeffs);
        REQUIRE(coeffGrad.extent(1)==numSamps);

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<composedMap->numCoeffs; ++i){
            coeffs(i) += fdstep;

            composedMap->SetCoeffs(coeffs);
            evals2 = composedMap->Evaluate(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                
                double fdDeriv = 0.0;
                for(unsigned int j=0; j<composedMap->outputDim; ++j)
                    fdDeriv += sens(j,ptInd) * (evals2(j,ptInd)-evals(j,ptInd))/fdstep;

                CHECK( coeffGrad(i,ptInd) == Approx(fdDeriv).margin(1e-3)); 
            }
            coeffs(i) -= fdstep;
        }
        
    }


    SECTION("Input Gradient"){

        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivities", composedMap->outputDim, numSamps);
        for(unsigned int j=0; j<numSamps; ++j){
            for(unsigned int i=0; i<composedMap->outputDim; ++i){
                sens(i,j) = 1.0 + 0.1*i + j;
            }
        }

        Kokkos::View<double**,Kokkos::HostSpace> evals = composedMap->Evaluate(in);
        Kokkos::View<double**,Kokkos::HostSpace> evals2;

        Kokkos::View<double**,Kokkos::HostSpace> inputGrad = composedMap->Gradient(in, sens);

        REQUIRE(inputGrad.extent(0)==composedMap->inputDim);
        REQUIRE(inputGrad.extent(1)==numSamps);

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<composedMap->inputDim; ++i){
            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) += fdstep;

            evals2 = composedMap->Evaluate(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                
                double fdDeriv = 0.0;
                for(unsigned int j=0; j<composedMap->outputDim; ++j)
                    fdDeriv += sens(j,ptInd) * (evals2(j,ptInd)-evals(j,ptInd))/fdstep;

                CHECK( inputGrad(i,ptInd) == Approx(fdDeriv).margin(1e-3)); 
            }

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) -= fdstep;
        }
        
    }

    SECTION("LogDeterminantCoeffGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> detGrad = composedMap->LogDeterminantCoeffGrad(in);
        REQUIRE(detGrad.extent(0)==composedMap->numCoeffs);
        REQUIRE(detGrad.extent(1)==numSamps);
        
        Kokkos::View<double*,Kokkos::HostSpace> logDet = composedMap->LogDeterminant(in);
        Kokkos::View<double*,Kokkos::HostSpace> logDet2;

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<composedMap->numCoeffs; ++i){
            coeffs(i) += fdstep;

            composedMap->SetCoeffs(coeffs);
            logDet2 = composedMap->LogDeterminant(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                CHECK( detGrad(i,ptInd) == Approx((logDet2(ptInd)-logDet(ptInd))/fdstep).margin(1e-3)); 
                
            }
            coeffs(i) -= fdstep;
        }

    }


    SECTION("LogDeterminantInputGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> detGrad = composedMap->LogDeterminantInputGrad(in);
        REQUIRE(detGrad.extent(0)==composedMap->inputDim);
        REQUIRE(detGrad.extent(1)==numSamps);
        
        
        Kokkos::View<double*,Kokkos::HostSpace> logDet = composedMap->LogDeterminant(in);
        Kokkos::View<double*,Kokkos::HostSpace> logDet2;

        // Compare with finite differences
        double fdstep = 1e-6;
        for(unsigned int i=0; i<composedMap->inputDim; ++i){

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) += fdstep;

            logDet2 = composedMap->LogDeterminant(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                CHECK( detGrad(i,ptInd) == Approx((logDet2(ptInd)-logDet(ptInd))/fdstep).margin(1e-3)); 
            }

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) -= fdstep;
        }

    }

}


TEST_CASE( "Testing 8 layer composed map", "[DeepComposedMap]" ) {

    MapOptions options;
    options.basisType = BasisTypes::ProbabilistHermite;
    
    unsigned int dim = 2;
    unsigned int numMaps = 8;
    unsigned int order = 1;
    unsigned int coeffSize = 0;
    int numChecks = 3; // number of checkpoints to use (including storing initial points)

    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> maps(numMaps);
    for(unsigned int i=0;i<numMaps;++i){

        maps.at(i) = MapFactory::CreateTriangular<Kokkos::HostSpace>(dim, dim, order, options);
        coeffSize += maps.at(i)->numCoeffs;
    }

    std::shared_ptr<ConditionalMapBase<MemorySpace>> composedMap = std::make_shared<ComposedMap<MemorySpace>>(maps, numChecks);

    Eigen::RowVectorXd coeffs(composedMap->numCoeffs);
    for(unsigned int i=0; i<composedMap->numCoeffs; ++i)
        coeffs(i) = 0.1*(i+1);
    

    unsigned int numSamps = 10;
    Kokkos::View<double**, Kokkos::HostSpace> in("Map Input", dim, numSamps);
    //Eigen::RowMatrixXd in(dim, numSamps);
    for(unsigned int i=0; i<dim; ++i){
        for(unsigned int j=0; j<numSamps; ++j){
            in(i,j) = double(i)/(dim) + double(j)/numSamps;
        }
    }

    composedMap->SetCoeffs(coeffs);
    auto out = composedMap->Evaluate(in);
    
    SECTION("Evaluate"){
        Kokkos::View<double**, Kokkos::HostSpace> trueOut("True output", dim, numSamps);
        Kokkos::deep_copy(trueOut, in);
        for(auto& comp : maps)
            trueOut = comp->Evaluate(trueOut);

        for(unsigned int i=0; i<in.extent(0); ++i){
            for(unsigned int j=0; j<numSamps; ++j)
                CHECK( out(i,j) == Approx(trueOut(i,j)).epsilon(1e-7).margin(1e-7));
        }
    }

    SECTION("LogDeterminant"){
        auto logDet = composedMap->LogDeterminant(in);

        REQUIRE(logDet.extent(0)==numSamps);
        Kokkos::View<double*, Kokkos::HostSpace> truth("True Log Det", numSamps);
        Kokkos::View<double*, Kokkos::HostSpace> partialDet("Partial log det", numSamps);
        Kokkos::View<double**, Kokkos::HostSpace> currPts("intermediate points", in.extent(0), in.extent(1));
        Kokkos::deep_copy(currPts, in);

        for(auto& map : maps){
            partialDet = map->LogDeterminant(currPts);
            currPts = map->Evaluate(currPts);
            truth += partialDet;
        }

        for(unsigned int j=0; j<numSamps; ++j)
            CHECK(logDet(j) == Approx(truth(j)).epsilon(1e-10));

    }

    SECTION("CoeffGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivities", composedMap->outputDim, numSamps);
        
        for(unsigned int j=0; j<numSamps; ++j){
            for(unsigned int i=0; i<composedMap->outputDim; ++i){
                sens(i,j) = 1.0 + 0.1*i + j;
            }
        }

        Kokkos::View<double**,Kokkos::HostSpace> evals = composedMap->Evaluate(in);
        Kokkos::View<double**,Kokkos::HostSpace> evals2;

        Kokkos::View<double**,Kokkos::HostSpace> coeffGrad = composedMap->CoeffGrad(in, sens);

        REQUIRE(coeffGrad.extent(0)==composedMap->numCoeffs);
        REQUIRE(coeffGrad.extent(1)==numSamps);

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<composedMap->numCoeffs; ++i){
            coeffs(i) += fdstep;

            composedMap->SetCoeffs(coeffs);
            evals2 = composedMap->Evaluate(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                
                double fdDeriv = 0.0;
                for(unsigned int j=0; j<composedMap->outputDim; ++j)
                    fdDeriv += sens(j,ptInd) * (evals2(j,ptInd)-evals(j,ptInd))/fdstep;

                CHECK( coeffGrad(i,ptInd) == Approx(fdDeriv).margin(1e-3)); 
            }
            coeffs(i) -= fdstep;
        }
        
    }


    SECTION("Input Gradient"){

        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivities", composedMap->outputDim, numSamps);
        for(unsigned int j=0; j<numSamps; ++j){
            for(unsigned int i=0; i<composedMap->outputDim; ++i){
                sens(i,j) = 1.0 + 0.1*i + j;
            }
        }

        Kokkos::View<double**,Kokkos::HostSpace> evals = composedMap->Evaluate(in);
        Kokkos::View<double**,Kokkos::HostSpace> evals2;

        Kokkos::View<double**,Kokkos::HostSpace> inputGrad = composedMap->Gradient(in, sens);

        REQUIRE(inputGrad.extent(0)==composedMap->inputDim);
        REQUIRE(inputGrad.extent(1)==numSamps);

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<composedMap->inputDim; ++i){
            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) += fdstep;

            evals2 = composedMap->Evaluate(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                
                double fdDeriv = 0.0;
                for(unsigned int j=0; j<composedMap->outputDim; ++j)
                    fdDeriv += sens(j,ptInd) * (evals2(j,ptInd)-evals(j,ptInd))/fdstep;

                CHECK( inputGrad(i,ptInd) == Approx(fdDeriv).margin(1e-3)); 
            }

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) -= fdstep;
        }
        
    }

    SECTION("LogDeterminantCoeffGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> detGrad = composedMap->LogDeterminantCoeffGrad(in);
        REQUIRE(detGrad.extent(0)==composedMap->numCoeffs);
        REQUIRE(detGrad.extent(1)==numSamps);
        
        Kokkos::View<double*,Kokkos::HostSpace> logDet = composedMap->LogDeterminant(in);
        Kokkos::View<double*,Kokkos::HostSpace> logDet2;

        // Compare with finite differences
        double fdstep = 1e-5;
        for(unsigned int i=0; i<composedMap->numCoeffs; ++i){
            coeffs(i) += fdstep;

            composedMap->SetCoeffs(coeffs);
            logDet2 = composedMap->LogDeterminant(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                CHECK( detGrad(i,ptInd) == Approx((logDet2(ptInd)-logDet(ptInd))/fdstep).margin(1e-3)); 
                
            }
            coeffs(i) -= fdstep;
        }

    }


    SECTION("LogDeterminantInputGrad"){

        Kokkos::View<double**,Kokkos::HostSpace> detGrad = composedMap->LogDeterminantInputGrad(in);
        REQUIRE(detGrad.extent(0)==composedMap->inputDim);
        REQUIRE(detGrad.extent(1)==numSamps);
        
        
        Kokkos::View<double*,Kokkos::HostSpace> logDet = composedMap->LogDeterminant(in);
        Kokkos::View<double*,Kokkos::HostSpace> logDet2;

        // Compare with finite differences
        double fdstep = 1e-6;
        for(unsigned int i=0; i<composedMap->inputDim; ++i){

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) += fdstep;

            logDet2 = composedMap->LogDeterminant(in);

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd){
                CHECK( detGrad(i,ptInd) == Approx((logDet2(ptInd)-logDet(ptInd))/fdstep).margin(1e-3)); 
            }

            for(unsigned int ptInd=0; ptInd<numSamps; ++ptInd)
                in(i,ptInd) -= fdstep;
        }

    }

}