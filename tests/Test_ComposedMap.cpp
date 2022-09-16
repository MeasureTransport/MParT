#include <catch2/catch_all.hpp>

#include "MParT/ComposedMap.h"
#include "MParT/MapFactory.h"

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "Testing 5 layer composed map", "[ComposedMap_Constructor]" ) {

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
    // Kokkos::View<double*,Kokkos::HostSpace> coeffs("Coefficients", composedMap->numCoeffs);
    // // intermediate output
    // Kokkos::View<double**, MemorySpace> intOutput("intermediate output", output.extent(0), output.extent(1));
    
    // // Copy points to output, then output = map(output) looped over each component
    // Kokkos::deep_copy(output, pts);
    // for(unsigned int i=0; i<comps_.size(); ++i){
        
    //     comps_.at(i)->EvaluateImpl(output, intOutput);
    //     Kokkos::deep_copy(output,intOutput);
    // }


    // SECTION("Inverse"){

    //     auto inv = triMap->Inverse(in,out);

    //     for(unsigned int i=0; i<numBlocks; ++i){
    //         for(unsigned int j=0; j<numSamps; ++j)
    //             CHECK( inv(i,j) == Approx(in(i+extraInputs,j)).epsilon(1e-6));
    //     }
    // }

    // SECTION("LogDeterminant"){
    //     auto logDet = triMap->LogDeterminant(in);

    //     REQUIRE(logDet.extent(0)==numSamps);
    //     Kokkos::View<double*, Kokkos::HostSpace> truth("True Log Det", numSamps);

    //     for(unsigned int i=0; i<numBlocks; ++i){
    //         auto blockLogDet = blocks.at(i)->LogDeterminant(Kokkos::subview(in, std::make_pair(0,int(i+1+extraInputs)), Kokkos::ALL()));

    //         for(unsigned int j=0; j<numSamps; ++j)
    //             truth(j) += blockLogDet(j);
    //     }

    //     for(unsigned int j=0; j<numSamps; ++j)
    //         CHECK(logDet(j) == Approx(truth(j)).epsilon(1e-10));

    // }

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

                CHECK( coeffGrad(i,ptInd) == Approx(fdDeriv).epsilon(1e-3)); 
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

                CHECK( inputGrad(i,ptInd) == Approx(fdDeriv).epsilon(1e-3)); 
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
                std::cout << detGrad(i,ptInd) - (logDet2(ptInd)-logDet(ptInd))/fdstep << std::endl;
                CHECK( detGrad(i,ptInd) == Approx((logDet2(ptInd)-logDet(ptInd))/fdstep).margin(1e-5)); 
                
            }
            coeffs(i) -= fdstep;
        }

    }

}
