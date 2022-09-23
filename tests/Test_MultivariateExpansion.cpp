#include <catch2/catch_all.hpp>

#include "MParT/MultivariateExpansion.h"
#include "MParT/OrthogonalPolynomial.h"

#include "MParT/Utilities/ArrayConversions.h"

#include <Eigen/Dense>

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing multivariate expansion", "[MultivariateExpansion]") {

    unsigned int inDim = 3;
    unsigned int outDim = 2;
    unsigned int maxDegree = 3; 
    unsigned int numPts = 2;

    FixedMultiIndexSet<Kokkos::HostSpace> mset(inDim, maxDegree);

    ProbabilistHermite basis;
    MultivariateExpansion<ProbabilistHermite,Kokkos::HostSpace> func(outDim, mset, basis);
    CHECK(func.numCoeffs == (mset.Size()*outDim));

    Kokkos::View<double*,Kokkos::HostSpace> coeffs("coefficients", func.numCoeffs);
    for(unsigned int i=0; i<func.numCoeffs; ++i)
        coeffs(i) = 0.01;
        
    func.SetCoeffs(coeffs);

    
    StridedMatrix<double, Kokkos::HostSpace> pts = Kokkos::View<double**,Kokkos::HostSpace>("Points",inDim,numPts);
    for(unsigned int ptInd=0; ptInd<numPts; ++ptInd){
        for(unsigned int d=0; d<inDim; ++d){
            pts(d,ptInd) = d+1 + double(ptInd+1)/numPts;
        }
    }

    SECTION("Evaluate"){
        StridedMatrix<double,Kokkos::HostSpace> evals = func.Evaluate(pts);
        
        for(unsigned int ptInd=0; ptInd<numPts; ++ptInd){
            for(unsigned int d=0; d<outDim; ++d){

                double val = 0; 
                for(unsigned int term=0; term<mset.Size(); ++term){
                    auto multi = mset.IndexToMulti(term);
                    double termVal = 1.0;

                    for(unsigned int i=0; i<inDim; ++i)
                        termVal *= basis.Evaluate(multi.at(i), pts(i,ptInd));

                    val += coeffs(term + d*mset.Size()) * termVal;
                }
                
                CHECK(evals(d,ptInd) == Approx(val).epsilon(1e-12));
            }
        }
    }

    SECTION("Coefficent Gradient"){
        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivity", outDim, numPts);
        Kokkos::View<double**,Kokkos::HostSpace> grads;

        for(unsigned int d=0; d<outDim; ++d){
            
            // Set the sensitivity wrt this output to 1
            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                sens(d,ptInd) = 1.0;

            // Evaluate the gradient
            grads = func.CoeffGrad(pts,sens);

            REQUIRE(grads.extent(0)==func.numCoeffs);
            REQUIRE(grads.extent(1)==numPts);

            // Check the solution
            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd){

                for(unsigned int term=0; term<mset.Size(); ++term){
                    auto multi = mset.IndexToMulti(term);
                    double termVal = 1.0;

                    for(unsigned int i=0; i<inDim; ++i)
                        termVal *= basis.Evaluate(multi.at(i), pts(i,ptInd));
        
                    CHECK(grads(term+d*mset.Size(),ptInd) == Approx(termVal).epsilon(1e-12));
                }
            }
            
            // Reset this row to 0
            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                sens(d,ptInd) = 0.0;

        }
        
    }


    SECTION("Input Gradient"){
        Kokkos::View<double**,Kokkos::HostSpace> sens("Sensitivity", outDim, numPts);
        Kokkos::View<double**,Kokkos::HostSpace> grads;

        for(unsigned int d=0; d<outDim; ++d){
            
            // Set the sensitivity wrt this output to 1
            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                sens(d,ptInd) = 1.0;

            // Evaluate the gradient
            grads = func.Gradient(pts,sens);

            REQUIRE(grads.extent(0)==inDim);
            REQUIRE(grads.extent(1)==numPts);

            // Check the solution
            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd){

                for(unsigned int inWrt=0; inWrt<inDim; ++inWrt){

                    double deriv = 0.0;
                    for(unsigned int term=0; term<mset.Size(); ++term){
                        auto multi = mset.IndexToMulti(term);
                        double termVal = 1.0;

                        for(unsigned int i=0; i<inDim; ++i){
                            if(i==inWrt){
                                termVal *= basis.Derivative(multi.at(i), pts(i,ptInd));
                            }else{
                                termVal *= basis.Evaluate(multi.at(i), pts(i,ptInd));
                            }
                        }
                        deriv += coeffs(term) * termVal;
                    }
                    CHECK(grads(inWrt,ptInd) == Approx(deriv).epsilon(1e-13));
                }
            }
            
            // Reset this row to 0
            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                sens(d,ptInd) = 0.0;

        }
        
    }


    SECTION("WrapCoeffs"){
        Eigen::VectorXd newCoeffs = Eigen::VectorXd::Random(func.numCoeffs);
        func.WrapCoeffs(newCoeffs);

        StridedMatrix<double,Kokkos::HostSpace> evals1 = func.Evaluate(pts);

        // Change the coefficients in place 
        for(unsigned int i=0;i<newCoeffs.size(); ++i)
            newCoeffs(i) += 1.0;

        StridedMatrix<double,Kokkos::HostSpace> evals2 = func.Evaluate(pts);
        
        for(unsigned int i=0; i<newCoeffs.size(); ++i)
            CHECK(&newCoeffs(i) == &func.Coeffs()(i));

        for(unsigned int d=0; d<outDim; ++d){
            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd){
                CHECK(!(evals1(d,ptInd) == Approx(evals2(d,ptInd)).epsilon(1e-14)));
            }
        }
    }
}
