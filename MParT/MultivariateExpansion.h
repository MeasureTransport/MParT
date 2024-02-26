#ifndef MPART_MultivariateExpansion_H
#define MPART_MultivariateExpansion_H

#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/KokkosHelpers.h"

#include <algorithm>

namespace mpart{

    /**
     @brief Defines a multivariate expansion based on the tensor product of 1d basis functions.
     
     @details 

     @tparam BasisEvaluatorType The type of the 1d basis functions
     @tparam MemorySpace The Kokkos memory space where the coefficients and evaluations are stored.
     */
    template<class BasisEvaluatorType, typename MemorySpace>
    class MultivariateExpansion : public ParameterizedFunctionBase<MemorySpace>
    {
    public:

        template<typename SetType>
        MultivariateExpansion(unsigned int              outDim, 
                              SetType            const& mset, 
                              BasisEvaluatorType const& basis1d) : ParameterizedFunctionBase<MemorySpace>(mset.Length(), outDim, mset.Size()*outDim),
                                                                   worker(mset, basis1d)
        {   
        };

        template<typename ExpansionType>
        MultivariateExpansion(unsigned int  outDim, 
                              ExpansionType expansion) : ParameterizedFunctionBase<MemorySpace>(expansion.InputSize(), outDim, expansion.NumCoeffs()*outDim),
                                                         worker(expansion)
        {
        };

        virtual ~MultivariateExpansion() = default;


        virtual void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                  StridedMatrix<double, MemorySpace>              output) override
        {
            using ExecutionSpace = typename MemoryToExecution<MemorySpace>::Space;
            
            const unsigned int numPts = pts.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = worker.CacheSize();

            // Define functor if there is a constant worker for all dimensions
            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    
                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::None);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::None);

                    unsigned int coeffStartInd = 0;
                    
                    for(unsigned int d=0; d<this->outputDim; ++d){

                        // Extract the coefficients for this output dimension
                        auto coeffs = Kokkos::subview(this->savedCoeffs, std::make_pair(coeffStartInd, coeffStartInd+worker.NumCoeffs()));
                         
                        // Evaluate the expansion
                        output(d,ptInd) = worker.Evaluate(cache.data(), coeffs);

                        coeffStartInd += worker.NumCoeffs();
                    }
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);
            
            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);
    
            Kokkos::fence();
        }

        virtual void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                  StridedMatrix<const double, MemorySpace> const& sens,
                                  StridedMatrix<double, MemorySpace>              output) override
        {
            using ExecutionSpace = typename MemoryToExecution<MemorySpace>::Space;
            
            const unsigned int numPts = pts.extent(1);
            const unsigned int inDim = pts.extent(0);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = worker.CacheSize();

            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    
                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                    Kokkos::View<double*,MemorySpace> grad(team_member.thread_scratch(1), inDim);

                    // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::Input);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Input);

                    unsigned int coeffStartInd = 0;
                    for(unsigned int i=0; i<inDim; ++i)
                        output(i, ptInd) = 0.0;

                    for(unsigned int d=0; d<this->outputDim; ++d){

                        // Extract the coefficients for this output dimension
                        auto coeffs = Kokkos::subview(this->savedCoeffs, std::make_pair(coeffStartInd, coeffStartInd+worker.NumCoeffs()));
                         
                        // Evaluate the expansion
                        worker.InputDerivative(cache.data(), coeffs, grad);

                        for(unsigned int i=0; i<inDim; ++i)
                            output(i, ptInd) += sens(d,ptInd) * grad(i);

                        coeffStartInd += worker.NumCoeffs();
                    }
                }
            };


            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize + inDim);
            
            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);

            Kokkos::fence();
        }


        virtual void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                   StridedMatrix<const double, MemorySpace> const& sens,
                                   StridedMatrix<double, MemorySpace>              output) override
        {
            using ExecutionSpace = typename MemoryToExecution<MemorySpace>::Space;
            
            const unsigned int numPts = pts.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = worker.CacheSize();
            unsigned int maxParams = worker.NumCoeffs();


            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    
                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                    Kokkos::View<double*,MemorySpace> grad(team_member.thread_scratch(1), maxParams);

                    // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::Parameters);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Parameters);

                    unsigned int coeffStartInd = 0;

                    for(unsigned int d=0; d<this->outputDim; ++d){

                        // Extract the coefficients for this output dimension
                        auto coeffs = Kokkos::subview(this->savedCoeffs, std::make_pair(coeffStartInd, coeffStartInd+worker.NumCoeffs()));
                         
                        // Evaluate the expansion
                        worker.CoeffDerivative(cache.data(), coeffs, grad);

                        for(unsigned int i=0; i<worker.NumCoeffs(); ++i)
                            output(coeffStartInd + i, ptInd) = sens(d,ptInd) * grad(i);

                        coeffStartInd += worker.NumCoeffs();
                    }
                }
            };


            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize + maxParams);
            
            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);

            Kokkos::fence();
        }

        std::vector<unsigned int> DiagonalCoeffIndices() const { return worker.NonzeroDiagonalEntries(); }

    private:

        MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace> worker;

    }; // class MultivariateExpansion
}


#endif 