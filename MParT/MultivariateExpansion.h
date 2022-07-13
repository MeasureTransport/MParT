#ifndef MPART_MultivariateExpansion_H
#define MPART_MultivariateExpansion_H

#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include <algorithm>

#include <Kokkos_Vector.hpp>

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
                              BasisEvaluatorType const& basis1d) : ParameterizedFunctionBase<MemorySpace>(mset.Length(), outDim, mset.Size()*outDim)
        {   std::cout << "Here 0..." << std::endl;
            workers.push_back( MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>(mset, basis1d) );
            std::cout << "Here 1..." << std::endl;
        };

        template<typename ExpansionType>
        MultivariateExpansion(unsigned int  outDim, 
                              ExpansionType expansion) : ParameterizedFunctionBase<MemorySpace>(expansion.InputSize(), outDim, expansion.NumCoeffs()*outDim)
        {
            workers.push_back( expansion );
        };

        // template<typename SetType>
        // MultivariateExpansion(std::vector<SetType> const& msets, 
        //                       BasisEvaluatorType   const& basis1d) : ParameterizedFunctionBase<MemorySpace>(msets.at(0).Length(),  // input dimension
        //                                                                                        msets.size(),          // output dimension
        //                                                                                        std::accumulate(msets.begin(), msets.end(), 0, [](size_t sum, const SetType& mset){ return sum + mset.Size(); })) // number of coefficients
        // {
        //     for(unsigned int i=0; i<msets.size(); ++i)
        //         workers.push_back( MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>(msets.at(i), basis1d) );
        // }

        virtual ~MultivariateExpansion(){std::cout << "In destructor..." << std::endl;};


        virtual void EvaluateImpl(Kokkos::View<const double**, MemorySpace> const& pts,
                                  Kokkos::View<double**, MemorySpace>            & output) override
        {
            using ExecutionSpace = typename MemoryToExecution<MemorySpace>::Space;
            assert(workers.size()>0);

            const unsigned int numPts = pts.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = workers[0].CacheSize();
            for(unsigned int i=1; i<workers.size(); ++i)
                cacheSize = std::max(cacheSize, workers[i].CacheSize());

            // Define functor if there is a constant worker for all dimensions
            auto functor1 = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    
                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                    workers[0].FillCache1(cache.data(), pt, DerivativeFlags::None);
                    workers[0].FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::None);

                    unsigned int coeffStartInd = 0;
                    
                    for(unsigned int d=0; d<this->outputDim; ++d){

                        // Extract the coefficients for this output dimension
                        auto coeffs = Kokkos::subview(this->savedCoeffs, std::make_pair(coeffStartInd, coeffStartInd+workers[0].NumCoeffs()));
                         
                        // Evaluate the expansion
                        output(d,ptInd) = workers[0].Evaluate(cache.data(), coeffs);

                        coeffStartInd += workers[0].NumCoeffs();
                    }
                }
            };

            // Define functor for case where different workers are used in each dimension
            auto functor2 = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    
                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    unsigned int coeffStartInd = 0;

                    for(unsigned int d=0; d<this->outputDim; ++d){

                        // Extract the coefficients for this output dimension
                        auto coeffs = Kokkos::subview(this->savedCoeffs, std::make_pair(coeffStartInd, coeffStartInd+workers[d].NumCoeffs()));
                         
                        // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                        workers[d].FillCache1(cache.data(), pt, DerivativeFlags::None);
                        workers[d].FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::None);

                        // Evaluate the expansion
                        output(d,ptInd) = workers[d].Evaluate(cache.data(), coeffs);

                        coeffStartInd += workers[d].NumCoeffs();
                    }
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);
            Kokkos::TeamPolicy<ExecutionSpace> policy;
            policy.set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));
            
            if(workers.size()==1){
                const unsigned int threadsPerTeam = std::min<unsigned int>(numPts, policy.team_size_recommended(functor1, Kokkos::ParallelForTag()));
                const unsigned int numTeams = std::ceil( double(numPts) / threadsPerTeam );
                
                policy = Kokkos::TeamPolicy<ExecutionSpace>(numTeams, threadsPerTeam).set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));

                // Paralel loop over each point computing T(x_1,...,x_D) for that point
                Kokkos::parallel_for(policy, functor1);
            }else{
                const unsigned int threadsPerTeam = std::min<unsigned int>(numPts, policy.team_size_recommended(functor2, Kokkos::ParallelForTag()));
                const unsigned int numTeams = std::ceil( double(numPts) / threadsPerTeam );
                
                policy = Kokkos::TeamPolicy<ExecutionSpace>(numTeams, threadsPerTeam).set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));

                // Paralel loop over each point computing T(x_1,...,x_D) for that point
                Kokkos::parallel_for(policy, functor2);
            }

            Kokkos::fence();
        }

        void CoeffGradImpl(Kokkos::View<const double**, MemorySpace> const& pts,  
                           Kokkos::View<const double**, MemorySpace> const& sens,
                           Kokkos::View<double**, MemorySpace>            & output) override
        {
            using ExecutionSpace = typename MemoryToExecution<MemorySpace>::Space;
            assert(workers.size()>0);

            const unsigned int numPts = pts.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = workers[0].CacheSize();
            unsigned int maxParams = workers[0].NumCoeffs();

            for(unsigned int i=1; i<workers.size(); ++i){
                cacheSize = std::max(cacheSize, workers[i].CacheSize());
                maxParams = std::max(maxParams, workers[i].NumCoeffs());
            }

            auto functor1 = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    
                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                    Kokkos::View<double*,MemorySpace> grad(team_member.thread_scratch(1), maxParams);

                    // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                    workers[0].FillCache1(cache.data(), pt, DerivativeFlags::Parameters);
                    workers[0].FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Parameters);

                    unsigned int coeffStartInd = 0;

                    for(unsigned int d=0; d<this->outputDim; ++d){

                        // Extract the coefficients for this output dimension
                        auto coeffs = Kokkos::subview(this->savedCoeffs, std::make_pair(coeffStartInd, coeffStartInd+workers[0].NumCoeffs()));
                         
                        // Evaluate the expansion
                        workers[0].CoeffDerivative(cache.data(), coeffs, grad);

                        for(unsigned int i=0; i<workers[0].NumCoeffs(); ++i)
                            output(coeffStartInd + i, ptInd) = sens(d,ptInd) * grad(i);

                        coeffStartInd += workers[0].NumCoeffs();
                    }
                }
            };

            auto functor2 = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    
                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                    Kokkos::View<double*,MemorySpace> grad(team_member.thread_scratch(1), maxParams);

                    unsigned int coeffStartInd = 0;
                    
                    for(unsigned int d=0; d<this->outputDim; ++d){

                        // Extract the coefficients for this output dimension
                        auto coeffs = Kokkos::subview(this->savedCoeffs, std::make_pair(coeffStartInd, coeffStartInd+workers[d].NumCoeffs()));
                         
                        // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                        workers[d].FillCache1(cache.data(), pt, DerivativeFlags::Parameters);
                        workers[d].FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Parameters);

                        // Evaluate the expansion
                        workers[d].CoeffDerivative(cache.data(), coeffs, grad);

                        for(unsigned int i=0; i<workers[d].NumCoeffs(); ++i)
                            output(coeffStartInd + i, ptInd) = sens(d,ptInd) * grad(i);

                        coeffStartInd += workers[d].NumCoeffs();
                    }
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize + maxParams);
            Kokkos::TeamPolicy<ExecutionSpace> policy;
            policy.set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));

            if(workers.size()==1){
                const unsigned int threadsPerTeam = std::min<unsigned int>(numPts, policy.team_size_recommended(functor1, Kokkos::ParallelForTag()));
                const unsigned int numTeams = std::ceil( double(numPts) / threadsPerTeam );
                
                policy = Kokkos::TeamPolicy<ExecutionSpace>(numTeams, threadsPerTeam).set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));

                // Paralel loop over each point computing T(x_1,...,x_D) for that point
                Kokkos::parallel_for(policy, functor1);

            }else{
                const unsigned int threadsPerTeam = std::min<unsigned int>(numPts, policy.team_size_recommended(functor2, Kokkos::ParallelForTag()));
                const unsigned int numTeams = std::ceil( double(numPts) / threadsPerTeam );
                
                policy = Kokkos::TeamPolicy<ExecutionSpace>(numTeams, threadsPerTeam).set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));

                // Paralel loop over each point computing T(x_1,...,x_D) for that point
                Kokkos::parallel_for(policy, functor2);
            }

            Kokkos::fence();
        }

    private:

        Kokkos::vector< MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace> > workers;

    }; // class MultivariateExpansion
}


#endif 