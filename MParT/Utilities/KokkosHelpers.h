#ifndef MPART_KOKKOSHELPERS_H
#define MPART_KOKKOSHELPERS_H

#include <Kokkos_Core.hpp>

namespace mpart{

    /** Sets up a team policy for iterating over a range where each thread requires the same amount of cache memory. Uses kokkos functions to figure out the recommended team size.
        @tparam ExecutionSpace The kokkos execution space where the parallel for loop will be executed.
        @tparam FunctorType The type of functor that will be evaluated.
        @param numPts The number of iterations in the for loop.
        @param cacheBytes The amount of memory, in bytes, required by each thread.
        @param functor The for loop work.
        @return A policy with allocated cache that can be used to iterate over the range.
    */
    template<typename ExecutionSpace, typename FunctorType>
    Kokkos::TeamPolicy<ExecutionSpace> GetCachedRangePolicy(unsigned int numPts, unsigned int cacheBytes, FunctorType const& functor)
    {
        Kokkos::TeamPolicy<ExecutionSpace> policy;
        policy.set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));
            
        const unsigned int threadsPerTeam = std::min<unsigned int>(numPts, policy.team_size_recommended(functor, Kokkos::ParallelForTag()));
        const unsigned int numTeams = std::ceil( double(numPts) / threadsPerTeam );
            
        policy = Kokkos::TeamPolicy<ExecutionSpace>(numTeams, threadsPerTeam).set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));

        return policy;
    };

} // namespace mpart


#endif 