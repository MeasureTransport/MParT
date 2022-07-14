#include "MParT/IdentityMap.h"

#include <numeric>

using namespace mpart;
template<typename MemorySpace>
IdentityMap<typename MemorySpace>::IdentityMap(unsigned int inDim, unsigned int outDim) : ConditionalMapBase<MemorySpace>(inDim,outDim,0)
{


}



// void IdentityMap::LogDeterminantImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts,
//                                        Kokkos::View<double*, Kokkos::HostSpace>             &output)
// {


// }


void IdentityMap<typename MemorySpace>::EvaluateImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts,
                                 Kokkos::View<double**, Kokkos::HostSpace>            & output)
{
    // Evaluate the output
    
    const unsigned int numPts = pts.extent(1);
    assert(output.extent(0)==numPts);

    for (unsigned int i = 0; i < numPts; i++)
    {
        output(i) = pts(i);
    }

}

// void IdentityMap::InverseImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts,
//                                  Kokkos::View<double**, Kokkos::HostSpace>            & output)
// {


// }

