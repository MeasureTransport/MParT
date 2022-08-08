#include "MParT/IdentityMap.h"

#include <numeric>

using namespace mpart;

template<typename MemorySpace>
IdentityMap<MemorySpace>::IdentityMap(unsigned int inDim, unsigned int outDim) : ConditionalMapBase<MemorySpace>(inDim, outDim, 0)
{

}



template<typename MemorySpace>
void IdentityMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                    StridedVector<double, MemorySpace>              output)
{

    // Add to logdet of full map
    for(unsigned int j=0; j<output.size(); ++j)
        output(j) = 0.0;


}

template<typename MemorySpace>
void IdentityMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{
    // Copy x_{N-M+1:N}
    StridedMatrix<const double, MemorySpace> tailPts = Kokkos::subview(
        pts, std::make_pair(int(this->inputDim - this->outputDim), int(this->inputDim)), Kokkos::ALL());
    Kokkos::deep_copy(output, tailPts);

}

template<typename MemorySpace>
void IdentityMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                             StridedMatrix<const double, MemorySpace> const& r,
                                             StridedMatrix<double, MemorySpace>              output)
{

    StridedMatrix<const double, MemorySpace> tailR = Kokkos::subview(
        r, std::make_pair(int(this->inputDim - this->outputDim), int(this->inputDim)), Kokkos::ALL());
    Kokkos::deep_copy(output, tailR);

}


template<typename MemorySpace>
void IdentityMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{

}

template<typename MemorySpace>
void IdentityMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                             StridedMatrix<double, MemorySpace>              output)
{

}

// Explicit template instantiation
template class mpart::IdentityMap<Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
    template class mpart::IdentityMap<Kokkos::DefaultExecutionSpace::memory_space>;
#endif