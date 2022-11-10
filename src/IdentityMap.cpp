#include "MParT/IdentityMap.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"
#include <numeric>

using namespace mpart;
using MemorySpace = Kokkos::HostSpace;


template<typename MemorySpace>
IdentityMap<MemorySpace>::IdentityMap(unsigned int inDim, unsigned int outDim) : ConditionalMapBase<MemorySpace>(inDim, outDim, 0)
{

}



template<typename MemorySpace>
void IdentityMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                    StridedVector<double, MemorySpace>              output)
{

    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(0,output.size());
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& j){
        output(j) = 0.0;
    });

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

    Kokkos::deep_copy(output, r);

}


template<typename MemorySpace>
void IdentityMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{
    assert(false);
}

template<typename MemorySpace>
void IdentityMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                            StridedMatrix<const double, MemorySpace> const& sens,
                            StridedMatrix<double, MemorySpace>              output)
{


    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> zeroPolicy({0, 0}, {int(this->inputDim - this->outputDim), output.extent_int(1)});
    Kokkos::parallel_for(zeroPolicy, KOKKOS_LAMBDA(const int& i, const int& j) {
        output(i,j) = 0.0;
    });


    StridedMatrix<double, MemorySpace> tailOut = Kokkos::subview(
        output, std::make_pair(int(this->inputDim - this->outputDim), int(this->inputDim)), Kokkos::ALL());

    Kokkos::deep_copy(tailOut, sens);

}

template<typename MemorySpace>
void IdentityMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                             StridedMatrix<double, MemorySpace>              output)
{
    assert(false);
}

template<typename MemorySpace>
void IdentityMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                            StridedMatrix<double, MemorySpace>              output)
{   
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> zeroPolicy({0, 0}, {output.extent(0), output.extent(1)});
    Kokkos::parallel_for(zeroPolicy, KOKKOS_LAMBDA(const int& i, const int& j) {
        output(i,j) = 0.0;
    });

}

// Explicit template instantiation
template class mpart::IdentityMap<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::IdentityMap<Kokkos::DefaultExecutionSpace::memory_space>;
#endif