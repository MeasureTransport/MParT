#include "MParT/AffineFunction.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Initialization.h"

#include "MParT/Utilities/LinearAlgebra.h"

using namespace mpart;


template<typename MemorySpace>
AffineFunction<MemorySpace>::AffineFunction(StridedVector<double,MemorySpace> b) : ParameterizedFunctionBase<MemorySpace>(b.size(),b.size(),0), 
                                                                  b_("b",b.layout()) 
{

    Kokkos::deep_copy(b_, b);
}

template<typename MemorySpace>
AffineFunction<MemorySpace>::AffineFunction(StridedMatrix<double,MemorySpace> A) : ParameterizedFunctionBase<MemorySpace>(A.extent(1),A.extent(0),0),
                                                                               A_("A", A.layout())
{
    Kokkos::deep_copy(A_, A);
    assert(A_.extent(0)<=A_.extent(1));
}


template<typename MemorySpace>
AffineFunction<MemorySpace>::AffineFunction(StridedMatrix<double,MemorySpace> A,
                                  StridedVector<double,MemorySpace> b) : ParameterizedFunctionBase<MemorySpace>(A.extent(1),A.extent(0),0),
                                                                  A_("A", A.layout()),
                                                                  b_("b",b.layout())
{   
    Kokkos::deep_copy(A_, A);
    Kokkos::deep_copy(b_, b);
    assert(A_.extent(0) <= A_.extent(1));
    assert(A_.extent(0) == b_.extent(0));
}


template<typename MemorySpace>
void AffineFunction<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                          StridedMatrix<double, MemorySpace>              output)
{
    // Linear part
    if(A_.extent(0)>0){
        dgemm<MemorySpace>(1.0, A_, pts, 0.0, output);
    }else{
        Kokkos::deep_copy(output, pts);
    }
    
    // Bias part
    if(b_.size()>0){

        unsigned int numPts = pts.extent(1);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({0, 0}, {numPts, this->outputDim});

        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& j, const int& i) {
            output(i,j) += b_(i);
        });
    }
}

template<typename MemorySpace>
void AffineFunction<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                           StridedMatrix<const double, MemorySpace> const& sens,
                                           StridedMatrix<double, MemorySpace>              output)
{
    return;
}
    
template<typename MemorySpace>
void AffineFunction<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                          StridedMatrix<const double, MemorySpace> const& sens,
                                          StridedMatrix<double, MemorySpace>              output)
{
    // Linear part
    if(A_.extent(0)>0){
        dgemm<MemorySpace>(1.0, transpose(A_), sens, 0.0, output);
    }else{
        Kokkos::deep_copy(output, sens);
    }
}


template class mpart::AffineFunction<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::AffineFunction<mpart::DeviceSpace>;
#endif 
