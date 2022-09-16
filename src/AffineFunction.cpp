#include "MParT/AffineFunction.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Initialization.h"

#if defined(MPART_ENABLE_GPU)
#include "magma_v2.h"
#endif 

using namespace mpart;

template<typename MemorySpace>
AffineFunction<MemorySpace>::AffineFunction(StridedVector<double,MemorySpace> b) : ParameterizedFunctionBase<MemorySpace>(b.size(),b.size(),0), 
                                                                  b_(b)
{}

template<typename MemorySpace>
AffineFunction<MemorySpace>::AffineFunction(StridedMatrix<double,MemorySpace> A) : ParameterizedFunctionBase<MemorySpace>(A.extent(1),A.extent(0),0),
                                                                  A_(A)
{
    // Make sure the columns of A have contiguous memory
    if(A_.stride_0() != 1){
        A_ = Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>("A col major", A_.extent(0), A_.extent(1));
        Kokkos::deep_copy(A_,A);
    }
    ldA = A_.stride_1();
}


template<typename MemorySpace>
AffineFunction<MemorySpace>::AffineFunction(StridedMatrix<double,MemorySpace> A,
                                  StridedVector<double,MemorySpace> b) : ParameterizedFunctionBase<MemorySpace>(A.extent(1),A.extent(0),0),
                                                                  A_(A),
                                                                  b_(b)
{   
    assert(A_.extent(0) == b_.extent(0));

    // Make sure the columns of A have contiguous memory
    if(A_.stride_0() != 1){
        A_ = Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>("A col major", A_.extent(0), A_.extent(1));
        Kokkos::deep_copy(A_,A);
    }
    ldA = A_.stride_1();
}


template<>
void AffineFunction<Kokkos::HostSpace>::EvaluateImpl(StridedMatrix<const double, Kokkos::HostSpace> const& pts,
                                                     StridedMatrix<double, Kokkos::HostSpace>              output)
{
    auto eigOut = KokkosToMat<double>(output);
    auto eigPts = ConstKokkosToMat<double>(pts);

    // Linear part
    if(A_.extent(0)>0){

        auto eigA = KokkosToMat<double>(A_);
        eigOut = eigA * eigPts;
    }else{
        eigOut = eigPts;
    }

    // Bias part
    if(b_.size()>0){
        auto eigB = KokkosToVec<double>(b_);
        eigOut.colwise() += eigB;
    }
}

template<typename MemorySpace>
void AffineFunction<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                           StridedMatrix<const double, MemorySpace> const& sens,
                                           StridedMatrix<double, MemorySpace>              output)
{
    return;
}
    
template<>
void AffineFunction<Kokkos::HostSpace>::GradientImpl(StridedMatrix<const double, Kokkos::HostSpace> const& pts,  
                                                StridedMatrix<const double, Kokkos::HostSpace> const& sens,
                                                StridedMatrix<double, Kokkos::HostSpace>              output)
{
    auto eigOut = KokkosToMat<double>(output);
    auto eigSens = ConstKokkosToMat<double>(sens);

    // Linear part
    if(A_.extent(0)>0){
        auto eigA = KokkosToMat<double>(A_);
        eigOut = eigA.transpose() * eigSens;
    }else{
        eigOut = eigSens;
    }
}


template class mpart::AffineFunction<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)


template<>
void AffineFunction<mpart::DeviceSpace>::EvaluateImpl(StridedMatrix<const double, mpart::DeviceSpace> const& pts,
                                                 StridedMatrix<double, mpart::DeviceSpace>              output)
{
    // Linear part
    if(A_.extent(0)>0){
        Kokkos::View<const double**, Kokkos::LayoutLeft, mpart::DeviceSpace> ptsLeft;
        Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace> outLeft;
    
        // Make sure the ptsLeft array is column major
        if(pts.stride_0()==1){
            ptsLeft = pts;
        }else{
            Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace> ptsTemp("PtsLeft", pts.extent(0), pts.extent(1));
            Kokkos::deep_copy(ptsTemp, pts);
            ptsLeft = ptsTemp;
        }
        
        bool copyOut = false;
        if(output.stride_0()==1){
            outLeft = output;
        }else{
            outLeft = Kokkos::View<double**, Kokkos::LayoutLeft,mpart::DeviceSpace>("OutLeft", output.extent(0), output.extent(1));
            copyOut = true;
        }
        
        int ldPts = ptsLeft.stride_1(); // Spacing between columns int ptsLeft
        int ldOut = outLeft.stride_1(); // Spacing between columns int outLeft
        

        // Perform the matrix multiplication
        magma_int_t device;
        magma_queue_t queue;
        magma_getdevice( &device );
        magma_queue_create( device, &queue );
        
        magma_dgemm( MagmaNoTrans,
                     MagmaNoTrans, 
                     static_cast<magma_int_t>(A_.extent(0)),
                     static_cast<magma_int_t>(ptsLeft.extent(1)),
                     static_cast<magma_int_t>(A_.extent(0)),
                     1.0, 
                     reinterpret_cast<magmaDouble_ptr>(A_.data()), 
                     static_cast<magma_int_t>(ldA),
                     reinterpret_cast<magmaDouble_ptr>(const_cast<double*>(ptsLeft.data())), 
                     static_cast<magma_int_t>(ldPts),
                     0.0, 
                     reinterpret_cast<magmaDouble_ptr>(outLeft.data()),
                     static_cast<magma_int_t>(ldOut),
                     queue );

        magma_queue_sync( queue ); // <- This seems to hang
        magma_queue_destroy( queue );

        // The layouts didn't match, so we have to copy back
        if(copyOut)
            Kokkos::deep_copy(output, outLeft);

    }else{
        Kokkos::deep_copy(output, pts);
    }

    // Bias part
    if(b_.size()>0){

        unsigned int numPts = pts.extent(1);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<mpart::DeviceSpace>::Space> policy({{0, 0}}, {{numPts, outputDim}});

        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& j, const int& i) {
            output(i,j) += b_(i);
        });
    }
}

template<>
void AffineFunction<mpart::DeviceSpace>::GradientImpl(StridedMatrix<const double, mpart::DeviceSpace> const& pts,  
                                                 StridedMatrix<const double, mpart::DeviceSpace> const& sens,
                                                 StridedMatrix<double, mpart::DeviceSpace>              output)
{
    // Linear part
    if(A_.extent(0)>0){

        Kokkos::View<const double**, Kokkos::LayoutLeft,mpart::DeviceSpace> sensLeft;
        Kokkos::View<double**, Kokkos::LayoutLeft,mpart::DeviceSpace> outLeft;
    
        // Make sure the sensLeft array is column major
        if(sens.stride_0()==1){
            sensLeft = sens;
        }else{
            Kokkos::View<double**, Kokkos::LayoutLeft,mpart::DeviceSpace> sensTemp("SensLeft", sens.extent(0), sens.extent(1));
            Kokkos::deep_copy(sensTemp, sens);
            sensLeft = sensTemp;
        }
        
        bool copyOut = false;
        if(output.stride_0()==1){
            outLeft = output;
        }else{
            outLeft = Kokkos::View<double**, Kokkos::LayoutLeft,mpart::DeviceSpace>("OutLeft", output.extent(0), output.extent(1));
            copyOut = true;
        }
        
        int ldSens = sensLeft.stride_1(); // Spacing between columns int ptsLeft
        int ldOut = outLeft.stride_1(); // Spacing between columns int outLeft
        

        // Perform the matrix multiplication
        int device;
        magma_queue_t queue;
        magma_getdevice( &device );
        magma_queue_create( device, &queue );
        

        magma_dgemm( MagmaTrans,
                     MagmaNoTrans, A_.extent(0), sensLeft.extent(1), A_.extent(0),
                     1.0, A_.data(), ldA,
                          sensLeft.data(), ldSens,
                     0.0, outLeft.data(), ldOut, queue );

        magma_queue_sync( queue ); // <- This seems to hang
        magma_queue_destroy( queue );

        // The layouts didn't match, so we have to copy back
        if(copyOut)
            Kokkos::deep_copy(output, outLeft);

    }else{
        Kokkos::deep_copy(output, sens);
    }
}


template class mpart::AffineFunction<DeviceSpace>;
#endif