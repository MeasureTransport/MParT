#include "MParT/AffineMap.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/MathFunctions.h"

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Initialization.h"


using namespace mpart;

template<typename MemorySpace>
AffineMap<MemorySpace>::AffineMap(StridedVector<double,MemorySpace> b) : ConditionalMapBase<MemorySpace>(b.size(),b.size(),0),
                                                                  b_("b", b.extent(0)),
                                                                  logDet_(0.0)
{
    Kokkos::deep_copy(b_, b);
}

template<typename MemorySpace>
AffineMap<MemorySpace>::AffineMap(StridedMatrix<double,MemorySpace> A) : ConditionalMapBase<MemorySpace>(A.extent(1),A.extent(0),0),
                                                                  A_("A", A.extent(0), A.extent(1))
{
    Kokkos::deep_copy(A_, A);
    assert(A_.extent(0)<=A_.extent(1));
    Factorize();
}


template<typename MemorySpace>
AffineMap<MemorySpace>::AffineMap(StridedMatrix<double,MemorySpace> A,
                                  StridedVector<double,MemorySpace> b) : ConditionalMapBase<MemorySpace>(A.extent(1),A.extent(0),0),
                                                                  A_("A", A.extent(0), A.extent(1)),
                                                                  b_("b", b.extent(0))
{
    Kokkos::deep_copy(A_, A);
    Kokkos::deep_copy(b_, b);
    assert(A_.extent(0)<=A_.extent(1));
    assert(A_.extent(0) == b_.extent(0));
    Factorize();
}


template<typename MemorySpace>
void AffineMap<MemorySpace>::Factorize(){
    if(A_.extent(0)!=A_.extent(1)){
        StridedMatrix<const double, MemorySpace> Asub = Kokkos::subview(A_, Kokkos::ALL(), std::make_pair(A_.extent(1)-A_.extent(0),A_.extent(1)));
        luSolver_.compute(Asub);
    }else{
        luSolver_.compute(A_);
    }
    logDet_ = MathSpace::log(MathSpace::abs(luSolver_.determinant()));
}


template<typename MemorySpace>
void AffineMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                StridedVector<double, MemorySpace>              output)
{
    unsigned int N = output.size();
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy{0u, N};

    Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& i) {
        output(i) = logDet_;
    });
}

template<typename MemorySpace>
void AffineMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                         StridedMatrix<const double, MemorySpace> const& r,
                                         StridedMatrix<double, MemorySpace>              output)
{

    int numPts = r.extent(1);

    // Make sure we work with a column major output matrix if there is a linear component
    bool copyOut = false;
    StridedMatrix<double,MemorySpace> outLeft;
    if(A_.extent(0)>0){
        if(output.stride_0()==1){
            outLeft = output;
        }else{
            outLeft = Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>("OutLeft", output.extent(0), output.extent(1));
            copyOut = true;
        }
    }else{
        outLeft = output;
    }

    // Bias part
    if(b_.size()>0){

        Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({{0, 0}}, {{numPts, this->outputDim}});

        // After this for loop, we will have out = r - b
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& j, const int& i) {
            outLeft(i,j) = r(i,j) - b_(i);
        });

    }else{
        Kokkos::deep_copy(outLeft, r);
    }

    // Linear part
    if(A_.extent(0)>0){

        int nrows = A_.extent(0);
        int ncols = A_.extent(1);

        // If the matrix is rectangular, treat it as the lower part of a block triangular matrix
        // The value of x1 contains the compute inverse for the first block
        if(nrows != ncols){

            StridedMatrix<const double, MemorySpace> xsub = Kokkos::subview(x1, std::make_pair(0,ncols-nrows), Kokkos::ALL());
            StridedMatrix<const double, MemorySpace> Asub = Kokkos::subview(A_, Kokkos::ALL(), std::make_pair(0,ncols-nrows));

            dgemm<MemorySpace>(-1.0, Asub, xsub, 1.0, outLeft);
        }

        // Now solve with the square block of A_
        luSolver_.solveInPlace(outLeft);

        // Copy the inverse back if necessary
        if(copyOut)
            Kokkos::deep_copy(output, outLeft);
    }

}



template<typename MemorySpace>
void AffineMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                         StridedMatrix<double, MemorySpace>              output)
{
    return;
}

template<typename MemorySpace>
void AffineMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                          StridedMatrix<double, MemorySpace>              output)
{
    unsigned int numPts = pts.extent(1);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({{0, 0}}, {{numPts, this->outputDim}});

    // Linear part
    if(A_.extent(0)>0){
        
        // Initialize output to zeros 
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& j, const int& i) {
            output(i,j) = 0.0;
        });

        dgemm<MemorySpace>(1.0, A_, pts, 0.0, output);

    }else{
        Kokkos::deep_copy(output, pts);
    }
 
    // Bias part
    if(b_.size()>0){
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& j, const int& i) {
            output(i,j) += b_(i);
        });
    }
}

template<typename MemorySpace>
void AffineMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                           StridedMatrix<const double, MemorySpace> const& sens,
                                           StridedMatrix<double, MemorySpace>              output)
{
    return;
}

template<typename MemorySpace>
void AffineMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                         StridedMatrix<double, MemorySpace>              output)
{
    return;
}

template<typename MemorySpace>
void AffineMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
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


template class mpart::AffineMap<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::AffineMap<mpart::DeviceSpace>;
#endif
