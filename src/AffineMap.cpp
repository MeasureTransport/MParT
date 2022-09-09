#include "MParT/AffineMap.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;

template<typename MemorySpace>
AffineMap<MemorySpace>::AffineMap(StridedVector<double,MemorySpace> b) : ConditionalMapBase<MemorySpace>(b.size(),b.size(),0), 
                                                                  b_(b), 
                                                                  logDet_(0.0)
{}

template<typename MemorySpace>
AffineMap<MemorySpace>::AffineMap(StridedMatrix<double,MemorySpace> A) : ConditionalMapBase<MemorySpace>(A.extent(1),A.extent(0),0),
                                                                  A_(A)
{
    assert(A_.extent(0)<=A_.extent(1));
    Factorize(A_);
}


template<typename MemorySpace>
AffineMap<MemorySpace>::AffineMap(StridedMatrix<double,MemorySpace> A,
                                  StridedVector<double,MemorySpace> b) : ConditionalMapBase<MemorySpace>(A.extent(0),A.extent(0),0),
                                                                  A_(A),
                                                                  b_(b)
{   
    assert(A_.extent(0)<=A_.extent(1));
    assert(A_.extent(0) == b_.extent(0));

    Factorize(A_);
}


template<>
void AffineMap<Kokkos::HostSpace>::Factorize(StridedMatrix<double,Kokkos::HostSpace> A){

    auto eigA = KokkosToMat(A);
    unsigned int nrows = eigA.rows();
    unsigned int ncols = eigA.cols();

    // Use eigen to compute the LU decomposition in place
    luSolver_ = std::make_shared<Eigen::PartialPivLU<Eigen::MatrixXd>>(eigA.block(0,ncols-nrows, nrows, nrows));
    logDet_ = log(abs(luSolver_->determinant()));
}

template<typename MemorySpace>
void AffineMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                StridedVector<double, MemorySpace>              output)
{
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(0,output.size());

    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& i) {
        output(i) = logDet_;
    });
}

template<>
void AffineMap<Kokkos::HostSpace>::InverseImpl(StridedMatrix<const double, Kokkos::HostSpace> const& x1,
                                               StridedMatrix<const double, Kokkos::HostSpace> const& r,
                                               StridedMatrix<double, Kokkos::HostSpace>              output)
{

    auto eigOut = KokkosToMat<double>(output);
    auto eigR = ConstKokkosToMat<double>(r);
    
    // Bias part
    if(b_.size()>0){
        auto eigB = KokkosToVec<double>(b_);
        eigOut = eigR.colwise() - eigB;

    }else{
        eigOut = eigR;
    }

    if(A_.extent(0)>0){
        auto eigA = KokkosToMat(A_);
        unsigned int nrows = eigA.rows();
        unsigned int ncols = eigA.cols();

        // If the matrix is rectangular, treat it as the lower part of a block triangular matrix
        // The value of x1 contains the compute inverse for the first block
        if(nrows != ncols){
            auto eigX = ConstKokkosToMat<double>(x1);
            eigOut -= eigA.block(0,0,nrows,ncols-nrows) * eigX.topRows(ncols-nrows);
        }

        eigOut = luSolver_->solve(eigOut);
    }
}

template<typename MemorySpace>
void AffineMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                         StridedMatrix<double, MemorySpace>              output)
{
    return;
}

template<>
void AffineMap<Kokkos::HostSpace>::EvaluateImpl(StridedMatrix<const double, Kokkos::HostSpace> const& pts,
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
void AffineMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                           StridedMatrix<const double, MemorySpace> const& sens,
                                           StridedMatrix<double, MemorySpace>              output)
{
    return;
}
    
template<>
void AffineMap<Kokkos::HostSpace>::GradientImpl(StridedMatrix<const double, Kokkos::HostSpace> const& pts,  
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


template class mpart::AffineMap<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::AffineMap<DeviceSpace>;
#endif