#include "MParT/AffineMap.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Initialization.h"
 

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
    Factorize();
}


template<typename MemorySpace>
AffineMap<MemorySpace>::AffineMap(StridedMatrix<double,MemorySpace> A,
                                  StridedVector<double,MemorySpace> b) : ConditionalMapBase<MemorySpace>(A.extent(1),A.extent(0),0),
                                                                  A_(A),
                                                                  b_(b)
{   
    assert(A_.extent(0)<=A_.extent(1));
    assert(A_.extent(0) == b_.extent(0));
    Factorize();
}


template<>
void AffineMap<Kokkos::HostSpace>::Factorize(){

    auto eigA = KokkosToMat(A_);
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

    Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& i) {
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

template<>
void AffineMap<mpart::DeviceSpace>::Factorize()
{
    // If A_ is not column major, create a column major version
    if(A_.stride_0()!=1){
        Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace> anew("A_", A_.extent(0), A_.extent(1));
        Kokkos::deep_copy(anew, A_);
        A_ = anew;
    }
    ldA = A_.stride_1();

    int nrows = A_.extent(0);
    int ncols = A_.extent(1);

    // Resize the space for storing the LU factorization
    LU_ = Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace>("LU", A_.extent(0), A_.extent(0));
    pivots_ = Kokkos::View<int*, mpart::DeviceSpace>("Pivots", A_.extent(0));

    int ldLU = LU_.stride_1();
    int info;

    // Copy the right block of the matrix A_ into LU_
    Kokkos::deep_copy(LU_, Kokkos::subview(A_, Kokkos::ALL(), std::make_pair(ncols-nrows, int(A_.extent(1)))));

    // Set up cuSolver options
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
    
    size_t d_workSize, h_workSize;
    cusolverDnXgetrf_bufferSize(GetInitializeStatusObject().GetCusolverHandle(), 
                                params,
                                LU_.extent(0),
                                LU_.extent(1), 
                                traits<double>::cuda_data_type, 
                                LU_.data(),
                                ldLU, 
                                traits<double>::cuda_data_type, 
                                &d_workSize, 
                                &h_workSize);


    Kokkos::View<double*, mpart::DeviceSpace> d_workspace("LU Workspace", d_workSize);
    Kokkos::View<double*, Kokkos::HostSpace> h_workspace("LU Workspace", h_workSize);
    Kokkos::View<double*, mpart::DeviceSpace> d_info("Info",1);

    cusolverDnXgetrf(GetInitializeStatusObject().GetCusolverHandle(), 
                     params, 
                     LU_.extent(0), 
                     LU_.extent(1), 
                     traits<double>::cuda_data_type,
                     LU_.data(), ldLU, 
                     pivots_,
                     traits<double>::cuda_data_type, 
                     d_workspace,
                     d_workSize, 
                     h_workspace, 
                     h_workSize,
                     d_info);
    
    info = ToHost(d_info)(0);

    // cusolverStatus_t res = 
    // cusolverDnDgetrf(GetInitializeStatusObject().GetCublasHandle(),
    //                  LU_.extent(0),
    //                  LU_.extent(1),
    //                  LU_.data(),
    //                  ldLU,
    //                  workspace.data(),
    //                  pivots_.data(),
    //                  &info);

    // Error handling
    if(info<0){
        std::stringstream msg;
        msg << "In AffineMap::Factorize(): Argument " << -info << " to cusolverDnDgetrf had an illegal value or memory allocation failed.  Could not compute factorization.";
        throw std::runtime_error(msg.str());
    }else if(info>0){
        std::stringstream msg;
        msg << "In AffineMap::Factorize(): The " << info << " diagonal entry of the matrix U is exactly zero.  The right block of matrix A is singular.";
        throw std::runtime_error(msg.str());
    }

    // Compute the log determinant 
    logDet_ = 0.0;
    Kokkos::parallel_reduce(LU_.extent(0), KOKKOS_CLASS_LAMBDA (const int& i, double& lsum ) {
        lsum += log(abs(LU_(i,i)));
    },logDet_);
}

template<>
void AffineMap<mpart::DeviceSpace>::InverseImpl(StridedMatrix<const double, mpart::DeviceSpace> const& x1,
                                                StridedMatrix<const double, mpart::DeviceSpace> const& r,
                                                StridedMatrix<double, mpart::DeviceSpace>              output)
{
    if((A_.extent(0)>0)&&(LU_.extent(0)==0))
        throw std::runtime_error("In AffineMap::InverseImpl: Cannot compute inverse because factorization of A has not occured.");
    
    int numPts = r.extent(1);

    // Make sure we work with a column major output matrix if there is a linear component and we need to use MAGMA
    bool copyOut = false;
    Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace> outLeft;
    if(A_.extent(0)>0){
        if(output.stride_0()==1){
            outLeft = output;
        }else{
            outLeft = Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace>("OutLeft", output.extent(0), output.extent(1));
            copyOut = true;
        }
    }else{
        outLeft = output;
    }
    
    // Bias part
    if(b_.size()>0){

        Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<mpart::DeviceSpace>::Space> policy({{0, 0}}, {{numPts, outputDim}});

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

        int ldOut = outLeft.stride_1();
        int ldLU = LU_.stride_1();

        // If the matrix is rectangular, treat it as the lower part of a block triangular matrix
        // The value of x1 contains the compute inverse for the first block
        if(nrows != ncols){

            Kokkos::View<const double**,Kokkos::LayoutLeft, mpart::DeviceSpace> xLeft;
            if(x1.stride_0()==1){
                xLeft = x1;
            }else{
                Kokkos::View<double**,Kokkos::LayoutLeft, mpart::DeviceSpace> xTemp("xLeft", ncols-nrows, x1.extent(1));
                Kokkos::deep_copy(xTemp, Kokkos::subview(x1, std::make_pair(0,ncols-nrows), std::make_pair(0,int(x1.extent(1)))));
                xLeft = xTemp;
            }

            int ldX = xLeft.stride_1(); // Spacing between columns int ptsLeft

            double alpha = -1.0;
            double beta = 1.0;
            cublasStatus_t info =  cublasDgemm(GetInitializeStatusObject().GetCublasHandle(),
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           nrows, xLeft.extent(1), ncols-nrows,
                           &alpha,
                           A_.data(), ldA,
                           xLeft.data(), ldX,
                           &beta,
                           outLeft.data(), ldOut;)

            // // Perform the matrix multiplication
            // magma_int_t device;
            // magma_queue_t queue;
            // magma_getdevice( &device );
            // magma_queue_create( device, &queue );
            
            // // After this degemm, we will have out = r - b - A_{11}*x_1
            // magma_dgemm( MagmaNoTrans,
            //             MagmaNoTrans, static_cast<magma_int_t>(nrows), 
            //             static_cast<magma_int_t>(xLeft.extent(1)),
            //             static_cast<magma_int_t>(ncols-nrows),
            //             -1.0, 
            //             reinterpret_cast<magmaDouble_ptr>(A_.data()), 
            //             static_cast<magma_int_t>(ldA),
            //             reinterpret_cast<magmaDouble_ptr>(const_cast<double*>(xLeft.data())), 
            //             static_cast<magma_int_t>(ldX),
            //             1.0, 
            //             reinterpret_cast<magmaDouble_ptr>(outLeft.data()), 
            //             static_cast<magma_int_t>(ldOut),
            //             queue ); 

            // magma_queue_sync( queue ); // <- This seems to hang
            // magma_queue_destroy( queue );
        }

        int info;
        Kokkos::View<double*, mpart::DeviceSpace> d_info("Info", 1);

        cusolverDnXgetrs(GetInitializeStatusObject().GetCusolverHandle(), 
                         params, 
                         CUBLAS_OP_N, 
                         LU_.extent(0), 
                         numPts,
                         traits<double>::cuda_data_type, 
                         LU_.data(), 
                         ldLU, 
                         pivots_.data(),
                         traits<double>::cuda_data_type, 
                         outLeft.data(), 
                         ldOut, 
                         d_info);
        info = ToHost(d_info)(0);

        // // Compute the inverse in-place using the outLeft matrix:  out = A_{12}^{-1}(r - b - A_{11}*x_1)
        // magma_int_t info;
        // magma_dgetrs_gpu(MagmaNoTrans, 
        //                  static_cast<magma_int_t>(A_.extent(0)),
        //                 static_cast<magma_int_t>(numPts), 
        //                 reinterpret_cast<magmaDouble_ptr>(LU_.data()), 
        //                 static_cast<magma_int_t>(ldLU), 
        //                 pivots_.data(), 
        //                 reinterpret_cast<magmaDouble_ptr>(outLeft.data()), 
        //                 static_cast<magma_int_t>(ldOut), 
        //                 &info);
        
        // Error checking
        if(info!=0){
            std::stringstream msg;
            msg << "In AffineMap::InverseImpl: Could not compute inverse with magma_dgetrs_gpu.  Argument " << -info << " to magma_dgetrs_gpu had a bad value.";
            throw std::runtime_error(msg.str());
        }

        // Copy the inverse back if necessary
        if(copyOut)
            Kokkos::deep_copy(output, outLeft);
    }
}


template<>
void AffineMap<mpart::DeviceSpace>::EvaluateImpl(StridedMatrix<const double, mpart::DeviceSpace> const& pts,
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
        double alpha = 1.0;
        double beta = 0.0;
        cublasStatus_t info =  cublasDgemm(GetInitializeStatusObject().GetCublasHandle(),
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           A_.extent(0), ptsLeft.extent(1), A_.extent(1),
                           &alpha,
                           A_.data(), ldA,
                           ptsLeft.data(), ldPts,
                           &beta,
                           outLeft.data(), ldOut;)

        // magma_int_t device;
        // magma_queue_t queue;
        // magma_getdevice( &device );
        // magma_queue_create( device, &queue );
        
        // magma_dgemm( MagmaNoTrans,
        //              MagmaNoTrans, 
        //              static_cast<magma_int_t>(A_.extent(0)),
        //              static_cast<magma_int_t>(ptsLeft.extent(1)),
        //              static_cast<magma_int_t>(A_.extent(0)),
        //              1.0, 
        //              reinterpret_cast<magmaDouble_ptr>(A_.data()), 
        //              static_cast<magma_int_t>(ldA),
        //              reinterpret_cast<magmaDouble_ptr>(const_cast<double*>(ptsLeft.data())), 
        //              static_cast<magma_int_t>(ldPts),
        //              0.0, 
        //              reinterpret_cast<magmaDouble_ptr>(outLeft.data()),
        //              static_cast<magma_int_t>(ldOut),
        //              queue );

        // magma_queue_sync( queue ); // <- This seems to hang, at least with MParT's build of magma
        // magma_queue_destroy( queue );
        
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
void AffineMap<mpart::DeviceSpace>::GradientImpl(StridedMatrix<const double, mpart::DeviceSpace> const& pts,  
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
        double alpha = 1.0;
        double beta = 0.0;
        cublasStatus_t info =  cublasDgemm(GetInitializeStatusObject().GetCublasHandle(),
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           A_.extent(1), sensLeft.extent(1), A_.extent(0),
                           &alpha,
                           A_.data(), ldA,
                           sensLeft.data(), ldSens,
                           &beta,
                           outLeft.data(), ldOut;)

        
        // The layouts didn't match, so we have to copy back
        if(copyOut)
            Kokkos::deep_copy(output, outLeft);

    }else{
        Kokkos::deep_copy(output, sens);
    }
}


template class mpart::AffineMap<DeviceSpace>;
#endif