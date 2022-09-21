#include "MParT/Utilities/LinearAlgebra.h"
#include "MParT/Initialization.h"

#include <Eigen/Dense>
using namespace mpart;



template<>
void mpart::dgemm(double                                     alpha, 
                  TransposeObject<Kokkos::HostSpace>         A,
                  TransposeObject<Kokkos::HostSpace>         B,
                  double                                     beta,
                  StridedMatrix<double, Kokkos::HostSpace>   C)
{

    auto eigC = KokkosToMat<double>(C);
    auto eigB = ConstKokkosToMat<double>(B.view);
    auto eigA = ConstKokkosToMat<double>(A.view);

    if(A.isTransposed){
        if(B.isTransposed){
            eigC = (beta*eigC + alpha* eigA.transpose() * eigB.transpose() ).eval();
        }else{
            eigC = (beta*eigC + alpha* eigA.transpose() * eigB ).eval();
        }
    }else{
        if(B.isTransposed){
            eigC = (beta*eigC + alpha* eigA * eigB.transpose() ).eval();
        }else{
            eigC = (beta*eigC + alpha* eigA * eigB ).eval();
        }
    }

}


template<typename MemorySpace>
Kokkos::View<double**,Kokkos::LayoutLeft,MemorySpace> PartialPivLU<MemorySpace>::solve(StridedMatrix<const double,MemorySpace> x)
{
     Kokkos::View<double**,  Kokkos::LayoutLeft, MemorySpace> output("solution", x.extent(0),x.extent(1));
     Kokkos::deep_copy(output, x);
     solveInPlace(output);
     return output;
}


template<>
void PartialPivLU<Kokkos::HostSpace>::compute(Kokkos::View<const double**,Kokkos::LayoutLeft,Kokkos::HostSpace> A)
{
    auto eigA = ConstKokkosToMat(A);
    unsigned int nrows = eigA.rows();
    unsigned int ncols = eigA.cols();

    // Use eigen to compute the LU decomposition in place
    luSolver_ = std::make_shared<Eigen::PartialPivLU<Eigen::MatrixXd>>(eigA);
    isComputed = true;
}

template<>
void PartialPivLU<Kokkos::HostSpace>::solveInPlace(Kokkos::View<double**,Kokkos::LayoutLeft,Kokkos::HostSpace> x)
{   
    auto eigX = KokkosToMat(x);
    eigX = (luSolver_->permutationP() * eigX).eval();
    luSolver_->matrixLU().template triangularView<Eigen::UnitLower>().solveInPlace(eigX);
    luSolver_->matrixLU().template triangularView<Eigen::Upper>().solveInPlace(eigX);
}


template<>
double PartialPivLU<Kokkos::HostSpace>::determinant() const
{
    assert(isComputed); 
    return luSolver_->determinant();
}



#if defined(MPART_ENABLE_GPU)

template<>
void mpart::dgemm(double                             alpha, 
           TransposeObject<mpart::DeviceSpace>       A,
           TransposeObject<mpart::DeviceSpace>       B,
           double                                    beta,
           StridedMatrix<double, mpart::DeviceSpace> C)
{   
    // Versions of A,B, and C with the column major layout expected by cublas
    StridedMatrix<const double, mpart::DeviceSpace> ALeft;
    StridedMatrix<const double, mpart::DeviceSpace> BLeft;
    StridedMatrix<double, mpart::DeviceSpace> CLeft;
    
    // Make sure the ptsLeft array is column major
    if(A.view.stride_0()==1){
        ALeft = A.view;
    }else{
        Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace> temp("temp", A.view.extent(0), A.view.extent(1));
        Kokkos::deep_copy(temp, A.view);
        ALeft = temp;
    }

    if(B.view.stride_0()==1){
        BLeft = B.view;
    }else{
        Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace> temp("temp", B.view.extent(0), B.view.extent(1));
        Kokkos::deep_copy(temp, B.view);
        BLeft = temp;
    }
    
    bool copyOut = false;
    if(C.stride_0()==1){
        CLeft = C;
    }else{
        CLeft = Kokkos::View<double**, Kokkos::LayoutLeft,mpart::DeviceSpace>("C Left", C.extent(0), C.extent(1));
        Kokkos::deep_copy(CLeft, C);
        copyOut = true;
    }
    
    int ldA = ALeft.stride_1(); // Spacing between columns int A
    int ldB = BLeft.stride_1(); // Spacing between columns int B
    int ldC = CLeft.stride_1(); // Spacing between columns int C

    // Perform the matrix multiplication
    cublasStatus_t info =  cublasDgemm(GetInitializeStatusObject().GetCublasHandle(),
                                       A.isTransposed ? CUBLAS_OP_T : CUBLAS_OP_N, 
                                       B.isTransposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                                       A.rows(), B.cols(), A.cols(),
                                       &alpha,
                                       ALeft.data(), ldA,
                                       BLeft.data(), ldB,
                                       &beta,
                                       CLeft.data(), ldC);
    
    // The layouts didn't match, so we have to copy back
    if(copyOut)
        Kokkos::deep_copy(C, CLeft);
}

template<>
void PartialPivLU<mpart::DeviceSpace>::compute(Kokkos::View<const double**,Kokkos::LayoutLeft,mpart::DeviceSpace> A)
{

    ldA = A.stride_1();

    int nrows = A.extent(0);
    int ncols = A.extent(1);
    assert(nrows==ncols);

    // Resize the space for storing the LU factorization
    LU_ = Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace>("LU", A.extent(0), A.extent(0));
    pivots_ = Kokkos::View<int64_t*, mpart::DeviceSpace>("Pivots", A.extent(0));

    int ldLU = LU_.stride_1();
    int info;

    // Copy the right block of the matrix A_ into LU_
    Kokkos::deep_copy(LU_, A);

    // Set up cuSolver options
    cusolverDnCreateParams(&params);
    cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);
    
    size_t d_workSize, h_workSize;
    cusolverDnXgetrf_bufferSize(GetInitializeStatusObject().GetCusolverHandle(), 
                                params,
                                LU_.extent(0),
                                LU_.extent(1), 
                                CUDA_R_64F, 
                                LU_.data(),
                                ldLU, 
                                CUDA_R_64F, 
                                &d_workSize, 
                                &h_workSize);


    Kokkos::View<double*, mpart::DeviceSpace> d_workspace("LU Workspace", d_workSize);
    Kokkos::View<double*, Kokkos::HostSpace> h_workspace("LU Workspace", h_workSize);
    Kokkos::View<int*, mpart::DeviceSpace> d_info("Info",1);

    cusolverDnXgetrf(GetInitializeStatusObject().GetCusolverHandle(), 
                     params, 
                     LU_.extent(0), 
                     LU_.extent(1), 
                     CUDA_R_64F,
                     LU_.data(), ldLU, 
                     pivots_.data(),
                     CUDA_R_64F, 
                     d_workspace.data(),
                     d_workSize, 
                     h_workspace.data(), 
                     h_workSize,
                     d_info.data());
    
    info = ToHost(d_info)(0);

    // Error handling
    if(info<0){
        std::stringstream msg;
        msg << "In PartialPivLU<mpart::DeviceSpace>::compute: Argument " << -info << " to cusolverDnDgetrf had an illegal value or memory allocation failed.  Could not compute factorization.";
        throw std::runtime_error(msg.str());
    }else if(info>0){
        std::stringstream msg;
        msg << "In PartialPivLU<mpart::DeviceSpace>::compute: The " << info << " diagonal entry of the matrix U is exactly zero.  The right block of matrix A is singular.";
        throw std::runtime_error(msg.str());
    }else{
        isComputed = true;
    }
}

template<>
void PartialPivLU<mpart::DeviceSpace>::solveInPlace(Kokkos::View<double**,Kokkos::LayoutLeft,mpart::DeviceSpace> x)
{   
    assert(isComputed);

    int ldX = x.stride_1();
    int ldLU = LU_.stride_1();

    int info;
    Kokkos::View<int*, mpart::DeviceSpace> d_info("Info", 1);

    cusolverDnXgetrs(GetInitializeStatusObject().GetCusolverHandle(), 
                     params, 
                     CUBLAS_OP_N, 
                     LU_.extent(0), 
                     x.extent(1),
                     CUDA_R_64F, 
                     LU_.data(), 
                     ldLU, 
                     pivots_.data(),
                     CUDA_R_64F, 
                     x.data(), 
                     ldX, 
                     d_info.data());
    info = ToHost(d_info)(0);

    // Error checking
    if(info!=0){
        std::stringstream msg;
        msg << "PartialPivLU<mpart::DeviceSpace>::solveInPlace: Could not compute inverse with cusolverDnXgetrs.  Argument " << -info << " to cusolverDnXgetrs had a bad value.";
        throw std::runtime_error(msg.str());
    }
}


template<>
double PartialPivLU<mpart::DeviceSpace>::determinant() const
{
    assert(isComputed); 
    double det = 1.0;
    Kokkos::parallel_reduce(LU_.extent(0), KOKKOS_CLASS_LAMBDA (const int& i, double& lprod) {
        lprod *= LU_(i,i);
    },Kokkos::Prod<double>(det));

    return det;
}


template struct mpart::PartialPivLU<mpart::DeviceSpace>;
#endif

template struct mpart::PartialPivLU<Kokkos::HostSpace>;