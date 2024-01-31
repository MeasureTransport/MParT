#ifndef MPART_LINEARALGEBRA_H
#define MPART_LINEARALGEBRA_H

#include <Kokkos_Core.hpp>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/GPUtils.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/KokkosHelpers.h"

#if defined(MPART_ENABLE_GPU)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

namespace mpart{


/**
 x += y
 */
template<typename ViewType1, typename ViewType2>
void AddInPlace(ViewType1 x, ViewType2 y) {
    constexpr size_t rank = GetViewRank<ViewType1>::Rank;
    if constexpr(rank == 1) {
        Kokkos::parallel_for(x.extent(0), KOKKOS_LAMBDA(const int i){x(i) += y(i);});
    } else if (rank == 2) {
        Kokkos::MDRangePolicy<Kokkos::Rank<2>,typename ViewType1::execution_space> policy({0, 0}, {x.extent(0), x.extent(1)});
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i, const int j){x(i, j) += y(i, j);});
    } else {
        assert(false);
    }
}


/**
z = x + y
*/
template<typename ScalarType, typename... Traits1, typename... Traits2>
Kokkos::View<ScalarType*, typename Kokkos::View<ScalarType*, Traits1...>::memory_space> operator+(Kokkos::View<const ScalarType*, Traits1...> x,
                                                 Kokkos::View<const ScalarType*, Traits2...> y)
{
    Kokkos::View<ScalarType*, typename Kokkos::View<ScalarType*, Traits1...>::memory_space> z("x+y", x.extent(0));
    Kokkos::deep_copy(z,x);
    mpart::AddInPlace(z,y);
    return z;
}
template<typename ScalarType, typename... Traits1, typename... Traits2>
Kokkos::View<ScalarType**, typename Kokkos::View<ScalarType**, Traits1...>::memory_space> operator+(Kokkos::View<const ScalarType**, Traits1...> x,
                                                  Kokkos::View<const ScalarType**, Traits2...> y)
{
    Kokkos::View<ScalarType**, typename Kokkos::View<ScalarType**, Traits1...>::memory_space> z("x+y", x.extent(0), y.extent(1));
    Kokkos::deep_copy(z,x);
    mpart::AddInPlace(z,y);
    return z;
}
template<typename ScalarType, typename... Traits1, typename... Traits2>
Kokkos::View<ScalarType*, typename Kokkos::View<ScalarType*, Traits1...>::memory_space> operator+(Kokkos::View<ScalarType*, Traits1...> x,
                                                 Kokkos::View<ScalarType*, Traits2...> y)
{
    Kokkos::View<ScalarType*, typename Kokkos::View<ScalarType*, Traits1...>::memory_space> z("x+y", x.extent(0));
    Kokkos::deep_copy(z,x);
    mpart::AddInPlace(z,Kokkos::View<const ScalarType*, Traits2...>(y));
    return z;
}
template<typename ScalarType, typename... Traits1, typename... Traits2>
Kokkos::View<ScalarType**, typename Kokkos::View<ScalarType**, Traits1...>::memory_space> operator+(Kokkos::View<ScalarType**, Traits1...> x,
                                                  Kokkos::View<ScalarType**, Traits2...> y)
{
    Kokkos::View<ScalarType**, Traits1...> z("x+y", x.extent(0), y.extent(1));
    Kokkos::deep_copy(z,x);
    mpart::AddInPlace(z,Kokkos::View<const ScalarType**, Traits2...>(y));
    return z;
}


template<typename ScalarType, typename... Traits1, typename... Traits2>
Kokkos::View<ScalarType*, typename Kokkos::View<ScalarType*, Traits1...>::memory_space>& operator+=(Kokkos::View<ScalarType*, Traits1...>& x,
                                                                                                    Kokkos::View<const ScalarType*, Traits2...> const& y)
{
    mpart::AddInPlace(x,y);
    return x;
}
template<typename ScalarType, typename... Traits1, typename... Traits2>
Kokkos::View<ScalarType**, typename Kokkos::View<ScalarType**, Traits1...>::memory_space>& operator+=(Kokkos::View<ScalarType**, Traits1...>& x,
                                                                                                      Kokkos::View<const ScalarType**, Traits2...> const& y)
{
    mpart::AddInPlace(x,y);
    return x;
}
template<typename ScalarType, typename... Traits1, typename... Traits2>
Kokkos::View<ScalarType*, Traits1...>& operator+=(Kokkos::View<ScalarType*, Traits1...>& x,
                                                  Kokkos::View<ScalarType*, Traits2...> const& y)
{
    mpart::AddInPlace(x,Kokkos::View<const ScalarType*, Traits2...>(y));
    return x;
}
template<typename ScalarType, typename... Traits1, typename... Traits2>
Kokkos::View<ScalarType**, Traits1...>& operator+=(Kokkos::View<ScalarType**, Traits1...>& x,
                                                   Kokkos::View<ScalarType**, Traits2...> const& y)
{
    mpart::AddInPlace(x,Kokkos::View<const ScalarType**, Traits2...>(y));
    return x;
}



/**
 * @brief Wrapper around a Kokkos view also storing a bool indicating whether the view should be transposed in matrix multiplications.
 *
 * @tparam MemorySpace
 */
template<typename MemorySpace>
struct TransposeObject{

    TransposeObject(StridedMatrix<const double, MemorySpace> viewIn,
                    bool isTransposedIn=false) : isTransposed(isTransposedIn),
                                                 view(viewIn){};

    bool isTransposed;
    StridedMatrix<const double, MemorySpace> view;

    inline int rows() const{ return isTransposed ? view.extent(1) : view.extent(0);};
    inline int cols() const{ return isTransposed ? view.extent(0) : view.extent(1);};
};

/** Returns a transpose object around a view. */
template<typename... Traits>
TransposeObject<typename Kokkos::View<Traits...>::memory_space> transpose(Kokkos::View<Traits...> view){return TransposeObject<typename Kokkos::View<Traits...>::memory_space>(view, true);};


template<typename MemorySpace>
TransposeObject<MemorySpace> transpose(TransposeObject<MemorySpace> tview){return TransposeObject<MemorySpace>(tview.view, tview.isTransposed ? false : true);};

/** Performs the matrix multiplication of two matrix A and B. */
template<typename... Traits1, typename... Traits2>
StridedMatrix<typename Kokkos::View<Traits1...>::non_const_value_type, typename Kokkos::View<Traits1...>::memory_space> operator*(Kokkos::View<Traits1...> A,
                                             Kokkos::View<Traits2...> B)
{
    return TransposeObject<typename Kokkos::View<Traits1...>::memory_space>(A) * TransposeObject<typename Kokkos::View<Traits2...>::memory_space>(B);
}

template<typename MemorySpace, typename... Traits1>
StridedMatrix<double, MemorySpace> operator*(TransposeObject<MemorySpace> A,
                                             Kokkos::View<Traits1...> B)
{
    return A * TransposeObject<MemorySpace>(B);
}

template<typename MemorySpace, typename... Traits1>
StridedMatrix<double, MemorySpace> operator*(Kokkos::View<Traits1...> A,
                                             TransposeObject<MemorySpace> B)
{
    return TransposeObject<MemorySpace>(A) * B;
}


template<typename MemorySpace, typename... Traits1, typename... Traits2>
void dgemm(double                              alpha,
           Kokkos::View<Traits1...>            A,
           Kokkos::View<Traits2...>            B,
           double                              beta,
           StridedMatrix<double, MemorySpace>  C)
{
    dgemm<MemorySpace>(alpha,
                       TransposeObject<MemorySpace>(StridedMatrix<const double, typename Kokkos::View<Traits1...>::memory_space>(A),false),
                       TransposeObject<MemorySpace>(StridedMatrix<const double, typename Kokkos::View<Traits2...>::memory_space>(B),false),
                       beta,
                       C);
}

/** BLAS-like function computing \f$ C = \alpha AB + \beta C\f$, \f$C=\alpha A^TB + \beta C\f$, \f$C=\alpha AB^T+\beta C\f$ or \f$C=\alpha A^TB^T + \beta C\f$.

    The matrix C must be preallocated.
*/
template<typename MemorySpace>
void dgemm(double                              alpha,
           TransposeObject<MemorySpace>        A,
           TransposeObject<MemorySpace>        B,
           double                              beta,
           StridedMatrix<double, MemorySpace>  C);

/** Returns C = A*B, or A^T*B or A^T*B^T or A*B^T
*/
template<typename MemorySpace>
StridedMatrix<double, MemorySpace> operator*(TransposeObject<MemorySpace> A,
                                             TransposeObject<MemorySpace> B)
{
    assert(A.cols() == B.rows());
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> C("C", A.rows(), B.cols());
    dgemm<MemorySpace>(1.0, A, B, 0.0, C);
    return C;
}

enum class ReduceDimMap { sum, norm };

template<ReduceDimMap Map>
KOKKOS_INLINE_FUNCTION double ReduceDimMapKernel(double) {return 0.;};

template<>
KOKKOS_INLINE_FUNCTION double ReduceDimMapKernel<ReduceDimMap::sum>(double x) {return x;};

template<>
KOKKOS_INLINE_FUNCTION double ReduceDimMapKernel<ReduceDimMap::norm>(double x) {return x*x;};


// This struct allows you to reduce the columns of a mxn matrix
// Performs alpha*A*[1,...,1]
template<ReduceDimMap Map, typename MemorySpace, unsigned int Dim = 1, typename = std::enable_if_t<Dim <= 1,int> >
struct ReduceDim {
    using value_type = double[];
    using size_type = typename StridedMatrix<double, MemorySpace>::size_type;

    // Keep track of the dimension we are not reducing
    size_type value_count;

    StridedMatrix<const double, MemorySpace> A_;
    double alpha_;

    ReduceDim(StridedMatrix<double, MemorySpace> A, double alpha): value_count(A.extent(1-Dim)), A_(A), alpha_(alpha) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_type reduce_idx, value_type sum) const {
        for(size_type full_idx=0; full_idx<value_count; ++full_idx) {
            double aij = 0.;
            if(Dim == 1) aij = A_(full_idx,reduce_idx);
            else aij = A_(reduce_idx,full_idx);
            sum[full_idx] += ReduceDimMapKernel<Map>(aij)*alpha_;
        }
    }

    KOKKOS_INLINE_FUNCTION void join (volatile value_type dst, const volatile value_type src) const {
        for (size_type i = 0; i < value_count; ++i) {
            dst[i] += src[i];
        }
    }

    KOKKOS_INLINE_FUNCTION void init (value_type sum) const {
        for (size_type i = 0; i < value_count; ++i) {
            sum[i] = 0.0;
        }
    }
};

/** Mimics the interface of the Eigen::PartialPivLU class, but using Kokkos::Views and CUBLAS/CUSOLVER linear algebra.

Note that the layout of the matrices used in this class is important.  Cublas expects column major (layout left).
*/
template<typename MemorySpace>
class PartialPivLU
{
public:

    PartialPivLU() {};
    PartialPivLU(Kokkos::View<const double**,Kokkos::LayoutLeft,MemorySpace> A){ compute(A);};

    /** Computes the LU factorization of a matrix A */
    void compute(Kokkos::View<const double**,Kokkos::LayoutLeft,MemorySpace> A);

    /** Computes A^{-1}x and stores the results in x.
    */
    void solveInPlace(Kokkos::View<double**,Kokkos::LayoutLeft,MemorySpace> x);

    /** Returns a view containing A^{-1}x. */
    Kokkos::View<double**,Kokkos::LayoutLeft,MemorySpace> solve(StridedMatrix<const double,MemorySpace> x);

    /** Returns the determinant of the matrix A based on its LU factorization. */
    double determinant() const;

private:

    bool isComputed;

    std::shared_ptr<Eigen::PartialPivLU<Eigen::MatrixXd>> luSolver_;

// Information used by cusolver getrf and getrs routines
#if defined(MPART_ENABLE_GPU)
    cusolverDnParams_t params;
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> LU_;
    Kokkos::View<int64_t*, MemorySpace> pivots_;
    int ldA;
#endif

};

/** Mimics the interface of the Eigen::LLT class, but using Kokkos::Views and CUBLAS/CUSOLVER linear algebra.

Note that the layout of the matrices used in this class is important.  Cublas expects column major (layout left).
*/
template<typename MemorySpace>
class Cholesky
{
public:

    Cholesky() {};
    Cholesky(Kokkos::View<const double**,Kokkos::LayoutLeft,MemorySpace> A){ compute(A);};

    /** Computes the Cholesky factorization of a matrix A */
    void compute(Kokkos::View<const double**,Kokkos::LayoutLeft,MemorySpace> A);

    /** Computes \f$A^{-1}x\f$ and stores the results in x.
    */
    void solveInPlace(Kokkos::View<double**,Kokkos::LayoutLeft,MemorySpace> x);

    /** Returns a view containing \f$A^{-1}x\f$. */
    Kokkos::View<double**,Kokkos::LayoutLeft,MemorySpace> solve(StridedMatrix<const double,MemorySpace> x);

    /** Computes \f$L^{-1}B\f$ and stores the results in B.
    */
    void solveInPlaceL(Kokkos::View<double**,Kokkos::LayoutLeft,MemorySpace> B);

    /** Computes \f$LX\f$ and stores the results in X*/
    Kokkos::View<double**,Kokkos::LayoutLeft,MemorySpace> multiplyL(Kokkos::View<const double**,Kokkos::LayoutLeft,MemorySpace> X);

    /** Returns the determinant of the matrix A based on its LU factorization. */
    double determinant() const;

private:

    bool isComputed;

    std::shared_ptr<Eigen::LLT<Eigen::MatrixXd>> cholSolver_;

// Information used by cusolver getrf and getrs routines
#if defined(MPART_ENABLE_GPU)
    cusolverDnParams_t params;
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> LLT_;
    int ldA;
    static const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
#endif

};

} // namespace mpart

#endif // #ifndef MPART_CUDALINEARALGEBRA_H