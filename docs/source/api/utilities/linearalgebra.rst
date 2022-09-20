===================
Linear Algebra
===================

Kokkos provides the :code:`Kokkos::View` class to store multi-dimensional arrays.  Kokkos itself does not provide tools BLAS (e.g., Mat-Mat multiplication) or LAPACK (e.g., LU factorizations and solves) operations.   MParT provides a small collection of c++ classes and functions for easily working with one- and two-dimensional :code:`Kokkos::View` objects as if they were vectors and matrices, respectively.    Below is a comparison of the MParT linear algebra tools and <Eigen `https://eigen.tuxfamily.org/index.php?title=Main_Page`>_.  These examples focus on views defined in the :code:`Kokkos::HostSpace` memory space, but these operators are defined for views in device memory (e.g., :code:`mpart::DeviceSpace`) as well.

The eigen code snippets require :code:`#include <Eigen/Core>` and :code:`#include <Eigen/Dense>` while the MParT snippets require :code:`#include <Kokkos_Core.hpp>`, :code`#include "MParT/Utilities/LinearAlgebra.h"`, and the addition of :code:`using namespace mpart;`.

Construction
-------------

For completeness, we first show how to construct two matrices :math:`A` and :math:`B` using either Eigen or Kokkos.

.. grid:: 2

    .. grid-item-card::  Matrix Construction (Eigen)

        Eigen::MatrixXd A(2,2);
        Eigen::MatrixXd B(2,2);

        A << 1.0, 2.0,
             0.5, 1.0;

        B << 1.0, 2.0,
             0.125, 0.5;

        
    .. grid-item-card::  Matrix Construction (Kokkos)

        Kokkos::View<double**, Kokkos::HostSpace> A("A", 2, 2);
        Kokkos::View<double** ,Kokkos::HostSpace> B("B", 2, 2);

        A(0,0) = 1.0; A(0,1) = 2.0;
        A(1,0) = 0.5; A(1,1) = 1.0;

        B(0,0) = 1.0; B(0,1) = 2.0;
        B(1,0) = 0.125; B(1,1) = 0.5;

Summation
----------

We will now compute :math:`C=2A+B` by first computing :math:`A+B` and then adding an additional :math:`A` to the result.  This is simply to demonstrate the use of both the :code:`+` and :code`+=` operators.

.. grid:: 2

    .. grid-item-card::  Matrix Sum (Eigen)

        Eigen::MatrixXd C;
        C = A + B;
        C += A;


    .. grid-item-card::  Matrix Sum (Kokkos+MParT)

        Kokkos::View<double** ,Kokkos::HostSpace> C;
        C = A + B;
        C += A;

Multiplication
---------------

The :code:`*` operator is overloaded in both Eigen and MParT to implement matrix multiplication.  Currently MParT does not provide any built-in tools for componentwise multiplication.  The snippet below computes the product :math:`AB`.

.. grid:: 2

    .. grid-item-card::  Matrix Sum (Eigen)

        Eigen::MatrixXd C;
        C = A*B;


    .. grid-item-card::  Matrix Sum (Kokkos+MParT)

        Kokkos::View<double** ,Kokkos::HostSpace> C;
        C = A*B;

Computing products with transposes is also straightforward.  Below we compute both :math:`A^TB` and :math`A^TB^T`.  Because MParT must use out-of-class implementations, the transpose syntax is slightly different between Eigen and MParT, but equally straightforward:

.. grid:: 2

    .. grid-item-card::  Matrix Sum (Eigen)

        Eigen::MatrixXd C;
        C = A.transpose() * B; // A^T B
        C = A.transpose() * B.transpose() // A^T B^T

    .. grid-item-card::  Matrix Sum (Kokkos+MParT)

        Kokkos::View<double** ,Kokkos::HostSpace> C;
        C = transpose(A) * B; // A^T B
        C = tranpose(A) * transpose(B); // A^T B^T



LU Factorization and Solve
---------------------------

MParT provides a class similar to Eigen's :code:`PartialPivLU` class for LU factorizations.  The `Kokkos::HostSpace` version of MParT's implementation is actually just a thin wrapper around the Eigen implementation.   The :code:`DeviceSpace` version uses the <cuSolver `https://docs.nvidia.com/cuda/cusolver/index.html`>_ library.  


The following snippets compute :math:`C=A^{-1} B` using an LU factorization of the matrix :math:`A`.  The determinant of :math:`A` is also computed from the factorization.

.. grid:: 2

    .. grid-item-card::  LU Solve (Eigen)

        Eigen::PartialPivLU<Eigen::MatrixXd> solver;
        solver.compute(A); // Compute the LU factorization of A

        Eigen::MatrixXd C;
        C = solver.solve(B);

        double det = solver.determinant();

    .. grid-item-card::  LU Solve (Kokkos+MParT)

        mpart::PartialPivLU solver;
        solver.compute(A); // Compute the LU factorization of A

        Kokkos::View<double**, Kokkos::HostSpace> C;
        C = solver.solve(B);

        double det = solver.determinant();

The MParT :code:`mpart::PartialPivLU` class also has a :code:`solveInPlace` function that computes :math:`A^{-1}B` in place by overwriting the matrix :math:`B`.  The use of :code:`solveInPlace` can reduce memory allocations because additional space to store the matrix :code:`C` is not needed.  Here is an example:

.. grid:: 2

    .. grid-item-card::  In-place LU Solve (Eigen)

        Eigen::PartialPivLU<Eigen::MatrixXd> solver;
        solver.compute(A);

        B = solver.solve(B);

    .. grid-item-card::  In-place LU Solve (Kokkos+MParT)

        mpart::PartialPivLU solver;
        solver.compute(A);
        solver.solveInPlace(B);



Linear Algebra Classes and Functions
-------------------------------------

.. doxygenclass:: mpart::PartialPivLU
    :members:
    :undoc-members:

.. doxygenfunction:: mpart::dgemm
    :members:
    :undoc-members:

.. doxygenfunction:: mpart::AddInPlace(Kokkos::View<double**,Kokkos::HostSpace>, Kokkos::View<const double**,Kokkos::HostSpace>)
    :members:
    :undoc-members:

.. doxygenfunction:: mpart::AddInPlace(Kokkos::View<double*,Kokkos::HostSpace>, Kokkos::View<const double*,Kokkos::HostSpace>)
    :members:
    :undoc-members:

.. doxygenfunction:: mpart::dgemm
    :members:
    :undoc-members:

.. doxygenfunction:: mpart::operator+(Kokkos::View<const double*,Kokkos::HostSpace>, Kokkos::View<const double*,Kokkos::HostSpace>)
    :members:
    :undoc-members:

.. doxygenfunction:: mpart::operator+(Kokkos::View<const double**,Kokkos::HostSpace>, Kokkos::View<const double**,Kokkos::HostSpace>)
    :members:
    :undoc-members: