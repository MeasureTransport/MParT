/**
 * Adapted from: https://github.com/patrikhuber/eos/blob/master/matlab/include/mexplus_eigen.hpp
 * @brief Handle custom conversion between mxarray and C++ matrix types
 */

#pragma once

#ifndef MPART_MEXPLUS_EIGEN_HPP
#define MPART_MEXPLUS_EIGEN_HPP

#include "mexplus/mxarray.h"

#include "Eigen/Core"
#include <Kokkos_Core.hpp>
#include "MParT/Utilities/ArrayConversions.h"


#include "mex.h"

namespace mexplus {

/**
 * @brief Define a template specialisation for Eigen::MatrixXd
 */
template <>
inline mxArray* MxArray::from(const Eigen::MatrixXd& eigen_matrix)
{
    const int num_rows = static_cast<int>(eigen_matrix.rows());
    const int num_cols = static_cast<int>(eigen_matrix.cols());
    MxArray out_array(MxArray::Numeric<double>(num_rows, num_cols));

    for (int c = 0; c < num_cols; ++c)
    {
        for (int r = 0; r < num_rows; ++r)
        {
            out_array.set(r, c, eigen_matrix(r, c));
        }
    }
    return out_array.release();
};



/**
 * @brief Define a template specialisation for Eigen::MatrixXd
 *
 */
template <>
inline void MxArray::to(const mxArray* in_array, Eigen::MatrixXd* eigen_matrix)
{
    MxArray array(in_array);

    if (array.dimensionSize() > 2)
    {
        mexErrMsgIdAndTxt(
            "MPART:matlab",
            "Given array has > 2 dimensions. Can only create 2-dimensional matrices (and vectors).");
    }

    if (array.dimensionSize() == 1 || array.dimensionSize() == 0)
    {
        mexErrMsgIdAndTxt("MPART:matlab", "Given array has 0 or 1 dimensions but we expected a 2-dimensional "
                                        "matrix (or row/column vector).");
        // Even when given a single value dimensionSize() is 2, with n=m=1. When does this happen?
    }

    if (!array.isDouble())
    {
        mexErrMsgIdAndTxt(
            "MPART:matlab",
            "Trying to create an Eigen::MatrixXd in C++, but the given data is not of type double.");
    }

    // We can be sure now that the array is 2-dimensional (or 0, but then we're screwed anyway)
    const auto nrows = array.dimensions()[0]; // or use array.rows()
    const auto ncols = array.dimensions()[1];

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eigen_map(
        array.getData<double>(), nrows, ncols);
    *eigen_matrix = eigen_map;
};

template <>
inline void MxArray::to(const mxArray* in_array, Eigen::Map<Eigen::MatrixXd>* eigen_matrix)
{
    MxArray array(in_array);

    if (array.dimensionSize() > 2)
    {
        mexErrMsgIdAndTxt(
            "MPART:matlab",
            "Given array has > 2 dimensions. Can only create 2-dimensional matrices (and vectors).");
    }

    if (array.dimensionSize() == 1 || array.dimensionSize() == 0)
    {
        mexErrMsgIdAndTxt("MPART:matlab", "Given array has 0 or 1 dimensions but we expected a 2-dimensional "
                                        "matrix (or row/column vector).");
    }

    if (!array.isDouble())
    {
        mexErrMsgIdAndTxt(
            "MPART:matlab",
            "Trying to create an Eigen::MatrixXd in C++, but the given data is not of type double.");
    }

    const auto nrows = array.dimensions()[0]; 
    const auto ncols = array.dimensions()[1];

    Eigen::Map<Eigen::MatrixXd> eigen_map(
        array.getData<double>(), nrows, ncols);
    *eigen_matrix = eigen_map;
};


template <>
inline void MxArray::to(const mxArray* in_array, Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>* kokkos_matrix)
{
    MxArray array(in_array);

    if (array.dimensionSize() > 2)
    {
        mexErrMsgIdAndTxt(
            "MPART:matlab",
            "Given array has > 2 dimensions. Can only create 2-dimensional matrices (and vectors).");
    }

    if (array.dimensionSize() == 1 || array.dimensionSize() == 0)
    {
        mexErrMsgIdAndTxt("MPART:matlab", "Given array has 0 or 1 dimensions but we expected a 2-dimensional "
                                        "matrix (or row/column vector).");
        // Even when given a single value dimensionSize() is 2, with n=m=1. When does this happen?
    }

    if (!array.isDouble())
    {
        mexErrMsgIdAndTxt(
            "MPART:matlab",
            "Trying to create an Eigen::MatrixXd in C++, but the given data is not of type double.");
    }

    // We can be sure now that the array is 2-dimensional (or 0, but then we're screwed anyway)
    unsigned int nrows = array.dimensions()[0]; // or use array.rows()
    unsigned int ncols = array.dimensions()[1];

    *kokkos_matrix = mpart::ToKokkos<double, Kokkos::LayoutLeft>(array.getData<double>(), nrows, ncols);
};

} /* namespace mexplus */

#endif /* MPART_MEXPLUS_EIGEN_H */