/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: matlab/include/mexplus_eigen.hpp
 *
 * Copyright 2016-2018 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
 * @brief Define a template specialisation for Eigen::MatrixXd for ... .
 *
 * The default precision in Matlab is double, but most matrices in eos (for example the PCA basis matrices
 * are stored as float values, so this defines conversion from these matrices to Matlab.
 *
 * Todo: Documentation.
 */
template <>
inline mxArray* MxArray::from(const Eigen::MatrixXd& eigen_matrix)
{
    const int num_rows = static_cast<int>(eigen_matrix.rows());
    const int num_cols = static_cast<int>(eigen_matrix.cols());
    MxArray out_array(MxArray::Numeric<double>(num_rows, num_cols));

    // This might not copy the data but it's evil and probably really dangerous!!!:
    // mxSetData(const_cast<mxArray*>(matrix.get()), (void*)value.data());

    // This copies the data. But I suppose it makes sense that we copy the data when we go
    // from C++ to Matlab, since Matlab can unload the C++ mex module at any time I think.
    // Loop is column-wise
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
 * @brief Define a template specialisation for Eigen::MatrixXd for ... .
 *
 * Todo: Documentation.
 * TODO: Maybe provide this one as MatrixXf as well as MatrixXd? Matlab's default is double?
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

    // I think I can just use Eigen::Matrix, not a Map - the Matrix c'tor that we call creates a Map anyway?
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eigen_map(
        array.getData<double>(), nrows, ncols);
    // Not sure that's alright - who owns the data? I think as it is now, everything points to the data in the
    // mxArray owned by Matlab, but I'm not 100% sure.
    // Actually, doesn't eigen_map go out of scope and get destroyed? This might be trouble? But this
    // assignment should (or might) copy, then it's fine? Check if it invokes the copy c'tor.
    // 2 May 2018: Yes this copies.
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

#endif /* MPART_MEXPLUS_EIGEN_HPP */