#ifndef MPART_MATHFUNCTIONS_H
#define MPART_MATHFUNCTIONS_H

#include <Kokkos_Core.hpp>
#include "ArrayConversions.h"

namespace mpart{

    /** Computes the factorial d! */
    KOKKOS_INLINE_FUNCTION unsigned int Factorial(unsigned int d)
    {
        unsigned int out = 1;
        for(unsigned int i=2; i<=d; ++i)
            out *= i;
        return out;
    }

    // This struct allows you to reduce the columns of a mxn matrix
    // Performs alpha*A*[1,...,1]
    template<typename MemorySpace>
    struct ReduceColumn {
        using value_type = double*;
        using size_type = Kokkos::View<double**, MemorySpace>::size_type;

        // Keep track of the row count, m
        size_type value_count;

        Kokkos::View<double**, MemorySpace> A_;
        double alpha_;

        ReduceColumn(Kokkos::View<double**, MemorySpace> A, double alpha): value_count(A.extent(0)), A_(A), alpha_(alpha) {}

        KOKKOS_INLINE_FUNCTION void opterator()(const size_type j, value_type sum) const {
            for(size_type i=0; i<value_count; ++i)
                sum[i] += A_(i,j)*alpha_;
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

}

#endif