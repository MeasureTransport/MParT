#ifndef MPART_MEXARRAYCONVERSIONS_H
#define MPART_MEXARRAYCONVERSIONS_H

#include <mex.h>
#include <Kokkos_Core.hpp>

namespace mpart{

    /** Converts a real-valued matlab vector to a Kokkos::View.  The memory in matlab vector is not copied for performance
        reasons.  However, this means that the user is responsible for ensuring the vector is not freed before the view.
    */
    Kokkos::View<double*, Kokkos::HostSpace>  MexToKokkos1d(const mxArray *mx);

    /** Converts a real-valued matlab matrix to a Kokkos view. The memory in matlab matrix is not copied for performance
        reasons.  However, this means that the user is responsible for ensuring the matrix is not freed before the view.*/
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>  MexToKokkos2d(const mxArray *mx);
}


#endif 