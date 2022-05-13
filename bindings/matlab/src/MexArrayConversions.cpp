#include "MexArrayConversions.h"

#include <sstream>

using namespace mpart;

Kokkos::View<double*, Kokkos::HostSpace>  mpart::MexToKokkos1d(const mxArray *mx)
{   
    size_t rows = mxGetN(mx);
    size_t cols = mxGetM(mx);

    // if(mxIsComplex(mx)){
    //     std::stringstream msg;
    //     msg << "In mpart::MexToKokkos2d.  The input matrix is complex, not real.";
    //     throw std::invalid_argument(msg.str());
    // }

    // if(! mxIsDouble(mx)){
    //     std::stringstream msg;
    //     msg << "In mpart::MexToKokkos2d.  The input matrix is not double precision.";
    //     throw std::invalid_argument(msg.str());
    // }

    // if((rows!=1)&&(cols!=1)){
    //     std::stringstream msg;
    //     msg << "In mpart::MexToKokkos1d.  The size of the input array is not 1d!  Did you mean to call MexToKokkos2d?";
    //     throw std::invalid_argument(msg.str());
    // }

    unsigned int length = std::max(rows,cols);
    double *data = (double *) mxGetData(mx);

    return Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(data, length);
}


Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>  mpart::MexToKokkos2d(const mxArray *mx)
{   
    size_t rows = mxGetN(mx);
    size_t cols = mxGetM(mx);

    if(mxIsComplex(mx)){
        std::stringstream msg;
        msg << "In mpart::MexToKokkos2d.  The input matrix is complex, not real.";
        throw std::invalid_argument(msg.str());
    }

    if(! mxIsDouble(mx)){
        std::stringstream msg;
        msg << "In mpart::MexToKokkos2d.  The input matrix is not double precision.";
        throw std::invalid_argument(msg.str());
    }

    double *data = (double *) mxGetData(mx);

    return Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(data, rows, cols);
}