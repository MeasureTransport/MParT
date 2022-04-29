#include "MParT/ConditionalMapBase.h"

using namespace mpart;

Kokkos::View<double**, Kokkos::HostSpace> ConditionalMapBase::Evaluate(Kokkos::View<double**, Kokkos::HostSpace> const& pts)
{
    Kokkos::View<double**, Kokkos::HostSpace> output("Map Evaluations", outputDim, pts.extent(1));
    Evaluate(pts, output);
    return output;
}

Kokkos::View<double**, Kokkos::HostSpace> ConditionalMapBase::Inverse(Kokkos::View<double**, Kokkos::HostSpace> const& x1, 
                                                                      Kokkos::View<double**, Kokkos::HostSpace> const& r)
{      
    // Throw an error if the inputs don't have the same number of columns
    if(x1.extent(1)!=r.extent(1)){
        std::stringstream msg;
        msg << "x1 and r have different numbers of columns.  x1.extent(1)=" << x1.extent(1) << ", but r.extent(1)=" << r.extent(1);
        throw std::invalid_argument(msg.str());
    }
    
    Kokkos::View<double**, Kokkos::HostSpace> output("Map Inverse Evaluations", inputDim, r.extent(1));
    Inverse(x1,r, output);
    return output;
}