#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;

Kokkos::View<double**, Kokkos::HostSpace> ConditionalMapBase::Evaluate(Kokkos::View<const double**, Kokkos::HostSpace> const& pts)
{
    Kokkos::View<double**, Kokkos::HostSpace> output("Map Evaluations", outputDim, pts.extent(1));
    Evaluate(pts, output);
    return output;
}

Eigen::MatrixXd ConditionalMapBase::Evaluate(Eigen::MatrixXd const& pts)
{
    Eigen::MatrixXd output(outputDim, pts.cols());
    Kokkos::View<const double**, Kokkos::HostSpace> ptsView = ConstMatToKokkos<double>(pts);
    Kokkos::View<double**, Kokkos::HostSpace> outView = MatToKokkos<double>(output);
    Evaluate(ptsView, outView);
    return output;
}

Kokkos::View<double**, Kokkos::HostSpace> ConditionalMapBase::Inverse(Kokkos::View<const double**, Kokkos::HostSpace> const& x1, 
                                                                      Kokkos::View<const double**, Kokkos::HostSpace> const& r)
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


Eigen::MatrixXd ConditionalMapBase::Inverse(Eigen::MatrixXd const& x1, Eigen::MatrixXd const& r)
{
    Eigen::MatrixXd output(inputDim, r.cols());

    Kokkos::View<const double**, Kokkos::HostSpace> x1View = ConstMatToKokkos<double>(x1);
    Kokkos::View<const double**, Kokkos::HostSpace> rView = ConstMatToKokkos<double>(r);
    Kokkos::View<double**, Kokkos::HostSpace> outView = MatToKokkos<double>(output);

    Inverse(x1View, rView, outView);
    return output;
}
        

Eigen::Map<Eigen::VectorXd> ConditionalMapBase::CoeffMap()
{
    return KokkosToVec(this->savedCoeffs);
}

// Eigen::Map<const Eigen::VectorXd> ConditionalMapBase::CoeffMap() const
// {
//     return KokkosToVec(this->savedCoeffs);
// }