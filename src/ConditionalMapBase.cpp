#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;

Kokkos::View<double**, Kokkos::HostSpace> ConditionalMapBase::Evaluate(Kokkos::View<const double**, Kokkos::HostSpace> const& pts)
{
    Kokkos::View<double**, Kokkos::HostSpace> output("Map Evaluations", outputDim, pts.extent(1));
    EvaluateImpl(pts, output);
    return output;
}

Eigen::RowMatrixXd ConditionalMapBase::Evaluate(Eigen::RowMatrixXd const& pts)
{
    Eigen::RowMatrixXd output(outputDim, pts.cols());
    Kokkos::View<const double**, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double>(pts);
    Kokkos::View<double**, Kokkos::HostSpace> outView = MatToKokkos<double>(output);
    EvaluateImpl(ptsView, outView);
    return output;
}


void ConditionalMapBase::SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs){ 

    // If coefficients already exist, make sure the sizes match
    if(this->savedCoeffs.is_allocated()){
        if(coeffs.size() != this->savedCoeffs.size()){
            std::stringstream msg;
            msg << "Error in ConditionalMapBase::SetCoeffs.  Current coefficient vector has size " << this->savedCoeffs.size() << ", but new coefficients have size " << coeffs.size() << ".";
            throw std::invalid_argument(msg.str());
        }
    }else{

        this->savedCoeffs = Kokkos::View<double*, Kokkos::HostSpace>("ConditionalMapBase Coefficients", coeffs.size());
    }

    Kokkos::deep_copy(this->savedCoeffs, coeffs); 
}

Kokkos::View<double*, Kokkos::HostSpace> ConditionalMapBase::LogDeterminant(Kokkos::View<const double**, Kokkos::HostSpace> const& pts)
{
    Kokkos::View<double*, Kokkos::HostSpace> output("Log Determinants", pts.extent(1));
    LogDeterminantImpl(pts, output);
    return output;
}
        
Eigen::VectorXd ConditionalMapBase::LogDeterminant(Eigen::RowMatrixXd const& pts)
{
    Eigen::VectorXd output(pts.cols());
    Kokkos::View<const double**, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double>(pts);
    Kokkos::View<double*, Kokkos::HostSpace> outView = VecToKokkos<double>(output);
    LogDeterminantImpl(ptsView, outView);
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
    InverseImpl(x1,r, output);
    return output;
}


Eigen::RowMatrixXd ConditionalMapBase::Inverse(Eigen::RowMatrixXd const& x1, Eigen::RowMatrixXd const& r)
{
    Eigen::RowMatrixXd output(inputDim, r.cols());

    Kokkos::View<const double**, Kokkos::HostSpace> x1View = ConstRowMatToKokkos<double>(x1);
    Kokkos::View<const double**, Kokkos::HostSpace> rView = ConstRowMatToKokkos<double>(r);
    Kokkos::View<double**, Kokkos::HostSpace> outView = MatToKokkos<double>(output);

    InverseImpl(x1View, rView, outView);
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