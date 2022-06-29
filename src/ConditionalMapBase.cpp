#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/Miscellaneous.h"

using namespace mpart;

template<typename MemorySpace>
Kokkos::View<double**, MemorySpace> ConditionalMapBase<MemorySpace>::Evaluate(Kokkos::View<const double**, MemorySpace> const& pts)
{
    Kokkos::View<double**, MemorySpace> output("Map Evaluations", outputDim, pts.extent(1));
    EvaluateImpl(pts, output);
    return output;
}

template<typename MemorySpace>
Eigen::RowMatrixXd ConditionalMapBase<MemorySpace>::Evaluate(Eigen::RowMatrixXd const& pts)
{
    CheckDeviceMismatch("Evaluate(Eigen::RowMatrixXd const& pts)");

    Eigen::RowMatrixXd output(outputDim, pts.cols());
    Kokkos::View<const double**, MemorySpace> ptsView = ConstRowMatToKokkos<double>(pts);
    Kokkos::View<double**, MemorySpace> outView = MatToKokkos<double>(output);
    EvaluateImpl(ptsView, outView);
    return output;
}

template<typename MemorySpace>
void ConditionalMapBase<MemorySpace>::SetCoeffs(Kokkos::View<double*, MemorySpace> coeffs){

    // If coefficients already exist, make sure the sizes match
    if(this->savedCoeffs.is_allocated()){
        if(coeffs.size() != numCoeffs){
            std::stringstream msg;
            msg << "Error in ConditionalMapBase<MemorySpace>::SetCoeffs.  Expected coefficient vector with size " << numCoeffs << ", but new coefficients have size " << coeffs.size() << ".";
            throw std::invalid_argument(msg.str());
        }

        if(this->savedCoeffs.size() != numCoeffs)
            Kokkos::resize(this->savedCoeffs, numCoeffs);
    }else{

        this->savedCoeffs = Kokkos::View<double*, MemorySpace>("ConditionalMapBase<MemorySpace> Coefficients", coeffs.size());
    }

    Kokkos::deep_copy(this->savedCoeffs, coeffs);
}

template<typename MemorySpace>
Kokkos::View<double*, MemorySpace> ConditionalMapBase<MemorySpace>::LogDeterminant(Kokkos::View<const double**, MemorySpace> const& pts)
{
    Kokkos::View<double*, MemorySpace> output("Log Determinants", pts.extent(1));
    LogDeterminantImpl(pts, output);
    return output;
}

template<typename MemorySpace>
Eigen::VectorXd ConditionalMapBase<MemorySpace>::LogDeterminant(Eigen::RowMatrixXd const& pts)
{   
    CheckDeviceMismatch("LogDeterminant(Eigen::RowMatrixXd const& pts)");

    Eigen::VectorXd output(pts.cols());
    Kokkos::View<const double**, MemorySpace> ptsView = ConstRowMatToKokkos<double>(pts);
    Kokkos::View<double*, MemorySpace> outView = VecToKokkos<double>(output);
    LogDeterminantImpl(ptsView, outView);
    return output;
}

template<typename MemorySpace>
Kokkos::View<double**, MemorySpace> ConditionalMapBase<MemorySpace>::Inverse(Kokkos::View<const double**, MemorySpace> const& x1,
                                                                      Kokkos::View<const double**, MemorySpace> const& r)
{
    // Throw an error if the inputs don't have the same number of columns
    if(x1.extent(1)!=r.extent(1)){
        std::stringstream msg;
        msg << "x1 and r have different numbers of columns.  x1.extent(1)=" << x1.extent(1) << ", but r.extent(1)=" << r.extent(1);
        throw std::invalid_argument(msg.str());
    }

    Kokkos::View<double**, MemorySpace> output("Map Inverse Evaluations", outputDim, r.extent(1));
    InverseImpl(x1,r, output);
    return output;
}

template<typename MemorySpace>
Eigen::RowMatrixXd ConditionalMapBase<MemorySpace>::Inverse(Eigen::RowMatrixXd const& x1, Eigen::RowMatrixXd const& r)
{   
    CheckDeviceMismatch("Inverse(Eigen::RowMatrixXd const& x1, Eigen::RowMatrixXd const& r)");

    Eigen::RowMatrixXd output(inputDim, r.cols());

    Kokkos::View<const double**, MemorySpace> x1View = ConstRowMatToKokkos<double>(x1);
    Kokkos::View<const double**, MemorySpace> rView = ConstRowMatToKokkos<double>(r);
    Kokkos::View<double**, MemorySpace> outView = MatToKokkos<double>(output);

    InverseImpl(x1View, rView, outView);
    return output;
}

template<typename MemorySpace>
Eigen::Map<Eigen::VectorXd> ConditionalMapBase<MemorySpace>::CoeffMap()
{   
    CheckDeviceMismatch("CoeffMap()");
    return KokkosToVec(this->savedCoeffs);
}

template<typename MemorySpace>
void ConditionalMapBase<MemorySpace>::CheckDeviceMismatch(std::string functionName) const
{   
    std::stringstream msg;
    msg << "Error in call to \"" << functionName << "\".  This function is only valid on the host space,";
    msg << " but called on a DeviceSpace ConditionalMapBase object.   You must manually copy the input";
    msg << " argument to device space if you want to call this function.";
    throw std::runtime_error(msg.str());
}

template<>
void ConditionalMapBase<Kokkos::HostSpace>::CheckDeviceMismatch(std::string functionName) const
{   
}



// Eigen::Map<const Eigen::VectorXd> ConditionalMapBase<MemorySpace>::CoeffMap() const
// {
//     return KokkosToVec(this->savedCoeffs);
// }

// Explicit template instantiation
template class mpart::ConditionalMapBase<Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
    template class mpart::ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>;
#endif