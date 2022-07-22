#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/Miscellaneous.h"

using namespace mpart;

template<>
void ParameterizedFunctionBase<Kokkos::HostSpace>::CheckDeviceMismatch(std::string) const
{
}

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::CheckDeviceMismatch(std::string functionName) const
{
    std::stringstream msg;
    msg << "Error in call to \"" << functionName << "\".  This function is only valid on the host space,";
    msg << " but called on a DeviceSpace ParameterizedFunctionBase object.   You must manually copy the input";
    msg << " argument to device space if you want to call this function.";
    throw std::runtime_error(msg.str());
}



template<typename MemorySpace>
StridedMatrix<double, MemorySpace> ParameterizedFunctionBase<MemorySpace>::Evaluate(StridedMatrix<const double, MemorySpace> const& pts)
{
    CheckCoefficients("Evaluate");

    Kokkos::View<double**, MemorySpace> output("Map Evaluations", outputDim, pts.extent(1));
    EvaluateImpl(pts, output);
    return output;
}

template<typename MemorySpace>
Eigen::RowMatrixXd ParameterizedFunctionBase<MemorySpace>::Evaluate(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    CheckCoefficients("Evaluate");

    Eigen::RowMatrixXd output(outputDim, pts.cols());
    StridedMatrix<const double, MemorySpace> ptsView = ConstRowMatToKokkos<double,MemorySpace>(pts);
    StridedMatrix<double, MemorySpace> outView = MatToKokkos<double,MemorySpace>(output);
    EvaluateImpl(ptsView, outView);
    return output;
}

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::SetCoeffs(Kokkos::View<double*, MemorySpace> coeffs){

    // If coefficients already exist, make sure the sizes match
    if(this->savedCoeffs.is_allocated()){
        if(coeffs.size() != numCoeffs){
            std::stringstream msg;
            msg << "Error in ParameterizedFunctionBase<MemorySpace>::SetCoeffs.  Expected coefficient vector with size " << numCoeffs << ", but new coefficients have size " << coeffs.size() << ".";
            throw std::invalid_argument(msg.str());
        }

        if(this->savedCoeffs.size() != numCoeffs)
            Kokkos::resize(this->savedCoeffs, numCoeffs);
    }else{

        this->savedCoeffs = Kokkos::View<double*, MemorySpace>("ParameterizedFunctionBase<MemorySpace> Coefficients", coeffs.size());
    }

    Kokkos::deep_copy(this->savedCoeffs, coeffs);
}

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::SetCoeffs(Eigen::Ref<Eigen::VectorXd> coeffs){
     SetCoeffs(VecToKokkos<double,MemorySpace>(coeffs));
}

template<>
Eigen::Map<Eigen::VectorXd> ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffMap()
{
    CheckCoefficients("CoeffMap");
    return KokkosToVec(this->savedCoeffs);
}

template<>
Eigen::Map<Eigen::VectorXd> ParameterizedFunctionBase<DeviceSpace>::CoeffMap()
{
    return KokkosToVec(ToHost(this->savedCoeffs));
}

template<typename MemorySpace>
StridedMatrix<double, MemorySpace> ParameterizedFunctionBase<MemorySpace>::CoeffGrad(StridedMatrix<const double, MemorySpace> const& pts,
                                                                                     StridedMatrix<const double, MemorySpace> const& sens)
{
    CheckCoefficients("CoeffGrad");
    Kokkos::View<double**, MemorySpace> output("Coeff Grad", numCoeffs, pts.extent(1));
    CoeffGradImpl(pts,sens, output);
    return output;
}

template<typename MemorySpace>
Eigen::RowMatrixXd ParameterizedFunctionBase<MemorySpace>::CoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts,
                                                              Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{
    CheckCoefficients("CoeffGrad");
    Kokkos::View<double**, Kokkos::LayoutRight, MemorySpace> outView("CoeffGrad", numCoeffs, pts.cols());

    StridedMatrix<const double, MemorySpace> ptsView = ConstRowMatToKokkos<double,MemorySpace>(pts);
    StridedMatrix<const double, MemorySpace> sensView = ConstRowMatToKokkos<double,MemorySpace>(sens);

    CoeffGradImpl(ptsView, sensView, outView);
    return KokkosToMat(outView);
}


template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::CheckCoefficients(std::string const& functionName) const
{
    if(this->numCoeffs==0)
        return;

    bool good = true;

    if(!this->savedCoeffs.is_allocated()){
        good = false;
    }else if(this->savedCoeffs.size()!=this->numCoeffs){
        good = false;
    }

    if(!good){
        std::stringstream msg;
        msg << "Error in \"" << functionName << "\", the coefficients have not been set yet.  Make sure to call SetCoeffs() before calling this function.";
        throw std::runtime_error(msg.str());
    }
}


// Explicit template instantiation
template class mpart::ParameterizedFunctionBase<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::ParameterizedFunctionBase<Kokkos::DefaultExecutionSpace::memory_space>;
#endif