#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/Miscellaneous.h"
#include "MParT/Utilities/GPUtils.h"

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


template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::Evaluate(StridedMatrix<const double, Kokkos::HostSpace> const& pts)
{
    CheckCoefficients("Evaluate");

    Kokkos::View<double**, Kokkos::HostSpace> output("Map Evaluations", outputDim, pts.extent(1));
    EvaluateImpl(pts, output);
    return output;
}

template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<Kokkos::HostSpace>::Evaluate(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    CheckCoefficients("Evaluate");

    Eigen::RowMatrixXd output(outputDim, pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts);
    StridedMatrix<double, Kokkos::HostSpace> outView = MatToKokkos<double,Kokkos::HostSpace>(output);
    EvaluateImpl(ptsView, outView);
    return output;
}


#if defined(MPART_ENABLE_GPU)
template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::Evaluate(StridedMatrix<const double, mpart::DeviceSpace> const& pts)
{
    CheckCoefficients("Evaluate");

    Kokkos::View<double**, mpart::DeviceSpace> output("Map Evaluations", outputDim, pts.extent(1));
    EvaluateImpl(pts, output);
    return output;
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::Evaluate(StridedMatrix<const double, Kokkos::HostSpace> const& pts)
{
    // Copy the points to the device space
    StridedMatrix<const double, mpart::DeviceSpace> pts_device = ToDevice<mpart::DeviceSpace>(pts);

    // Evaluate on the device space
    StridedMatrix<double, mpart::DeviceSpace> evals_device = this->Evaluate(pts_device);

    // Copy back to the host space
    return ToHost(evals_device);
}

template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::Evaluate(StridedMatrix<const double, mpart::DeviceSpace> const& pts)
{
    // Copy the points to host
    StridedMatrix<const double, Kokkos::HostSpace> pts_host = ToHost(pts);

    // Evaluate on the host
    StridedMatrix<double, Kokkos::HostSpace> evals_host = this->Evaluate(pts_host);

    // Copy back to the device
    return ToDevice<mpart::DeviceSpace>(evals_host);
}


template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<mpart::DeviceSpace>::Evaluate(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    CheckCoefficients("Evaluate");

    Eigen::RowMatrixXd output(outputDim, pts.cols());
    StridedMatrix<const double, mpart::DeviceSpace> ptsView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts));
    return KokkosToMat( ToHost(this->Evaluate(ptsView)));
}

#endif



template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::Gradient(StridedMatrix<const double, Kokkos::HostSpace> const& pts, StridedMatrix<const double, Kokkos::HostSpace> const& sens)
{
    CheckCoefficients("Gradient");

    Kokkos::View<double**, Kokkos::HostSpace> output("Gradients", inputDim, pts.extent(1));
    GradientImpl(pts, sens, output);
    return output;
}

template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<Kokkos::HostSpace>::Gradient(Eigen::Ref<const Eigen::RowMatrixXd> const& pts, Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{
    CheckCoefficients("Gradient");

    Eigen::RowMatrixXd output(inputDim, pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts);
    StridedMatrix<const double, Kokkos::HostSpace> sensView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(sens);
    StridedMatrix<double, Kokkos::HostSpace> outView = MatToKokkos<double,Kokkos::HostSpace>(output);
    GradientImpl(ptsView, sensView, outView);
    return output;
}


#if defined(MPART_ENABLE_GPU)
template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::Gradient(StridedMatrix<const double, mpart::DeviceSpace> const& pts, StridedMatrix<const double, mpart::DeviceSpace> const& sens)
{
    CheckCoefficients("Gradient");

    Kokkos::View<double**, mpart::DeviceSpace> output("Map Evaluations", outputDim, pts.extent(1));
    GradientImpl(pts, sens, output);
    return output;
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::Gradient(StridedMatrix<const double, Kokkos::HostSpace> const& pts, StridedMatrix<const double, Kokkos::HostSpace> const& sens)
{
    // Copy the points to the device space
    StridedMatrix<const double, mpart::DeviceSpace> pts_device = ToDevice<mpart::DeviceSpace>(pts);
    StridedMatrix<const double, mpart::DeviceSpace> sens_device = ToDevice<mpart::DeviceSpace>(sens);
    // Evaluate on the device space
    StridedMatrix<double, mpart::DeviceSpace> evals_device = this->Gradient(pts_device, sens_device);

    // Copy back to the host space
    return ToHost(evals_device);
}

template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::Gradient(StridedMatrix<const double, mpart::DeviceSpace> const& pts, StridedMatrix<const double, mpart::DeviceSpace> const& sens)
{
    // Copy the points to host
    StridedMatrix<const double, Kokkos::HostSpace> pts_host = ToHost(pts);
    StridedMatrix<const double, Kokkos::HostSpace> sens_host = ToHost(sens);

    // Evaluate on the host
    StridedMatrix<double, Kokkos::HostSpace> evals_host = this->Gradient(pts_host, sens_host);

    // Copy back to the device
    return ToDevice<mpart::DeviceSpace>(evals_host);
}


template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<mpart::DeviceSpace>::Gradient(Eigen::Ref<const Eigen::RowMatrixXd> const& pts, Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{
    CheckCoefficients("Evaluate");

    Eigen::RowMatrixXd output(outputDim, pts.cols());
    StridedMatrix<const double, mpart::DeviceSpace> ptsView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts));
    StridedMatrix<const double, mpart::DeviceSpace> sensView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(sens));

    return KokkosToMat( ToHost(this->Gradient(ptsView, sensView)));
}

#endif



template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs){

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
void ParameterizedFunctionBase<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs){

    if(coeffs.size() != numCoeffs){
        std::stringstream msg;
        msg << "Error in ParameterizedFunctionBase<MemorySpace>::WrapCoeffs.  Expected coefficient vector with size " << numCoeffs << ", but new coefficients have size " << coeffs.size() << ".";
        throw std::invalid_argument(msg.str());
    }
    this->savedCoeffs = coeffs;
}

#if defined(MPART_ENABLE_GPU)

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::SetCoeffs(Kokkos::View<double*, mpart::DeviceSpace> coeffs)
{

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
void ParameterizedFunctionBase<MemorySpace>::WrapCoeffs(Kokkos::View<double*, mpart::DeviceSpace> coeffs)
{

    if(coeffs.size() != numCoeffs){
        std::stringstream msg;
        msg << "Error in ParameterizedFunctionBase<MemorySpace>::WrapCoeffs.  Expected coefficient vector with size " << numCoeffs << ", but new coefficients have size " << coeffs.size() << ".";
        throw std::invalid_argument(msg.str());
    }

    this->savedCoeffs = coeffs;
}
#endif

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::SetCoeffs(Eigen::Ref<Eigen::VectorXd> coeffs)
{
     SetCoeffs(Kokkos::View<double*,MemorySpace>(VecToKokkos<double,MemorySpace>(coeffs)));
}

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::WrapCoeffs(Eigen::Ref<Eigen::VectorXd> coeffs)
{
     WrapCoeffs(Kokkos::View<double*,MemorySpace>(VecToKokkos<double,MemorySpace>(coeffs)));
}

template<>
Eigen::Map<Eigen::VectorXd> ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffMap()
{
    CheckCoefficients("CoeffMap");
    return KokkosToVec(this->savedCoeffs);
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffGrad(StridedMatrix<const double, Kokkos::HostSpace> const& pts,
                                                                                                 StridedMatrix<const double, Kokkos::HostSpace> const& sens)
{
    CheckCoefficients("CoeffGrad");
    Kokkos::View<double**, Kokkos::HostSpace> output("Coeff Grad", numCoeffs, pts.extent(1));
    CoeffGradImpl(pts,sens, output);
    return output;
}

template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts,
                                                                           Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{
    CheckCoefficients("CoeffGrad");
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts);
    StridedMatrix<const double, Kokkos::HostSpace> sensView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(sens);

    Kokkos::View<double**,Kokkos::LayoutRight,Kokkos::HostSpace> outView = CoeffGrad(ptsView, sensView);
    return KokkosToMat(outView);
}


#if defined(MPART_ENABLE_GPU)
template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::CoeffGrad(StridedMatrix<const double, mpart::DeviceSpace> const& pts,
                                                                                                   StridedMatrix<const double, mpart::DeviceSpace> const& sens)
{
    CheckCoefficients("CoeffGrad");
    Kokkos::View<double**, mpart::DeviceSpace> output("Coeff Grad", numCoeffs, pts.extent(1));
    CoeffGradImpl(pts,sens, output);
    return output;
}

template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffGrad(StridedMatrix<const double, mpart::DeviceSpace> const& pts,
                                                                                                  StridedMatrix<const double, mpart::DeviceSpace> const& sens)
{
    return ToDevice<mpart::DeviceSpace>( this->CoeffGrad(ToHost(pts), ToHost(sens)));
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::CoeffGrad(StridedMatrix<const double, Kokkos::HostSpace> const& pts,
                                                                                                  StridedMatrix<const double, Kokkos::HostSpace> const& sens)
{
    return ToHost( this->CoeffGrad(ToDevice<mpart::DeviceSpace>(pts), ToDevice<mpart::DeviceSpace>(sens)));
}

template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<mpart::DeviceSpace>::CoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts,
                                                                            Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{
    StridedMatrix<const double, mpart::DeviceSpace> ptsView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts) );
    StridedMatrix<const double, mpart::DeviceSpace> sensView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(sens) );

    return KokkosToMat( ToHost( CoeffGrad(ptsView, sensView) ));
}


#endif

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

    template<>
    Eigen::Map<Eigen::VectorXd> ParameterizedFunctionBase<DeviceSpace>::CoeffMap()
    {
        return KokkosToVec(ToHost(this->savedCoeffs));
    }

    template class mpart::ParameterizedFunctionBase<DeviceSpace>;
#endif