#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/Miscellaneous.h"

using namespace mpart;

template<>
void ConditionalMapBase<Kokkos::HostSpace>::CheckDeviceMismatch(std::string) const
{
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



template<typename MemorySpace>
Kokkos::View<double**, MemorySpace> ConditionalMapBase<MemorySpace>::Evaluate(Kokkos::View<const double**, MemorySpace> const& pts)
{   
    CheckCoefficients("Evaluate");

    Kokkos::View<double**, MemorySpace> output("Map Evaluations", outputDim, pts.extent(1));
    EvaluateImpl(pts, output);
    return output;
}

template<>
Eigen::RowMatrixXd ConditionalMapBase<Kokkos::HostSpace>::Evaluate(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    CheckCoefficients("Evaluate");

    Eigen::RowMatrixXd output(outputDim, pts.cols());
    Kokkos::View<const double**, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double>(pts);
    Kokkos::View<double**, Kokkos::HostSpace> outView = MatToKokkos<double>(output);
    EvaluateImpl(ptsView, outView);
    return output;
}

template<typename MemorySpace>
Eigen::RowMatrixXd ConditionalMapBase<MemorySpace>::Evaluate(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    CheckDeviceMismatch("Evaluate(Eigen::RowMatrixXd const& pts)");

    Eigen::RowMatrixXd output;
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

template<>
void ConditionalMapBase<Kokkos::HostSpace>::SetCoeffs(Eigen::Ref<Eigen::VectorXd> coeffs){
     SetCoeffs(VecToKokkos<double>(coeffs));
}

template<typename MemorySpace>
void ConditionalMapBase<MemorySpace>::SetCoeffs(Eigen::Ref<Eigen::VectorXd> coeffs){
     CheckDeviceMismatch("SetCoeffs(Eigen::Ref<Eigen::VectorXd> coeffs)");
}


template<typename MemorySpace>
Kokkos::View<double*, MemorySpace> ConditionalMapBase<MemorySpace>::LogDeterminant(Kokkos::View<const double**, MemorySpace> const& pts)
{   
    CheckCoefficients("LogDeterminant");
    Kokkos::View<double*, MemorySpace> output("Log Determinants", pts.extent(1));
    LogDeterminantImpl(pts, output);
    return output;
}

template<>
Eigen::VectorXd ConditionalMapBase<Kokkos::HostSpace>::LogDeterminant(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{   
    CheckCoefficients("LogDeterminant");
   
    Eigen::VectorXd output(pts.cols());
    Kokkos::View<const double**, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double>(pts);
    Kokkos::View<double*, Kokkos::HostSpace> outView = VecToKokkos<double>(output);
    LogDeterminantImpl(ptsView, outView);
    return output;
}

template<typename MemorySpace>
Eigen::VectorXd ConditionalMapBase<MemorySpace>::LogDeterminant(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    CheckDeviceMismatch("LogDeterminant(Eigen::RowMatrixXd const& pts)");

    Eigen::VectorXd output;
    return output;
}

template<typename MemorySpace>
Kokkos::View<double**, MemorySpace> ConditionalMapBase<MemorySpace>::Inverse(Kokkos::View<const double**, MemorySpace> const& x1,
                                                                      Kokkos::View<const double**, MemorySpace> const& r)
{
    CheckCoefficients("Inverse");
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

template<>
Eigen::RowMatrixXd ConditionalMapBase<Kokkos::HostSpace>::Inverse(Eigen::Ref<const Eigen::RowMatrixXd> const& x1, Eigen::Ref<const Eigen::RowMatrixXd> const& r)
{       
    CheckCoefficients("Inverse");
    
    Eigen::RowMatrixXd output(inputDim, r.cols());

    Kokkos::View<const double**, Kokkos::HostSpace> x1View = ConstRowMatToKokkos<double>(x1);
    Kokkos::View<const double**, Kokkos::HostSpace> rView = ConstRowMatToKokkos<double>(r);
    Kokkos::View<double**, Kokkos::HostSpace> outView = MatToKokkos<double>(output);

    InverseImpl(x1View, rView, outView);
    return output;
}


template<typename MemorySpace>
Eigen::RowMatrixXd ConditionalMapBase<MemorySpace>::Inverse(Eigen::Ref<const Eigen::RowMatrixXd> const& x1, Eigen::Ref<const Eigen::RowMatrixXd> const& r)
{
    CheckDeviceMismatch("Inverse(Eigen::RowMatrixXd const& x1, Eigen::RowMatrixXd const& r)");

    Eigen::RowMatrixXd output;
    return output;
}

template<>
Eigen::Map<Eigen::VectorXd> ConditionalMapBase<Kokkos::HostSpace>::CoeffMap()
{   
    CheckCoefficients("CoeffMap");
    return KokkosToVec(this->savedCoeffs);
<<<<<<< HEAD
}
=======
}

template<typename MemorySpace>
Eigen::Map<Eigen::VectorXd> ConditionalMapBase<MemorySpace>::CoeffMap()
{
    CheckDeviceMismatch("CoeffMap()");
    double *dummy = nullptr;
    return Eigen::Map<Eigen::VectorXd>(dummy, 0);
}



template<typename MemorySpace>
Kokkos::View<double**, MemorySpace> ConditionalMapBase<MemorySpace>::CoeffGrad(Kokkos::View<const double**, MemorySpace> const& pts, 
                                                                               Kokkos::View<const double**, MemorySpace> const& sens)
{
    CheckCoefficients("CoeffGrad");
    Kokkos::View<double**, MemorySpace> output("Coeff Grad", numCoeffs, pts.extent(1));
    CoeffGradImpl(pts,sens, output);
    return output;
}

template<>
Eigen::RowMatrixXd ConditionalMapBase<Kokkos::HostSpace>::CoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts, 
                                                                    Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{
    CheckCoefficients("CoeffGrad");
    Eigen::RowMatrixXd output(numCoeffs, pts.cols());

    Kokkos::View<const double**, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double>(pts);
    Kokkos::View<const double**, Kokkos::HostSpace> sensView = ConstRowMatToKokkos<double>(sens);
    Kokkos::View<double**, Kokkos::HostSpace> outView = MatToKokkos<double>(output);
    
    CoeffGradImpl(ptsView, sensView, outView);

    return output;
}

template<typename MemorySpace>
Eigen::RowMatrixXd ConditionalMapBase<MemorySpace>::CoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts, 
                                                              Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{   
    CheckDeviceMismatch("CoeffGrad(Eigen::Ref<Eigen::RowMatrixXd> const& pts, Eigen::Ref<Eigen::RowMatrixXd> const& sens)");
    Eigen::RowMatrixXd output;
    return output;
}


template<typename MemorySpace>
Kokkos::View<double**, MemorySpace> ConditionalMapBase<MemorySpace>::LogDeterminantCoeffGrad(Kokkos::View<const double**, MemorySpace> const& pts)
{
    CheckCoefficients("LogDeterminantCoeffGrad");
    Kokkos::View<double**, MemorySpace> output("LogDeterminantCoeffGrad", numCoeffs, pts.extent(1));
    LogDeterminantCoeffGradImpl(pts,output);
    return output;
}

template<>
Eigen::RowMatrixXd ConditionalMapBase<Kokkos::HostSpace>::LogDeterminantCoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    CheckCoefficients("LogDeterminantCoeffGrad");
    Eigen::RowMatrixXd output(numCoeffs, pts.cols());

    Kokkos::View<const double**, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double>(pts);
    Kokkos::View<double**, Kokkos::HostSpace> outView = MatToKokkos<double>(output);
    
    LogDeterminantCoeffGradImpl(ptsView, outView);

    return output;
}

template<typename MemorySpace>
Eigen::RowMatrixXd ConditionalMapBase<MemorySpace>::LogDeterminantCoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{   
    CheckDeviceMismatch("LogDeterminantCoeffGrad(Eigen::Ref<Eigen::RowMatrixXd> const& pts)");
    Eigen::RowMatrixXd output;
    return output;
}

template<typename MemorySpace>
void ConditionalMapBase<MemorySpace>::CheckCoefficients(std::string const& functionName) const
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


// Eigen::Map<const Eigen::VectorXd> ConditionalMapBase<MemorySpace>::CoeffMap() const
// {
//     return KokkosToVec(this->savedCoeffs);
// }

// Explicit template instantiation
template class mpart::ConditionalMapBase<Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
    template class mpart::ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>;
#endif
>>>>>>> rpath-issues
