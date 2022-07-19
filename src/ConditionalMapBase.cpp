#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/Miscellaneous.h"

using namespace mpart;

template<typename MemorySpace>
Kokkos::View<double*, MemorySpace> ConditionalMapBase<MemorySpace>::LogDeterminant(StridedMatrix<const double, MemorySpace> const& pts)
{   
    this->CheckCoefficients("LogDeterminant");
    Kokkos::View<double*, MemorySpace> output("Log Determinants", pts.extent(1));
    LogDeterminantImpl(pts, output);
    return output;
}

template<>
Eigen::VectorXd ConditionalMapBase<Kokkos::HostSpace>::LogDeterminant(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{   
    this->CheckCoefficients("LogDeterminant");
   
    Eigen::VectorXd output(pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double>(pts);
    Kokkos::View<double*, Kokkos::HostSpace> outView = VecToKokkos<double>(output);
    LogDeterminantImpl(ptsView, outView);
    return output;
}

template<typename MemorySpace>
Eigen::VectorXd ConditionalMapBase<MemorySpace>::LogDeterminant(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    this->CheckDeviceMismatch("LogDeterminant(Eigen::RowMatrixXd const& pts)");

    Eigen::VectorXd output;
    return output;
}

template<typename MemorySpace>
StridedMatrix<double, MemorySpace> ConditionalMapBase<MemorySpace>::Inverse(StridedMatrix<const double, MemorySpace> const& x1,
                                                                            StridedMatrix<const double, MemorySpace> const& r)
{
    this->CheckCoefficients("Inverse");
    // Throw an error if the inputs don't have the same number of columns
    if(x1.extent(1)!=r.extent(1)){
        std::stringstream msg;
        msg << "x1 and r have different numbers of columns.  x1.extent(1)=" << x1.extent(1) << ", but r.extent(1)=" << r.extent(1);
        throw std::invalid_argument(msg.str());
    }

    Kokkos::View<double**, MemorySpace> output("Map Inverse Evaluations", this->outputDim, r.extent(1));
    InverseImpl(x1,r, output);
    return output;
}

template<>
Eigen::RowMatrixXd ConditionalMapBase<Kokkos::HostSpace>::Inverse(Eigen::Ref<const Eigen::RowMatrixXd> const& x1, Eigen::Ref<const Eigen::RowMatrixXd> const& r)
{       
    this->CheckCoefficients("Inverse");
    
    Eigen::RowMatrixXd output(outputDim, r.cols());

    StridedMatrix<const double, Kokkos::HostSpace> x1View = ConstRowMatToKokkos<double>(x1);
    StridedMatrix<const double, Kokkos::HostSpace> rView = ConstRowMatToKokkos<double>(r);
    StridedMatrix<double, Kokkos::HostSpace> outView = MatToKokkos<double>(output);

    InverseImpl(x1View, rView, outView);
    return output;
}


template<typename MemorySpace>
Eigen::RowMatrixXd ConditionalMapBase<MemorySpace>::Inverse(Eigen::Ref<const Eigen::RowMatrixXd> const& x1, Eigen::Ref<const Eigen::RowMatrixXd> const& r)
{
    this->CheckDeviceMismatch("Inverse(Eigen::RowMatrixXd const& x1, Eigen::RowMatrixXd const& r)");

    Eigen::RowMatrixXd output;
    return output;
}


template<typename MemorySpace>
StridedMatrix<double, MemorySpace> ConditionalMapBase<MemorySpace>::LogDeterminantCoeffGrad(StridedMatrix<const double, MemorySpace> const& pts)
{
    this->CheckCoefficients("LogDeterminantCoeffGrad");
    Kokkos::View<double**, MemorySpace> output("LogDeterminantCoeffGrad", this->numCoeffs, pts.extent(1));
    LogDeterminantCoeffGradImpl(pts,output);
    return output;
}

template<>
Eigen::RowMatrixXd ConditionalMapBase<Kokkos::HostSpace>::LogDeterminantCoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    this->CheckCoefficients("LogDeterminantCoeffGrad");
    Eigen::RowMatrixXd output(numCoeffs, pts.cols());

    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double>(pts);
    Kokkos::View<double**, Kokkos::HostSpace> outView = MatToKokkos<double>(output);
    
    LogDeterminantCoeffGradImpl(ptsView, outView);

    return output;
}

template<typename MemorySpace>
Eigen::RowMatrixXd ConditionalMapBase<MemorySpace>::LogDeterminantCoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{   
    this->CheckDeviceMismatch("LogDeterminantCoeffGrad(Eigen::Ref<Eigen::RowMatrixXd> const& pts)");
    Eigen::RowMatrixXd output;
    return output;
}


// Explicit template instantiation
template class mpart::ConditionalMapBase<Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
    template class mpart::ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>;
#endif