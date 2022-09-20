#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/Miscellaneous.h"

using namespace mpart;

template<>
template<>
Kokkos::View<double*, Kokkos::HostSpace> ConditionalMapBase<Kokkos::HostSpace>::LogDeterminant(StridedMatrix<const double, Kokkos::HostSpace> const& pts)
{
    this->CheckCoefficients("LogDeterminant");
    Kokkos::View<double*, Kokkos::HostSpace> output("Log Determinants", pts.extent(1));
    LogDeterminantImpl(pts, output);
    return output;
}

template<>
Eigen::VectorXd ConditionalMapBase<Kokkos::HostSpace>::LogDeterminant(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    this->CheckCoefficients("LogDeterminant");

    StridedMatrix<const double,Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts);
    Kokkos::View<double*,Kokkos::HostSpace> outView = LogDeterminant(ptsView);
    return KokkosToVec(outView);
}


#if defined(MPART_ENABLE_GPU)

template<>
template<>
Kokkos::View<double*, mpart::DeviceSpace> ConditionalMapBase<mpart::DeviceSpace>::LogDeterminant(StridedMatrix<const double, mpart::DeviceSpace> const& pts)
{
    this->CheckCoefficients("LogDeterminant");
    Kokkos::View<double*, mpart::DeviceSpace> output("Log Determinants", pts.extent(1));
    LogDeterminantImpl(pts, output);
    return output;
}

template<>
template<>
Kokkos::View<double*, Kokkos::HostSpace> ConditionalMapBase<mpart::DeviceSpace>::LogDeterminant(StridedMatrix<const double, Kokkos::HostSpace> const& pts)
{   
    return ToHost( this->LogDeterminant( ToDevice<mpart::DeviceSpace>(pts) ));
}

template<>
template<>
Kokkos::View<double*, mpart::DeviceSpace> ConditionalMapBase<Kokkos::HostSpace>::LogDeterminant(StridedMatrix<const double, mpart::DeviceSpace> const& pts)
{   
    return ToDevice<mpart::DeviceSpace>( this->LogDeterminant( ToHost(pts) ));
}

template<>
Eigen::VectorXd ConditionalMapBase<mpart::DeviceSpace>::LogDeterminant(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{   
    StridedMatrix<const double,Kokkos::HostSpace> pts_view = ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts);
    Kokkos::View<double*, Kokkos::HostSpace> outView = ToHost( this->LogDeterminant( ToDevice<mpart::DeviceSpace>(pts_view) ));
    return KokkosToVec<double>(outView);
}


#endif 

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ConditionalMapBase<Kokkos::HostSpace>::Inverse(StridedMatrix<const double, Kokkos::HostSpace> const& x1,
                                                                                        StridedMatrix<const double, Kokkos::HostSpace> const& r)
{
    this->CheckCoefficients("Inverse");
    // Throw an error if the inputs don't have the same number of columns
    if(x1.extent(1)!=r.extent(1)){
        std::stringstream msg;
        msg << "x1 and r have different numbers of columns.  x1.extent(1)=" << x1.extent(1) << ", but r.extent(1)=" << r.extent(1);
        throw std::invalid_argument(msg.str());
    }

    Kokkos::View<double**, Kokkos::HostSpace> output("Map Inverse Evaluations", this->outputDim, r.extent(1));
    InverseImpl(x1,r, output);
    return output;
}

template<>
Eigen::RowMatrixXd ConditionalMapBase<Kokkos::HostSpace>::Inverse(Eigen::Ref<const Eigen::RowMatrixXd> const& x1, Eigen::Ref<const Eigen::RowMatrixXd> const& r)
{
    this->CheckCoefficients("Inverse");

    StridedMatrix<const double, Kokkos::HostSpace> x1View = ConstRowMatToKokkos<double,Kokkos::HostSpace>(x1);
    StridedMatrix<const double, Kokkos::HostSpace> rView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(r);

    Kokkos::View<double**,Kokkos::LayoutRight,Kokkos::HostSpace> outView = Inverse(x1View, rView);
    return KokkosToMat(outView);
}

#if defined(MPART_ENABLE_GPU)

template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ConditionalMapBase<mpart::DeviceSpace>::Inverse(StridedMatrix<const double, mpart::DeviceSpace> const& x1,
                                                                                          StridedMatrix<const double, mpart::DeviceSpace> const& r)
{
    this->CheckCoefficients("Inverse");
    // Throw an error if the inputs don't have the same number of columns
    if(x1.extent(1)!=r.extent(1)){
        std::stringstream msg;
        msg << "x1 and r have different numbers of columns.  x1.extent(1)=" << x1.extent(1) << ", but r.extent(1)=" << r.extent(1);
        throw std::invalid_argument(msg.str());
    }

    Kokkos::View<double**, mpart::DeviceSpace> output("Map Inverse Evaluations", this->outputDim, r.extent(1));
    InverseImpl(x1,r, output);
    return output;
}

template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ConditionalMapBase<Kokkos::HostSpace>::Inverse(StridedMatrix<const double, mpart::DeviceSpace> const& x1,
                                                                                         StridedMatrix<const double, mpart::DeviceSpace> const& r)
{
    return ToDevice<mpart::DeviceSpace>( this->CoeffGrad(ToHost(x1), ToHost(r)));
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ConditionalMapBase<mpart::DeviceSpace>::Inverse(StridedMatrix<const double, Kokkos::HostSpace> const& x1,
                                                                                        StridedMatrix<const double, Kokkos::HostSpace> const& r)
{
    return ToHost( this->Inverse(ToDevice<mpart::DeviceSpace>(x1), ToDevice<mpart::DeviceSpace>(r)));
}


template<>
Eigen::RowMatrixXd ConditionalMapBase<mpart::DeviceSpace>::Inverse(Eigen::Ref<const Eigen::RowMatrixXd> const& x1, Eigen::Ref<const Eigen::RowMatrixXd> const& r)
{
    StridedMatrix<const double, mpart::DeviceSpace> xView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(x1) );
    StridedMatrix<const double, mpart::DeviceSpace> rView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(r) );

    return KokkosToMat( ToHost( Inverse(xView, rView) ));
}

#endif 

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ConditionalMapBase<Kokkos::HostSpace>::LogDeterminantCoeffGrad(StridedMatrix<const double, Kokkos::HostSpace> const& pts)
{
    this->CheckCoefficients("LogDeterminantCoeffGrad");
    Kokkos::View<double**, Kokkos::HostSpace> output("LogDeterminantCoeffGrad", this->numCoeffs, pts.extent(1));
    LogDeterminantCoeffGradImpl(pts,output);
    return output;
}

template<>
Eigen::RowMatrixXd ConditionalMapBase<Kokkos::HostSpace>::LogDeterminantCoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    this->CheckCoefficients("LogDeterminantCoeffGrad");
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts);

    Kokkos::View<double**,Kokkos::LayoutRight,Kokkos::HostSpace> outView = LogDeterminantCoeffGrad(ptsView);

    return KokkosToMat(outView);
}

#if defined(MPART_ENABLE_GPU)

template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ConditionalMapBase<mpart::DeviceSpace>::LogDeterminantCoeffGrad(StridedMatrix<const double, mpart::DeviceSpace> const& pts)
{
    this->CheckCoefficients("LogDeterminantCoeffGrad");
    Kokkos::View<double**, mpart::DeviceSpace> output("LogDeterminantCoeffGrad", this->numCoeffs, pts.extent(1));
    LogDeterminantCoeffGradImpl(pts,output);
    return output;
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ConditionalMapBase<mpart::DeviceSpace>::LogDeterminantCoeffGrad(StridedMatrix<const double, Kokkos::HostSpace> const& pts)
{
    // Copy the points to the device space 
    StridedMatrix<const double, mpart::DeviceSpace> pts_device = ToDevice<mpart::DeviceSpace>(pts);

    // Evaluate on the device space 
    StridedMatrix<double, mpart::DeviceSpace> evals_device = this->LogDeterminantCoeffGrad(pts_device);

    // Copy back to the host space
    return ToHost(evals_device);
}

template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ConditionalMapBase<Kokkos::HostSpace>::LogDeterminantCoeffGrad(StridedMatrix<const double, mpart::DeviceSpace> const& pts)
{
    // Copy the points to the host 
    StridedMatrix<const double, Kokkos::HostSpace> pts_host = ToHost(pts);

    // Evaluate on the host 
    StridedMatrix<double, Kokkos::HostSpace> evals_host = this->LogDeterminantCoeffGrad(pts_host);

    // Copy back to the device
    return ToDevice<mpart::DeviceSpace>(evals_host);
}


template<>
Eigen::RowMatrixXd ConditionalMapBase<mpart::DeviceSpace>::LogDeterminantCoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    CheckCoefficients("LogDeterminantCoeffGrad");

    Eigen::RowMatrixXd output(outputDim, pts.cols());
    StridedMatrix<const double, mpart::DeviceSpace> ptsView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts));
    return KokkosToMat( ToHost(this->LogDeterminantCoeffGrad(ptsView)));
}


#endif

// Explicit template instantiation
template class mpart::ConditionalMapBase<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::ConditionalMapBase<DeviceSpace>;
#endif