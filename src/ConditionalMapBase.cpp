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

template<typename MemorySpace>
Eigen::VectorXd ConditionalMapBase<MemorySpace>::LogDeterminant(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    this->CheckCoefficients("LogDeterminant");

    StridedMatrix<const double,MemorySpace> ptsView = ConstRowMatToKokkos<double,MemorySpace>(pts);
    Kokkos::View<double*,MemorySpace> outView = LogDeterminant(ptsView);
    return KokkosToVec(outView);
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

template<typename MemorySpace>
Eigen::RowMatrixXd ConditionalMapBase<MemorySpace>::Inverse(Eigen::Ref<const Eigen::RowMatrixXd> const& x1, Eigen::Ref<const Eigen::RowMatrixXd> const& r)
{
    this->CheckCoefficients("Inverse");

    StridedMatrix<const double, MemorySpace> x1View = ConstRowMatToKokkos<double,MemorySpace>(x1);
    StridedMatrix<const double, MemorySpace> rView = ConstRowMatToKokkos<double,MemorySpace>(r);

    Kokkos::View<double**,Kokkos::LayoutRight,MemorySpace> outView = Inverse(x1View, rView);
    return KokkosToMat(outView);
}

template<typename MemorySpace>
StridedMatrix<double, MemorySpace> ConditionalMapBase<MemorySpace>::LogDeterminantCoeffGrad(StridedMatrix<const double, MemorySpace> const& pts)
{
    this->CheckCoefficients("LogDeterminantCoeffGrad");
    Kokkos::View<double**, MemorySpace> output("LogDeterminantCoeffGrad", this->numCoeffs, pts.extent(1));
    LogDeterminantCoeffGradImpl(pts,output);
    return output;
}

template<typename MemorySpace>
Eigen::RowMatrixXd ConditionalMapBase<MemorySpace>::LogDeterminantCoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    this->CheckCoefficients("LogDeterminantCoeffGrad");
    StridedMatrix<const double, MemorySpace> ptsView = ConstRowMatToKokkos<double,MemorySpace>(pts);

    Kokkos::View<double**,Kokkos::LayoutRight,MemorySpace> outView = LogDeterminantCoeffGrad(ptsView);

    return KokkosToMat(outView);
}


// Explicit template instantiation
template class mpart::ConditionalMapBase<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>;
#endif