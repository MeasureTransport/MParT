#include "MParT/MarginalAffineMap.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/Miscellaneous.h"


using namespace mpart;

template<typename MemorySpace>
MarginalAffineMap<MemorySpace>::MarginalAffineMap(StridedVector<double,MemorySpace> scale,
                                  StridedVector<double,MemorySpace> shift,
                                  std::shared_ptr<ConditionalMapBase<MemorySpace>> map,
                                  bool moveCoeffs) :
                                  ConditionalMapBase<MemorySpace>(map->inputDim, map->outputDim, map->numCoeffs),
                                  scale_("scale", scale.extent(0)),
                                  shift_("shift", shift.extent(0)),
                                  map_(map)
{
    Kokkos::deep_copy(scale_, scale);
    Kokkos::deep_copy(shift_, shift);
    if (scale_.size() != shift_.size())
        ProcAgnosticError<MemorySpace, std::runtime_error>::error("MarginalAffineMap: scale and shift must have the same size");
    if (scale_.size() != map->inputDim)
        ProcAgnosticError<MemorySpace, std::runtime_error>::error("MarginalAffineMap: scale and shift must have the same size as the input dimension of the map");

    logDet_ = 0.;
    Kokkos::parallel_reduce("MarginalAffineMap logdet", scale.extent(0), KOKKOS_LAMBDA(const int&i, double& ldet){
        ldet += Kokkos::log(scale_(i));
    }, logDet_);
    this->SetCoeffs(map->Coeffs());
    if (moveCoeffs) {
        map->WrapCoeffs(this->savedCoeffs);
    }
}

template<typename MemorySpace>
void MarginalAffineMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                StridedVector<double, MemorySpace>              output)
{
    int n1 = pts.extent(0), n2 = pts.extent(1);
    Kokkos::View<double**, MemorySpace> tmp("tmp", n1, n2);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({0, 0}, {n1, n2});
    Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& i, const int& j) {
        tmp(i,j) = pts(i,j)*scale_(i) + shift_(i);
    });
    map_->LogDeterminantImpl(tmp, output);
    Kokkos::parallel_for("MarginalAffineMap LogDeterminant", output.size(), KOKKOS_CLASS_LAMBDA(const int& i) {
        output(i) += logDet_;
    });
}

template<typename MemorySpace>
void MarginalAffineMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                         StridedMatrix<const double, MemorySpace> const& r,
                                         StridedMatrix<double, MemorySpace>              output)
{
    int x1_n1 = this->inputDim - this->outputDim, x1_n2 = x1.extent(1);
    Kokkos::View<const double**, MemorySpace> x1_tmp;
    if(x1_n1 > 0) {
        Kokkos::View<double**, MemorySpace> x1_t("x1_tmp", x1_n1, x1_n2);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> prefix_policy({0, 0}, {x1_n1, x1_n2});
        Kokkos::parallel_for(prefix_policy, KOKKOS_CLASS_LAMBDA(const int& i, const int& j) {
            x1_t(i,j) = x1(i,j)*scale_(i) + shift_(i);
        });
        x1_tmp = x1_t;
    } else {
        x1_tmp = x1;
    }
    map_->InverseImpl(x1_tmp, r, output);
    int n1 = output.extent(0), n2 = output.extent(1);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> output_policy({0, 0}, {n1, n2});
    Kokkos::parallel_for(output_policy, KOKKOS_CLASS_LAMBDA(const int& i, const int& j) {
        int input_idx = i+x1_n1;
        output(i,j) = (output(i,j) - shift_(input_idx))/scale_(input_idx);
    });
}



template<typename MemorySpace>
void MarginalAffineMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                         StridedMatrix<double, MemorySpace>              output)
{
    int n1 = pts.extent(0), n2 = pts.extent(1);
    Kokkos::View<double**, MemorySpace> tmp("tmp", n1, n2);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({0, 0}, {n1, n2});
    Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& i, const int& j) {
        tmp(i,j) = pts(i,j)*scale_(i) + shift_(i);
    });
    map_->LogDeterminantCoeffGradImpl(tmp, output);
}

template<typename MemorySpace>
void MarginalAffineMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                          StridedMatrix<double, MemorySpace>              output)
{
    int n1 = pts.extent(0), n2 = pts.extent(1);
    Kokkos::View<double**, MemorySpace> tmp("tmp", n1, n2);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({0, 0}, {n1, n2});
    Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& i, const int& j) {
        tmp(i,j) = pts(i,j)*scale_(i) + shift_(i);
    });
    map_->EvaluateImpl(tmp, output);
}

template<typename MemorySpace>
void MarginalAffineMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                           StridedMatrix<const double, MemorySpace> const& sens,
                                           StridedMatrix<double, MemorySpace>              output)
{
    int n1 = pts.extent(0), n2 = pts.extent(1);
    Kokkos::View<double**, MemorySpace> tmp("tmp", n1, n2);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({0, 0}, {n1, n2});
    Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& i, const int& j) {
        tmp(i,j) = pts(i,j)*scale_(i) + shift_(i);
    });
    map_->CoeffGradImpl(tmp, sens, output);
}

template<typename MemorySpace>
void MarginalAffineMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                         StridedMatrix<double, MemorySpace>              output)
{
    int n1 = pts.extent(0), n2 = pts.extent(1);
    Kokkos::View<double**, MemorySpace> tmp("tmp", n1, n2);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({0, 0}, {n1, n2});
    Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& i, const int& j) {
        tmp(i,j) = pts(i,j)*scale_(i) + shift_(i);
    });
    map_->LogDeterminantInputGradImpl(tmp, output);
    int out_n1 = output.extent(0), out_n2 = output.extent(1);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy2({0, 0}, {out_n1, out_n2});
    int scale_idx_start = map_->inputDim-map_->outputDim;
    Kokkos::parallel_for("MarginalAffineMap LogDeterminantInputGrad", policy2, KOKKOS_CLASS_LAMBDA(const int& i, const int& j) {
        int scale_idx = i+scale_idx_start;
        output(i,j) *= scale_(scale_idx);
    });
}

template<typename MemorySpace>
void MarginalAffineMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                          StridedMatrix<const double, MemorySpace> const& sens,
                                          StridedMatrix<double, MemorySpace>              output)
{
    int n1 = pts.extent(0), n2 = pts.extent(1);
    Kokkos::View<double**, MemorySpace> tmp("tmp", n1, n2);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({0, 0}, {n1, n2});
    Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& i, const int& j) {
        tmp(i,j) = pts(i,j)*scale_(i) + shift_(i);
    });
    map_->GradientImpl(tmp, sens, output);
    int out_n1 = output.extent(0), out_n2 = output.extent(1);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy2({0, 0}, {out_n1, out_n2});
    Kokkos::parallel_for("MarginalAffineMap Gradient", policy2, KOKKOS_CLASS_LAMBDA(const int& i, const int& j) {
        output(i,j) *= scale_(i);
    });
}


template class mpart::MarginalAffineMap<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::MarginalAffineMap<mpart::DeviceSpace>;
#endif