#ifndef MPART_PULLBACKDENSITY_H
#define MPART_PULLBACKDENSITY_H

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/DensityBase.h"
#include "MParT/Utilities/LinearAlgebra.h"
#include "MParT/ConditionalMapBase.h"

namespace mpart {

template<typename MemorySpace>
class PullbackDensity: public DensityBase<MemorySpace> {

    public:
    PullbackDensity() = delete;
    PullbackDensity(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<DensityBase<MemorySpace>> reference);

    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) override;

    void GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) override;

    void CoeffGradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output);

    StridedMatrix<double, MemorySpace> CoeffGradLogDensity(StridedMatrix<const double, MemorySpace> const &pts);

    Eigen::RowMatrixXd CoeffGradLogDensity(Eigen::Ref<const Eigen::RowMatrixXd> const &pts);

    private:
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    std::shared_ptr<DensityBase<MemorySpace>> reference_;
};

} // namespace mpart

#endif //MPART_PULLBACKDENSITY_H