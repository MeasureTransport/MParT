#ifndef MPART_PULLBACKDENSITY_H
#define MPART_PULLBACKDENSITY_H

#include "MParT/Distributions/DensityBase.h"
#include "MParT/Utilities/LinearAlgebra.h"
#include "MParT/ConditionalMapBase.h"

namespace mpart {

template<typename MemorySpace>
class PullbackDensity: public DensityBase<MemorySpace {

    public:
    PullbackDensity() = delete;
    PullbackDensity(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<DensityBase<MemorySpace>> reference): map_(map), reference_(reference) {};

    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) override {
        StridedMatrix<double, MemorySpace> mappedPts = map_->Evaluate(pts);
        reference_->LogDensityImpl(mappedPts, output);
        StridedVector<double, MemorySpace> logJacobian = map_->LogDeterminant(pts);
        output += logJacobian;
    };

    void GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) override {
        StridedMatrix<double, MemorySpace> mappedPts = map_->Evaluate(pts);
        StridedMatrix<double, MemorySpace> sens_map = reference_->GradLogDensity(mappedPts);
        map_->GradientImpl(mappedPts, sens_map, output);
        StridedMatrix<double, MemorySpace> gradLogJacobian = map_->LogDeterminantInputGrad(pts);
        output += gradLogJacobian;
    };

    private:
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    std::shared_ptr<DensityBase<MemorySpace>> reference_;
};

} // namespace mpart

#endif //MPART_PULLBACKDENSITY_H