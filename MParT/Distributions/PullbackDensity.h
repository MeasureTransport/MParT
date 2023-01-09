#ifndef MPART_PULLBACKDENSITY_H
#define MPART_PULLBACKDENSITY_H

#include "MParT/Distributions/DensityBase.h"
#include "MParT/Utilities/LinearAlgebra.h"
#include "MParT/ConditionalMapBase.h"

namespace mpart {

template<typename MemorySpace>
class PullbackDensity: public DensityBase<MemorySpace> {

    public:
    PullbackDensity() = delete;
    PullbackDensity(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<DensityBase<MemorySpace>> reference):
        DensityBase<MemorySpace>(reference->Dim()), map_(map), reference_(reference) {
        if (map_->outputDim != reference_->Dim()) {
            throw std::invalid_argument("PullbackDensity: map output dimension does not match reference density dimension");
        }
        if (map_->inputDim != map_->outputDim) {
            throw std::invalid_argument("PullbackDensity: map input dimension does not match map output dimension");
        }
    }

    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) override {
        StridedMatrix<double, MemorySpace> mappedPts = map_->Evaluate(pts);
        reference_->LogDensityImpl(mappedPts, output);
        StridedVector<double, MemorySpace> logJacobian = map_->LogDeterminant(pts);
        output += logJacobian;
    };

    void GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) override {
        StridedMatrix<const double, MemorySpace> mappedPts = map_->Evaluate(pts);
        StridedMatrix<double, MemorySpace> sens_map = reference_->GradLogDensity(mappedPts);
        map_->GradientImpl(pts, sens_map, output);
        StridedMatrix<double, MemorySpace> gradLogJacobian = map_->LogDeterminantInputGrad(pts);
        output += gradLogJacobian;
    };

    private:
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    std::shared_ptr<DensityBase<MemorySpace>> reference_;
};

} // namespace mpart

#endif //MPART_PULLBACKDENSITY_H