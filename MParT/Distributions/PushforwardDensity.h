#ifndef MPART_PUSHFORWARDDENSITY_H
#define MPART_PUSHFORWARDDENSITY_H

#include "MParT/Distributions/DensityBase.h"
#include "MParT/Utilities/LinearAlgebra.h"
#include "MParT/ConditionalMapBase.h"

namespace mpart {

template<typename MemorySpace>
class PushforwardDensity: public DensityBase<MemorySpace> {

    public:
    PushforwardDensity() = delete;
    PushforwardDensity(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<DensityBase<MemorySpace>> reference):
        DensityBase<MemorySpace>(reference->Dim()), map_(map), reference_(reference) {
        if (map_->inputDim != reference_->Dim()) {
            throw std::invalid_argument("PushforwardDensity: map output dimension does not match reference density dimension");
        }
        if (map_->outputDim != map_->inputDim) {
            throw std::invalid_argument("PushforwardDensity: map input dimension does not match map output dimension");
        }
    }

    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) override {
        Kokkos::View<double**, MemorySpace> prefix_null("prefix_null", 0, pts.extent(1));
        StridedMatrix<double, MemorySpace> mappedPts = map_->Inverse(prefix_null, pts);
        reference_->LogDensityImpl(mappedPts, output);
        StridedVector<double, MemorySpace> logJacobian = map_->LogDeterminant(mappedPts);
        output += logJacobian;
    };

    void GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) override {
        throw std::runtime_error("GradLogDensity not implemented for PushforwardDensity");
    };

    private:
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    std::shared_ptr<DensityBase<MemorySpace>> reference_;
};

} // namespace mpart

#endif //MPART_PUSHFORWARDDENSITY_H