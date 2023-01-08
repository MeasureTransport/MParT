#ifndef MPART_PUSHFORWARDSAMPLER_H
#define MPART_PUSHFORWARDSAMPLER_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/ConditionalMapBase.h"

namespace mpart {

template<typename MemorySpace>
class PushforwardSampler: public SampleGenerator<MemorySpace> {

    public:
    PushforwardSampler() = delete;
    PushforwardSampler(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<SampleGenerator<MemorySpace>> reference):
    SampleGenerator<MemorySpace>(reference->Dim()), map_(map), reference_(reference) {};

    void SampleImpl(StridedMatrix<double, MemorySpace> output) override {
        unsigned int N = output.extent(1);
        StridedMatrix<double, MemorySpace> pts = reference_->Sample(N);
        map_->EvaluateImpl(pts, output);
    };

    void SetSeed(unsigned int seed) override {
        SampleGenerator<MemorySpace>::SetSeed(seed);
        reference_->SetSeed(seed);
    }

    private:
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    std::shared_ptr<SampleGenerator<MemorySpace>> reference_;
};

} // namespace mpart

#endif // MPART_PUSHFORWARDSAMPLER_H