#ifndef MPART_PULLBACKSAMPLER_H
#define MPART_PULLBACKSAMPLER_H

#include "MParT/Distributions/SamplerGenerator.h"
#include "MParT/ConditionalMapBase.h"

namespace mpart {

template<typename MemorySpace>
class PullbackSampler: public SamplerGenerator<MemorySpace> {

    public:
    PullbackSampler() = delete;
    PullbackSampler(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<SamplerGenerator<MemorySpace>> reference): map_(map), reference_(reference) {};

    void SampleImpl(StridedMatrix<double, MemorySpace> output) override {
        unsigned int N = output.extent(1);
        StridedMatrix<double, MemorySpace> pts = reference_->Sample(N);
        Kokkos::View<double**, MemorySpace> prefix_null("prefix_null", 0, pts.extent(1));
        map_->InverseImpl(prefix_null, pts, output);
    };

    private:
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    std::shared_ptr<SamplerGenerator<MemorySpace>> reference_;
};

} // namespace mpart

#endif // MPART_PULLBACKSAMPLER_H