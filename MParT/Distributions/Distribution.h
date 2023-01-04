#ifndef MPART_Distribution_H
#define MPART_Distribution_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/DensityBase.h"

namespace mpart {

template<typename MemorySpace>
class Distribution: public SampleGenerator<MemorySpace>, public DensityBase<MemorySpace> {
    public:
    Distribution() = delete;
    Distribution(int dim_): SampleGenerator<MemorySpace>(dim_), DensityBase<MemorySpace>(dim_) {}
    Distribution(SampleGenerator<MemorySpace> &sampler, DensityBase<MemorySpace> &density): SampleGenerator<MemorySpace>(sampler), DensityBase<MemorySpace>(density) {
        if(sampler->dim_ != density->dim_) {
            throw std::runtime_error("Dimension mismatch between sampler and density.");
        }
    };

    virtual ~Distribution() = default;
};

} // namespace mpart

#endif //MPART_Distribution_H