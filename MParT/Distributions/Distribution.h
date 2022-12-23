#ifndef MPART_Distribution_H
#define MPART_Distribution_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/DensityBase.h"

namespace mpart {

template<typename MemorySpace>
class Distribution: public SampleGenerator, public Density {
    public:
    Distribution() = delete;
    Distribution(SampleGenerator<MemorySpace> &sampler, DensityBase<MemorySpace> &density): SampleGenerator(sampler), DensityBase(density) {
        if(sampler_->dim_ != density_->dim_) {
            throw std::runtime_error("Dimension mismatch between sampler and density.");
        }
    };

    virtual ~Distribution() = default;
};

} // namespace mpart

#endif //MPART_Distribution_H