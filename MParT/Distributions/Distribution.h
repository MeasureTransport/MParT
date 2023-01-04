#ifndef MPART_Distribution_H
#define MPART_Distribution_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/DensityBase.h"

namespace mpart {

template<typename MemorySpace, typename SamplerType, typename DensityType>
class Distribution: public SamplerType, public DensityType {
    public:
    Distribution() = delete;
    Distribution(SamplerType &sampler, DensityType &density): SamplerType(sampler), DensityType(density) {
        if(sampler.dim_ != density.dim_) {
            throw std::runtime_error("Dimension mismatch between sampler and density.");
        }
    };

    virtual ~Distribution() = default;

    unsigned int Dim() const { return SamplerType::dim_; };

};

} // namespace mpart

#endif //MPART_Distribution_H