#ifndef MPART_Distribution_H
#define MPART_Distribution_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/DensityBase.h"

namespace mpart {

template<typename MemorySpace, typename SamplerType, typename DensityType>
class Distribution {
    public:
    Distribution() = delete;
    Distribution(std::shared_ptr<SamplerType> sampler, std::shared_ptr<DensityType> density): sampler_(sampler), density_(density) {
        if(sampler->dim_ != density->dim_) {
            throw std::runtime_error("Dimension mismatch between sampler and density.");
        }
    };

    virtual ~Distribution() = default;

    private:
    std::shared_ptr<SamplerType> sampler_;
    std::shared_ptr<DensityType> density_;

};

} // namespace mpart

#endif //MPART_Distribution_H