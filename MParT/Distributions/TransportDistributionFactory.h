#ifndef MPART_TRANSPORTDISTRIBUTIONFACTORY_H
#define MPART_TRANSPORTDISTRIBUTIONFACTORY_H

#include "Distribution.h"
#include "PullbackDensity.h"
#include "PushforwardDensity.h"
#include "PullbackSampler.h"
#include "PushforwardSampler.h"

namespace mpart {

    template<typename MemorySpace>
    using PullbackDistribution = Distribution<PullbackDensity<MemorySpace>, PullbackSampler<MemorySpace>>;
    template<typename MemorySpace>
    using PushforwardDistribution = Distribution<PushforwardDensity<MemorySpace>, PushforwardSampler<MemorySpace>>;

    template<typename MemorySpace>
    std::shared_ptr<PullbackDistribution<MemorySpace>> CreatePullback(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<Distribution<MemorySpace>> reference) {
        auto sampler = std::make_shared<PullbackSampler<MemorySpace>>(map, reference->GetSampler());
        auto density = std::make_shared<PullbackDensity<MemorySpace>>(map, reference->GetDensity());
        return std::make_shared<PullbackDistribution<MemorySpace>>(sampler, density);
    };

    template<typename MemorySpace>
    std::shared_ptr<PushforwardDistribution<MemorySpace>> CreatePushforward(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<Distribution<MemorySpace>> reference) {
        auto sampler = std::make_shared<PushforwardSampler<MemorySpace>>(map, reference->GetSampler());
        auto density = std::make_shared<PushforwardDensity<MemorySpace>>(map, reference->GetDensity());
        return std::make_shared<PushforwardDistribution<MemorySpace>>(sampler, density);
    };

};

#endif // MPART_TRANSPORTDISTRIBUTIONFACTORY_H