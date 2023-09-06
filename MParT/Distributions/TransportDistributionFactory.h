#ifndef MPART_TRANSPORTDISTRIBUTIONFACTORY_H
#define MPART_TRANSPORTDISTRIBUTIONFACTORY_H

#include "Distribution.h"
#include "PullbackDensity.h"
#include "GaussianSamplerDensity.h"
#include "PushforwardDensity.h"
#include "PullbackSampler.h"
#include "PushforwardSampler.h"

namespace mpart {
    namespace TransportDistributionFactory {

    template<typename MemorySpace>
    std::shared_ptr<Distribution<MemorySpace>> CreatePullback(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<Distribution<MemorySpace>> base) {
        auto sampler = std::make_shared<PullbackSampler<MemorySpace>>(map, base->GetSampler());
        auto density = std::make_shared<PullbackDensity<MemorySpace>>(map, base->GetDensity());
        return std::make_shared<Distribution<MemorySpace>>(sampler, density);
    };

    template<typename MemorySpace>
    std::shared_ptr<Distribution<MemorySpace>> CreatePushforward(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<Distribution<MemorySpace>> base) {
        auto sampler = std::make_shared<PushforwardSampler<MemorySpace>>(map, base->GetSampler());
        auto density = std::make_shared<PushforwardDensity<MemorySpace>>(map, base->GetDensity());
        return std::make_shared<Distribution<MemorySpace>>(sampler, density);
    };

    template<typename MemorySpace>
    std::shared_ptr<Distribution<MemorySpace>> CreateGaussianPullback(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
        unsigned int outputDim = map->outputDim;
        auto dist = std::make_shared<GaussianSamplerDensity<MemorySpace>>(outputDim);
        auto sampler = std::make_shared<PullbackSampler<MemorySpace>>(map, dist);
        auto density = std::make_shared<PullbackDensity<MemorySpace>>(map, dist);
        return std::make_shared<Distribution<MemorySpace>>(sampler, density);
    }
    template<typename MemorySpace>
    std::shared_ptr<Distribution<MemorySpace>> CreateGaussianPushforward(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
        unsigned int inputDim = map->outputDim;
        auto dist = std::make_shared<GaussianSamplerDensity<MemorySpace>>(outputDim);
        auto sampler = std::make_shared<PullbackSampler<MemorySpace>>(map, dist);
        auto density = std::make_shared<PullbackDensity<MemorySpace>>(map, dist);
        return std::make_shared<Distribution<MemorySpace>>(sampler, density);
    }
    }
};

#endif // MPART_TRANSPORTDISTRIBUTIONFACTORY_H