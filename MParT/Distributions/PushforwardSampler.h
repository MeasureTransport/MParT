#ifndef MPART_PUSHFORWARDSAMPLER_H
#define MPART_PUSHFORWARDSAMPLER_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/ConditionalMapBase.h"

namespace mpart {

/**
 * @brief A class to sample from the pushforward distribution of a transport map given the map and a sampler.
 *        If \f$X\sim\mu\f$, and \f$T\f$ satisfies \f$T(X)\sim\nu\f$, then this class allows you to sample from
 *        \f$\nu\f$ given a sampler for \f$\mu\f$ and the map \f$T\f$.
 *
 * @tparam MemorySpace Space where data is stored for computation
 */
template<typename MemorySpace>
class PushforwardSampler: public SampleGenerator<MemorySpace> {

    public:
    PushforwardSampler() = delete;

    /**
     * @brief Construct a new Pushforward Sampler object from map \f$T\f$ and sampler of \f$\mu\f$
     *
     * @param map Transport map \f$T:\mu\to\nu\f$
     * @param sampler Object to sample according to measure \f$\mu\f$
     */
    PushforwardSampler(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<SampleGenerator<MemorySpace>> sampler):
    SampleGenerator<MemorySpace>(sampler->Dim()), map_(map), sampler_(sampler) {
        if (map_->outputDim != sampler_->Dim()) {
            throw std::invalid_argument("PushforwardSampler: map output dimension does not match sampler dimension");
        }
        if (map_->inputDim != map_->outputDim) {
            throw std::invalid_argument("PushforwardSampler: map input dimension does not match map output dimension");
        }
    }

    /**
     * @brief Sample from \f$\nu\f$ using the pushforward distribution
     *
     * @param output (MxN) matrix, where M is the dimension of \f$\nu\f$ and N is the number of samples
     */
    void SampleImpl(StridedMatrix<double, MemorySpace> output) override {
        unsigned int N = output.extent(1);
        StridedMatrix<double, MemorySpace> pts = sampler_->Sample(N);
        map_->EvaluateImpl(pts, output);
    };

    /**
     * @brief Set the Seed for the random Pool of the sampler for reproducibility
     *
     * @param seed new seed to use
     */
    void SetSeed(unsigned int seed) override {
        SampleGenerator<MemorySpace>::SetSeed(seed);
        sampler_->SetSeed(seed);
    }

    private:
    /**
     * @brief Transport map \f$T:\mu\to\nu\f$
     *
     */
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    /**
     * @brief Object to sample from measure \f$\nu\f$
     *
     */
    std::shared_ptr<SampleGenerator<MemorySpace>> sampler_;
};

} // namespace mpart

#endif // MPART_PUSHFORWARDSAMPLER_H