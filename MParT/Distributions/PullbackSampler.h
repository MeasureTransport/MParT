#ifndef MPART_PULLBACKSAMPLER_H
#define MPART_PULLBACKSAMPLER_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/ConditionalMapBase.h"

namespace mpart {

/**
 * @brief A class to sample from the pullback distribution of a transport map given the map and a sampler.
 *        If \f$X\sim\mu\f$, and \f$T\f$ satisfies \f$T(X)\sim\nu\f$, then this class allows you to sample from
 *        \f$\mu\f$ given a sampler for \f$\nu\f$ and the map \f$T\f$.
 *
 * @tparam MemorySpace Space where data is stored for computation
 */
template<typename MemorySpace>
class PullbackSampler: public SampleGenerator<MemorySpace> {

    public:
    PullbackSampler() = delete;
    /**
     * @brief Construct a new Pullback Sampler object from the transport map and object to sample from \f$\nu\f$
     *
     * @param map transport map with pushforward \f$\nu\f$
     * @param sampler sampler of \f$\nu\f$
     */
    PullbackSampler(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<SampleGenerator<MemorySpace>> sampler):
    SampleGenerator<MemorySpace>(sampler->Dim()), map_(map), sampler_(sampler) {
        if (map_->outputDim != sampler_->Dim()) {
            throw std::invalid_argument("PullbackSampler: map output dimension does not match sampler sampler dimension");
        }
        if (map_->inputDim != map_->outputDim) {
            throw std::invalid_argument("PullbackSampler: map input dimension does not match map output dimension");
        }
    }

    /**
     * @brief Sample from \f$\mu\f$ using the pullback distribution
     *
     * @param output (MxN) matrix, where M is the dimension of \f$\mu\f$ and N is the number of samples
     */
    void SampleImpl(StridedMatrix<double, MemorySpace> output) override {
        unsigned int N = output.extent(1);
        StridedMatrix<double, MemorySpace> pts = sampler_->Sample(N);
        Kokkos::View<double**, MemorySpace> prefix_null("prefix_null", 0, pts.extent(1));
        map_->InverseImpl(prefix_null, pts, output);
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

#endif // MPART_PULLBACKSAMPLER_H