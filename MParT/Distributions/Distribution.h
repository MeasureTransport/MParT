#ifndef MPART_Distribution_H
#define MPART_Distribution_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/DensityBase.h"

namespace mpart {

template<typename MemorySpace>
class Distribution{
    public:
    Distribution() = delete;
    Distribution(std::shared_ptr<SampleGenerator<MemorySpace>> sampler, std::shared_ptr<DensityBase<MemorySpace>> density): sampler_(sampler), density_(density) {
        if(sampler->Dim() != density->Dim()) {
            throw std::runtime_error("Dimension mismatch between sampler and density.");
        }
    };

    virtual ~Distribution() = default;

    unsigned int Dim() const { return sampler_->Dim(); };

    void SetSeed(unsigned int seed) {
        sampler_->SetSeed(seed);
    };

    StridedMatrix<double, MemorySpace> Sample(unsigned int N) {
        return sampler_->Sample(N);
    };
    void SampleImpl(StridedMatrix<double, MemorySpace> output) {
        sampler_->SampleImpl(output);
    };

    StridedVector<double, MemorySpace> LogDensity(StridedMatrix<const double, MemorySpace> const &pts) {
        return density_->LogDensity(pts);
    };
    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) {
        density_->LogDensityImpl(pts, output);
    };

    StridedMatrix<double, MemorySpace> LogDensityInputGrad(StridedMatrix<const double, MemorySpace> const &pts) {
        return density_->LogDensityInputGrad(pts);
    };
    void LogDensityInputGradImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) {
        density_->LogDensityInputGradImpl(pts, output);
    };

    std::shared_ptr<SampleGenerator<MemorySpace>> GetSampler() const { return sampler_; };
    std::shared_ptr<DensityBase<MemorySpace>> GetDensity() const { return density_; };

    private:
    std::shared_ptr<SampleGenerator<MemorySpace>> sampler_;
    std::shared_ptr<DensityBase<MemorySpace>> density_;

}; // class Distribution

template<typename MemorySpace, typename SamplerDensity, typename... T>
std::shared_ptr<Distribution<MemorySpace>> CreateDistribution(T... args) {
    std::shared_ptr<SamplerDensity> samplerDensity = std::make_shared<SamplerDensity>(args...);
    return std::make_shared<Distribution<MemorySpace>>(samplerDensity, samplerDensity);
};

} // namespace mpart

#endif //MPART_Distribution_H