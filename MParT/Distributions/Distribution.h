#ifndef MPART_Distribution_H
#define MPART_Distribution_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/DensityBase.h"

namespace mpart {

template<typename MemorySpace, typename SamplerType, typename DensityType>
class Distribution{
    public:
    Distribution() = delete;
    Distribution(std::shared_ptr<SamplerType> sampler, std::shared_ptr<DensityType> density): sampler_(sampler), density_(density) {
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

    StridedMatrix<double, MemorySpace> GradLogDensity(StridedMatrix<const double, MemorySpace> const &pts) {
        return density_->GradLogDensity(pts);
    };
    void GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) {
        density_->GradLogDensityImpl(pts, output);
    };

    std::shared_ptr<SamplerType> GetSampler() const { return sampler_; };
    std::shared_ptr<DensityType> GetDensity() const { return density_; };

    private:
    std::shared_ptr<SamplerType> sampler_;
    std::shared_ptr<DensityType> density_;

}; // class Distribution

template<typename MemorySpace, typename SamplerDensity, typename... T>
std::shared_ptr<Distribution<MemorySpace, SamplerDensity, SamplerDensity>> CreateDistribution(T... args) {
    std::shared_ptr<SamplerDensity> samplerDensity = std::make_shared<SamplerDensity>(args...);
    return std::make_shared<Distribution<MemorySpace, SamplerDensity, SamplerDensity>>(samplerDensity, samplerDensity);
};

} // namespace mpart

#endif //MPART_Distribution_H