#ifndef MPART_Distribution_H
#define MPART_Distribution_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/DensityBase.h"

namespace mpart {

template<typename MemorySpace>
class Distribution{
    public:
    Distribution() = delete;
    Distribution(std::shared_ptr<SampleGenerator<MemorySpace>> sampler, std::shared_ptr<DensityBase<MemorySpace>> density): sampler_(sampler), density_(density), dim_(sampler.dim_) {
        if(sampler_->dim_ != density_->dim_) {
            throw std::runtime_error("Dimension mismatch between sampler and density.");
        }
    };

    virtual ~Distribution() = default;

    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> &output) {
        return density_->LogDensityImpl(pts, output);
    }

    StridedVector<double, MemorySpace> LogDensity(StridedMatrix<const double, MemorySpace> const &pts) {
        return density_->LogDensity(pts);
    }

    void GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) {
        return density_->GradLogDensityImpl(pts, output);
    }

    StridedMatrix<double, MemorySpace> GradLogDensity(StridedMatrix<const double, MemorySpace> const &pts) {
        return density_->GradLogDensity(pts);
    }

    void SampleImpl(StridedMatrix<double, MemorySpace> output) {
        return sampler_->SampleImpl(output);
    }

    StridedMatrix<double, MemorySpace> Sample(unsigned int N) {
        return sampler_->Sample(N);
    }

    const unsigned int dim_;
    private:
    std::shared_ptr<SampleGenerator<MemorySpace>> sampler_;
    std::shared_ptr<DensityBase<MemorySpace>> density_;
};

template<typename MemorySpace, template<typename> typename DensitySampler, typename... T>
Distribution<MemorySpace, DensitySampler<MemorySpace>, DensitySampler<MemorySpace>> CreateDistribution(T... args) {
    auto dist = std::make_shared<DensitySampler<MemorySpace>>(args...);
    return Distribution<MemorySpace>(dist, dist);
}

} // namespace mpart

#endif //MPART_Distribution_H