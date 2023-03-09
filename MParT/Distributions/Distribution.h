#ifndef MPART_Distribution_H
#define MPART_Distribution_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/DensityBase.h"

namespace mpart {

/**
 * @brief An object that represents a probability distribution, binding a density and sampler.
 *        We assume density integrates to unity for this class.
 * @tparam MemorySpace where the density and sampler store data
 */
template<typename MemorySpace>
class Distribution{
    public:
    Distribution() = delete;

    /**
     * @brief Construct a new Distribution object from a sampler and a density
     *
     * @param sampler an object to generate samples from a distribution
     * @param density an object to evaluate the (normalized) density of a distribution
     */
    Distribution(std::shared_ptr<SampleGenerator<MemorySpace>> sampler, std::shared_ptr<DensityBase<MemorySpace>> density): sampler_(sampler), density_(density) {
        // Ensure the sampler and density have the same dimension
        if(sampler->Dim() != density->Dim()) {
            throw std::runtime_error("Dimension mismatch between sampler and density.");
        }
    };

    virtual ~Distribution() = default;

    /**
     * @brief Return the dimension of this distribution
     *
     * @return unsigned int dimension of the distribution
     */
    unsigned int Dim() const { return sampler_->Dim(); };

    /**
     * @brief Set the Seed for the sampler
     *
     * @param seed the new seed for the sampler in this distribution
     */
    void SetSeed(unsigned int seed) {
        sampler_->SetSeed(seed);
    };

    /**
     * @brief Wrap around the sample method from the sampler
     *
     * @param N The number of samples to take
     * @return StridedMatrix<double, MemorySpace> samples from the sampler
     */
    StridedMatrix<double, MemorySpace> Sample(unsigned int N) {
        return sampler_->Sample(N);
    };

    /**
     * @brief Wrap around the SampleImpl method from the sampler
     *
     * @param output where to store all the samples
     */
    void SampleImpl(StridedMatrix<double, MemorySpace> output) {
        sampler_->SampleImpl(output);
    };

    /**
     * @brief Wrap around the LogDensity method from the density
     *
     * @param pts dim x N data matrix where N is the number of points
     * @return StridedVector<double, MemorySpace> N-length vector of density evaluations
     */
    StridedVector<double, MemorySpace> LogDensity(StridedMatrix<const double, MemorySpace> const &pts) {
        return density_->LogDensity(pts);
    };

    /**
     * @brief Wrap around the LogDensityImpl method from the density
     *
     * @param pts dim x N data matrix where N is the number of points
     * @param output N-length vector for density evaluations
     */
    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) {
        density_->LogDensityImpl(pts, output);
    };

    /**
     * @brief Wrap around the LogDensityInputGrad method from the density
     *
     * @param pts dim x N data matrix where N is the number of points
     * @return StridedMatrix<double, MemorySpace> dim x N matrix with \f$\frac{\partial p}{\partial x_i}(\mathbf{t}_j)\f$ in the (i,j) entry
     */
    StridedMatrix<double, MemorySpace> LogDensityInputGrad(StridedMatrix<const double, MemorySpace> const &pts) {
        return density_->LogDensityInputGrad(pts);
    };

    /**
     * @brief Wrap around the LogDensityInputGradImpl method from the density
     *
     * @param pts dim x N data matrix where N is the number of points
     * @param output dim x N matrix with \f$\frac{\partial p}{\partial x_i}(\mathbf{t}_j)\f$ in the (i,j) entry (where \f$\mathbf{t}_j\f$ is the j-th data point)
     */
    void LogDensityInputGradImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) {
        density_->LogDensityInputGradImpl(pts, output);
    };

    /**
     * @brief Get the Sampler object from the distribution
     *
     * @return std::shared_ptr<SampleGenerator<MemorySpace>> a shared pointer to the sampler that this object uses
     */
    std::shared_ptr<SampleGenerator<MemorySpace>> GetSampler() const { return sampler_; };

    /**
     * @brief Get the Density object from the distribution
     *
     * @return std::shared_ptr<DensityBase<MemorySpace>> a shared pointer to the density this object uses
     */
    std::shared_ptr<DensityBase<MemorySpace>> GetDensity() const { return density_; };

    private:
    std::shared_ptr<SampleGenerator<MemorySpace>> sampler_;
    std::shared_ptr<DensityBase<MemorySpace>> density_;

}; // class Distribution

/**
 * @brief Convenient way to create a Distribution object from arguments to create an object that's both a sampler + density
 *
 * @tparam MemorySpace Where the data is stored for computation
 * @tparam SamplerDensity The type of object that's both a sampler and a density
 * @tparam T Parameter pack representing the argument types for the SamplerDensity constructor
 * @param args arguments to create the SamplerDensity instance
 * @return std::shared_ptr<Distribution<MemorySpace>> A distribution with sampler & density as the same object, constructed from args
 */
template<typename MemorySpace, typename SamplerDensity, typename... T>
std::shared_ptr<Distribution<MemorySpace>> CreateDistribution(T... args) {
    std::shared_ptr<SamplerDensity> samplerDensity = std::make_shared<SamplerDensity>(args...);
    return std::make_shared<Distribution<MemorySpace>>(samplerDensity, samplerDensity);
};

} // namespace mpart

#endif //MPART_Distribution_H