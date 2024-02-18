#ifndef MPART_PUSHFORWARDDENSITY_H
#define MPART_PUSHFORWARDDENSITY_H

#include "MParT/Distributions/DensityBase.h"
#include "MParT/Utilities/LinearAlgebra.h"
#include "MParT/ConditionalMapBase.h"

namespace mpart {

/**
 * @brief A class to represent a density of a Pullback distribution given a map and a density.
 *        If \f$X\sim\mu\f$, then \f$T\f$ satisfies \f$T(X)\sim\nu\f$ and this class represents the density of the distribution
 *        \f$\nu\f$ given the density of \f$\mu\f$ and the map \f$T\f$.
 *
 * @tparam MemorySpace Where data is stored for computation
 */
template<typename MemorySpace>
class PushforwardDensity: public DensityBase<MemorySpace> {

    public:
    PushforwardDensity() = delete;
    /**
     * @brief Construct a new Pushforward Density object given the transport map and the density \f$\mu\f$
     *
     * @param map transport map \f$T:\mu\to\nu\f$
     * @param density density of the distribution \f$\mu\f$
     */
    PushforwardDensity(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<DensityBase<MemorySpace>> density):
        DensityBase<MemorySpace>(density->Dim()), map_(map), density_(density) {
        if (map_->inputDim != density_->Dim()) {
            throw std::invalid_argument("PushforwardDensity: map output dimension does not match density dimension");
        }
        if (map_->outputDim != map_->inputDim) {
            throw std::invalid_argument("PushforwardDensity: map input dimension does not match map output dimension");
        }
    }

    /**
     * @brief Given points distributed according to \f$\nu\f$, calculate the log PDF of those points using the pushforward
     *
     * @param pts (MxN) data matrix where each column is identically distributed according to \f$\nu\f$
     * @param output N-length vector to store the log density evaluation of \f$T(X)\f$
     */
    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) override {
        Kokkos::View<double**, MemorySpace> prefix_null("prefix_null", 0, pts.extent(1));
        StridedMatrix<double, MemorySpace> mappedPts = map_->Inverse(prefix_null, pts);
        density_->LogDensityImpl(mappedPts, output);
        StridedVector<double, MemorySpace> logJacobian = map_->LogDeterminant(mappedPts);
	Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy{0, output.extent(0)};
        Kokkos::parallel_for("Subtract logJac", policy, KOKKOS_LAMBDA(const unsigned int i){
            output(i) -= logJacobian(i);
        });
    };

    // CANNOT EVALUATE GRADIENT OF T WRT INPUTS OR COEFFS
    void LogDensityInputGradImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) override {
        throw std::runtime_error("LogDensityInputGrad not implemented for PushforwardDensity");
    };

    private:
    /**
     * @brief Transport map \f$T:\mu\to\nu\f$
     *
     */
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    /**
     * @brief Density of measure \f$\mu\f$
     *
     */
    std::shared_ptr<DensityBase<MemorySpace>> density_;
};

} // namespace mpart

#endif //MPART_PUSHFORWARDDENSITY_H
