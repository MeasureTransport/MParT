#ifndef MPART_COMPOSEDMAP_H
#define MPART_COMPOSEDMAP_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/Miscellaneous.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>

#include <deque>

namespace mpart{

/**
 @brief Provides a definition of a transport map defined by the composition of several maps.
 @details
This class defines a map \f$T:\mathbb{R}^N\rightarrow \mathbb{R}^N\f$ with the composed structure
\f[
T(x) = T_L(x) \circ \ \dots \ \circ T_1(x),
\f]
where each map \f$T_i:\mathbb{R}^{N}\rightarrow \mathbb{R}^{N}\f$ is a square transport map.

Checkpointing can be used for deep maps to reduce the amount of memory that is required for gradient computations.
The checkpointing logic was adapted from the `revolve` algorithm of <a href="https://dl.acm.org/doi/10.1145/347837.347846">[Griewank and Walther, 2000]</a>,
which is an optimal binomial checkpointing scheme.

 */
template<typename MemorySpace>
class ComposedMap : public ConditionalMapBase<MemorySpace>
{
public:

    /** @brief Construct a block triangular map from a collection of other ConditionalMapBase objects.

    @param maps A vector of ConditionalMapBase objects defining each \f$T_k\f$ in the composition. Note: each map must be square (i.e., have equal input and output dimension).
    @param moveCoeffs Whether you want this ComposedMap object to be responsible for the coefficients already in maps. If true, the new object will take ownership of all coefficient vectors within maps. If false, the new object will not have any coefficients set.
    @param maxChecks The maximum number of checkpoints to use during gradient computations.  If maxChecks==1, then no checkpointing will be utilized and all forward states will be recomputed.  If maxChecks==components.size(), then all states will be stored and reused during the backward pass.  This is the most efficient option, but can require an intractable amount of memory for high-dimensional or deep parameterizations.   The default value is -1, which will set the maximum number of checkpoints to be equal to the number of layers (i.e., map.size()).
    */
    ComposedMap(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> const& maps, bool moveCoeffs=false, int maxChecks=-1);

    virtual ~ComposedMap() = default;

    /** @brief Sets the coefficients for all components of the map.

        @details This function will copy the provided coeffs vectors into the savedCoeffs object in the ComposedMap class.   To avoid
        duplicating the coefficients, the savedCoeffs member variable for each component will then be set to a subview of this vector.
        @param coeffs A vector containing coefficients for all components.  If component \f$k\f$ is defined by \f$C_k\f$ coefficients,
                  then this vector should have length \f$\sum_{k=1}^K C_i\f$ and the coefficients for component \f$k\f$ should
                  start at index \f$\sum_{j=1}^{k-1} C_j\f$.
    */
    using ConditionalMapBase<MemorySpace>::SetCoeffs;
    using ConditionalMapBase<MemorySpace>::WrapCoeffs;
    void SetCoeffs(Kokkos::View<const double*, Kokkos::HostSpace> coeffs) override;
    void WrapCoeffs(Kokkos::View<double*, MemorySpace> coeffs) override;

    #if defined(MPART_ENABLE_GPU)
    void SetCoeffs(Kokkos::View<const double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs) override;
    #endif

    virtual std::shared_ptr<ConditionalMapBase<MemorySpace>> GetComponent(unsigned int i){ return maps_.at(i);}

    /** @brief Computes the log determinant of the Jacobian matrix of this map.

    @details
    @param pts The points where the jacobian should be evaluated.  Should have \f$N\f$ rows.  Each column contains a single point.
    @param output A vector with length equal to the number of columns in pts containing the log determinant of the Jacobian.  This
                  vector should be correctly allocated and sized before calling this function.
    */
    void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                            StridedVector<double, MemorySpace>              output) override;


    /** @brief Evaluates the map.

    @details
    @param pts The points where the map should be evaluated.  Should have \f$N\f$ rows.  Each column contains a single point.
    @param output A matrix with \f$M\f$ rows to store the map output.  Should have the same number of columns as pts and be
                  allocated before calling this function.
    */
    void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<double, MemorySpace>              output) override;


    void InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                     StridedMatrix<const double, MemorySpace> const& r,
                     StridedMatrix<double, MemorySpace>              output) override;


    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                       StridedMatrix<const double, MemorySpace> const& sens,
                       StridedMatrix<double, MemorySpace>              output) override;


    void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                     StridedMatrix<double, MemorySpace>              output) override;


    void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                     StridedMatrix<double, MemorySpace>              output) override;


    void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<const double, MemorySpace> const& sens,
                      StridedMatrix<double, MemorySpace>              output) override;
private:

    unsigned int maxChecks_;
    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> maps_;

    /* Class for coordinating checkpoints during gradient evaluations. */
    class Checkpointer {
    public:

        Checkpointer(unsigned int maxSaves,
                     StridedMatrix<const double, MemorySpace> initialPts,
                     std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>>& comps);

        /** Returns the input to a layer in the composed map.

        Stores checkpoints along the way to assist in backwards passes to compute gradients.
        @param[in] layerInd The index of the layer we want the input to.
        @return A kokkos view containing the input to layer layerInd.
        */
        Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> GetLayerInput(unsigned int layerInd);

    protected:

        /** Given the current state of the checkpoints and a need to evaluate the input to layer layerInd, this
            function computes the next layer input that should be checkpointed.
            @param[in] layerInd The index of the layer that we eventually want to get the input for.
            @return An integer specifying the next layer index that should be checkpointed in a forward pass.
        */
        int GetNextCheckpoint(unsigned int layerInd) const;

        const unsigned int maxSaves_;

        Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> workspace1_;
        Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> workspace2_;
        std::deque<Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>> checkpoints_;
        std::deque<unsigned int> checkpointLayers_;

        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>>& maps_;
    };

}; // class ComposedMap

}

#endif
