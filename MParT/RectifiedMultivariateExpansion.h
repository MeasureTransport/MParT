#ifndef MPART_RECTIFIEDMULTIVARIATEEXPANSION_H
#define MPART_RECTIFIEDMULTIVARIATEEXPANSION_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/Utilities/RootFinding.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/KokkosHelpers.h"

#include <algorithm>

namespace mpart{

    /**
     @brief Defines a multivariate expansion based on the tensor product of 1d basis functions.

     @details

     @tparam BasisEvaluatorType The type of the 1d basis functions
     @tparam MemorySpace The Kokkos memory space where the coefficients and evaluations are stored.
     */
    template<typename MemorySpace, class OffdiagEval, class DiagEval, class Rectifier>
    class RectifiedMultivariateExpansion : public ConditionalMapBase<MemorySpace>
    {
    public:
        using ExecutionSpace = typename MemoryToExecution<MemorySpace>::Space;
        using DiagWorker_T = MultivariateExpansionWorker<
            BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous,
                Kokkos::pair<OffdiagEval, DiagEval>,
            Rectifier>, MemorySpace
        >;
        using OffdiagWorker_T = MultivariateExpansionWorker<
            BasisEvaluator<BasisHomogeneity::Homogeneous, OffdiagEval>
        >;


        RectifiedMultivariateExpansion(OffdiagWorker_T const& worker_off_,
                                       DiagWorker_T const& worker_diag_):
                                    ConditionalMapBase<MemorySpace>(worker_diag_.InputSize(), 1, worker_off_.NumCoeffs() + worker_diag_.NumCoeffs()),
                                    setSize_off(worker_off_.NumCoeffs()),
                                    setSize_diag(worker_diag_.NumCoeffs()),
                                    worker_off(worker_off_),
                                    worker_diag(worker_diag_)
        {
            // TODO: Check that the inputs are compatible
            // MVE_diag has no terms constant in last input
            // Then, ensure all of these have the appropriate coefficient views
        };

        ~RectifiedMultivariateExpansion() = default;


        void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                          StridedMatrix<double, MemorySpace>              output) override
        {
            // Take first dim-1 dimensions of pts and evaluate expansion_off
            // Add that to the evaluation of expansion_diag on pts
            StridedVector<double, MemorySpace> output_slice = Kokkos::subview(output, 0, Kokkos::ALL());
            StridedVector<const double, MemorySpace> coeff_off = CoeffOff();
            StridedVector<const double, MemorySpace> coeff_diag = CoeffDiag();

            const unsigned int numPts = pts.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = std::max(worker_diag.CacheSize(), worker_off.CacheSize());

            // Define functor if there is a constant worker for all dimensions
            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    auto pt_off = Kokkos::subview(pt, std::pair<int,int>(0,pt.size()-1));

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                    worker_off.FillCache1(cache.data(), pt_off, DerivativeFlags::None);
                    worker_off.FillCache2(cache.data(), pt_off, pt_off(pt_off.size()-1), DerivativeFlags::None);

                    // Evaluate the expansion
                    output_slice(ptInd) = worker_off.Evaluate(cache.data(), coeff_off);

                    // Fill in entries in the cache that are dependent on x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::None);
                    worker_diag.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::None);
                    output_slice(ptInd) += worker_diag.Evaluate(cache.data(), coeff_diag);
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);

            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);

            Kokkos::fence();
        }

        void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                          StridedMatrix<const double, MemorySpace> const& sens,
                          StridedMatrix<double, MemorySpace>              output) override
        {
            // Take first dim-1 dimensions of pts and take gradient of expansion_off
            // Add that to the gradient of expansion_diag on pts
            StridedVector<const double, MemorySpace> sens_slice = Kokkos::subview(sens, 0, Kokkos::ALL());
            unsigned int inDim = pts.extent(0);

            StridedVector<const double, MemorySpace> coeff_off = CoeffOff();
            StridedVector<const double, MemorySpace> coeff_diag = CoeffDiag();

            const unsigned int numPts = pts.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = std::max(worker_diag.CacheSize(), worker_off.CacheSize());

            // Define functor if there is a constant worker for all dimensions
            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    auto pt_off = Kokkos::subview(pts, std::pair<int,int>(0,pt.size()-1), ptInd);

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                    Kokkos::View<double*,MemorySpace> grad_off(team_member.thread_scratch(1), inDim-1);
                    StridedVector<double, MemorySpace> grad_out = Kokkos::subview(output, Kokkos::ALL(), ptInd);

                    // Fill in entries in the cache that are independent of x_d
                    worker_off.FillCache1(cache.data(), pt_off, DerivativeFlags::Input);
                    worker_off.FillCache2(cache.data(), pt_off, pt_off(pt_off.size()-1), DerivativeFlags::Input);

                    // Evaluate the expansion
                    worker_off.InputDerivative(cache.data(), coeff_off, grad_off);

                    // Fill in the entries in the cache dependent on x_d
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::Input);
                    worker_diag.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Input);

                    // Evaluate the expansion
                    worker_diag.InputDerivative(cache.data(), coeff_diag, grad_out);

                    for(unsigned int i=0; i<inDim-1; ++i) {
                        grad_out(i) = sens_slice(ptInd) * (grad_off(i) + grad_out(i));
                    }
                    grad_out(inDim - 1) = sens_slice(ptInd) * grad_out(inDim - 1);
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize+inDim-1);

            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);

            Kokkos::fence();
        }


        void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                           StridedMatrix<const double, MemorySpace> const& sens,
                           StridedMatrix<double, MemorySpace>              output) override
        {
            StridedVector<const double, MemorySpace> coeff_off = CoeffOff();
            StridedVector<const double, MemorySpace> coeff_diag = CoeffDiag();
            StridedVector<const double, MemorySpace> sens_slice = Kokkos::subview(sens, 0, Kokkos::ALL());

            const unsigned int numPts = pts.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = std::max(worker_diag.CacheSize(), worker_off.CacheSize());
            unsigned int maxParams = coeff_off.size() + coeff_diag.size();
            Kokkos::pair<unsigned int, unsigned int> coeff_off_idx {0u,coeff_off.size()};
            Kokkos::pair<unsigned int, unsigned int> coeff_diag_idx {coeff_off.size(), maxParams};

            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    auto pt_off = Kokkos::subview(pt, std::pair<int,int>(0,pt.extent(0)-1));

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                    auto grad_off = Kokkos::subview(output, coeff_off_idx, ptInd);
                    auto grad_diag = Kokkos::subview(output, coeff_diag_idx, ptInd);

                    // Fill in entries in the cache that are independent of x_d.
                    worker_off.FillCache1(cache.data(), pt_off, DerivativeFlags::Parameters);
                    worker_off.FillCache2(cache.data(), pt_off, pt_off(pt_off.size()-1), DerivativeFlags::Parameters);

                    // Evaluate the expansion
                    worker_off.CoeffDerivative(cache.data(), coeff_off, grad_off);

                    // Fill in entries in the cache that are dependent on x_d.
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::Parameters);
                    worker_diag.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Parameters);

                    // Evaluate the expansion
                    worker_diag.CoeffDerivative(cache.data(), coeff_diag, grad_diag);

                    // TODO: Move this into own kernel?
                    for(unsigned int i=0; i<maxParams; ++i)
                        output(i, ptInd) *= sens_slice(ptInd);
                }
            };


            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);
            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);
            Kokkos::fence();
        }

        void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                StridedVector<double, MemorySpace>              output) override
        {
            // Take logdet of diagonal expansion
            unsigned int numPts = pts.extent(1);
            StridedVector<const double, MemorySpace> coeff_diag = CoeffDiag();
            unsigned int cacheSize = worker_diag.CacheSize();

            // Take logdet of diagonal expansion
            auto functor = KOKKOS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::None);
                    worker_diag.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Diagonal);
                    // Evaluate the expansion
                    output(ptInd) = Kokkos::log(worker_diag.DiagonalDerivative(cache.data(), coeff_diag, 1));
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);

            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);

            Kokkos::fence();
        }

        void InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                         StridedMatrix<const double, MemorySpace> const& r,
                         StridedMatrix<double, MemorySpace>              output) override
        {
            // We know x1 should be the same as the input to expansion_off
            // Since we are working with r = g(x) + f(x,y) --> y = f(x,.)^{-1}(r - g(x))
            StridedVector<double, MemorySpace> out_slice = Kokkos::subview(output, 0, Kokkos::ALL());
            StridedVector<const double, MemorySpace> r_slice = Kokkos::subview(r, 0, Kokkos::ALL());

            StridedVector<const double, MemorySpace> coeff_off = CoeffOff();
            StridedVector<const double, MemorySpace> coeff_diag = CoeffDiag();

            const unsigned int numPts = x1.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = std::max(worker_diag.CacheSize(), worker_off.CacheSize());

            // Options for root finding
            const double xtol = 1e-6, ytol = 1e-6;

            // Define functor if there is a constant worker for all dimensions
            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();
                int info; // TODO: Handle info?
                if(ptInd<numPts){
                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(x1, Kokkos::ALL(), ptInd);

                    // Check for NaNs
                    for(unsigned int ii=0; ii<pt.size(); ++ii){
                        if(std::isnan(pt(ii))){
                            out_slice(ptInd) = std::numeric_limits<double>::quiet_NaN();
                            return;
                        }
                    }

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    // Fill in entries in the cache that are independent of x_d.
                    worker_off.FillCache1(cache.data(), pt, DerivativeFlags::None);
                    worker_off.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::None);

                    // Note r = g(x) + f(x,y) --> y = f(x,.)^{-1}(r - g(x))
                    double yd = r_slice(ptInd) - worker_off.Evaluate(cache.data(), coeff_off);

                    // Fill in entries in the cache that are independent on x_d.
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::None);
                    SingleWorkerEvaluator<decltype(pt), decltype(coeff_diag)> evaluator {cache.data(), pt, coeff_diag, worker_diag};
                    out_slice(ptInd) = RootFinding::InverseSingleBracket<MemorySpace>(yd, evaluator, pt(pt.size()-1), xtol, ytol, info);
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);

            // Paralel loop over each point computing T^{-1}(x,.)(r) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);

            Kokkos::fence();
        }

        void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                         StridedMatrix<double, MemorySpace>              output) override
        {

            unsigned int numPts = pts.extent(1);
            StridedVector<const double, MemorySpace> coeff_diag = CoeffDiag();
            unsigned int cacheSize = worker_diag.CacheSize();

            // Take logdet of diagonal expansion
            auto functor = KOKKOS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    auto out = Kokkos::subview(output, Kokkos::ALL(), ptInd);

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    // Fill in the mixed entries grad_{x,y}d_y
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::MixedInput);
                    worker_diag.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::MixedInput);

                    // Evaluate the expansion for mixed derivative
                    worker_diag.MixedInputDerivative(cache.data(), coeff_diag, out);

                    // Find diagonal derivative d_y
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::Diagonal);
                    worker_diag.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Diagonal);
                    double diag_deriv = worker_diag.DiagonalDerivative(cache.data(), coeff_diag, 1);

                    // grad_{x,y} log(d_y T(x,y)) = [grad_x d_y T(x,y), d_y^2 T(x,y)] / d_y T(x,y)
                    for(unsigned int ii=0; ii<out.size(); ++ii) out(ii) /= diag_deriv;
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);

            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);
            Kokkos::fence();
        }

        void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                         StridedMatrix<double, MemorySpace>              output) override
        {
            // Take logdetcoeffgrad of diagonal expansion, output to bottom block
            StridedMatrix<double, MemorySpace> output_off = Kokkos::subview(output,
                std::make_pair(0u,worker_off.NumCoeffs()), Kokkos::ALL());
            Kokkos::deep_copy(output_off, 0.0);
            StridedMatrix<double, MemorySpace> output_diag = Kokkos::subview(output,
                std::make_pair(worker_off.NumCoeffs(),worker_off.NumCoeffs()+worker_diag.NumCoeffs()),
                Kokkos::ALL());
            StridedVector<const double, MemorySpace> coeff_diag = CoeffDiag();
            unsigned int numPts = pts.extent(1);
            unsigned int cacheSize = worker_diag.CacheSize();

            // Take logdet of diagonal expansion
            auto functor = KOKKOS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    auto out = Kokkos::subview(output_diag, Kokkos::ALL(), ptInd);

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    // Fill in cache with mixed entries grad_c d_y T(x,y; c)
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::MixedCoeff);
                    worker_diag.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::MixedCoeff);

                    // Evaluate the Mixed coeff derivatives
                    worker_diag.MixedCoeffDerivative(cache.data(), coeff_diag, 1, out);

                    // Find diagonal derivative d_y
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::Diagonal);
                    worker_diag.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Diagonal);
                    double diag_deriv = worker_diag.DiagonalDerivative(cache.data(), coeff_diag, 1);
                    for(unsigned int ii=0; ii<out.size(); ++ii) out(ii) /= diag_deriv;
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);

            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);
            Kokkos::fence();
        }

    private:
        template<typename PointType, typename CoeffType>
        struct SingleWorkerEvaluator {
            double* cache;
            PointType pt;
            CoeffType coeffs;
            DiagWorker_T worker;

            SingleWorkerEvaluator(double* cache_, PointType pt_, CoeffType coeffs_, DiagWorker_T worker_):
                cache(cache_), pt(pt_), coeffs(coeffs_), worker(worker_) {}
            double operator()(double x) {
                worker.FillCache2(cache, pt, x, DerivativeFlags::None);
                return worker.Evaluate(cache, coeffs);
            }
        };

        OffdiagWorker_T worker_off;
        DiagWorker_T worker_diag;
        const unsigned int setSize_off;
        const unsigned int setSize_diag;
        StridedVector<const double, MemorySpace> CoeffOff() const { return Kokkos::subview(this->savedCoeffs, std::make_pair(0u, setSize_off)); }
        StridedVector<const double, MemorySpace> CoeffDiag() const { return Kokkos::subview(this->savedCoeffs, std::make_pair(setSize_off, setSize_off+setSize_diag)); }
    }; // class RectifiedMultivariateExpansion
}


#endif