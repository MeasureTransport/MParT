#ifndef MPART_UNIVARIATEEXPANSION_H
#define MPART_UNIVARIATEEXPANSION_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/BasisEvaluator.h"
#include "MParT/Utilities/KokkosHelpers.h"
#include "MParT/Utilities/RootFinding.h"
#include <iostream>


namespace mpart {

/**
 * @brief A class to represent a univariate expansion \f$f(x) = \sum_{k=0}^p c_kp_k(x)\f$
 *
 * Note that, while this extends ConditionalMapBase, this is only a true
 * ConditionalMapBase if \f$f\f$ is strictly monotone.
 *
 * @tparam MemorySpace
 * @tparam BasisType
 */
template<typename MemorySpace, typename BasisType>
class UnivariateExpansion: public ConditionalMapBase<MemorySpace>{
    using ExecutionSpace = typename MemorySpace::execution_space;
    public:
    UnivariateExpansion(unsigned int maxOrder, BasisType basisType = BasisType()):
        ConditionalMapBase<MemorySpace>(1, 1, maxOrder+1), maxOrder_(maxOrder), basis_(basisType)
    {};

    void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& points,
        StridedMatrix<double, MemorySpace> out) override {
        auto point_slice = Kokkos::subview(points, 0, Kokkos::ALL());
        auto out_slice = Kokkos::subview(out, 0, Kokkos::ALL());
        StridedVector<const double, MemorySpace> coeffs = this->savedCoeffs;
        unsigned int numPts = points.extent(1);
        unsigned int cacheSize = maxOrder_ + 1;

        auto functor = KOKKOS_CLASS_LAMBDA(typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){

                // Get a pointer to the shared memory that Kokkos set up for this team
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                basis_.EvaluateAll(cache.data(), maxOrder_, point_slice(ptInd));
                double result = 0.0;
                for(int i = 0; i <= maxOrder_; i++) {
                    result += cache(i) * coeffs(i);
                }
                out_slice(ptInd) = result;
            }
        };
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);
        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
        Kokkos::fence();
    }

    void GradientImpl(StridedMatrix<const double, MemorySpace> const& points,
        StridedMatrix<const double, MemorySpace> const& sens,
        StridedMatrix<double, MemorySpace> out) override {

        auto point_slice = Kokkos::subview(points, 0, Kokkos::ALL());
        auto sens_slice = Kokkos::subview(sens, 0, Kokkos::ALL());
        auto out_slice = Kokkos::subview(out, 0, Kokkos::ALL());
        StridedVector<const double, MemorySpace> coeffs = this->savedCoeffs;
        unsigned int numPts = points.extent(1);
        unsigned int cacheSize = maxOrder_ + 1;

        auto functor = KOKKOS_CLASS_LAMBDA(typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){

                // Get a pointer to the shared memory that Kokkos set up for this team
                Kokkos::View<double*,MemorySpace> cache_eval(team_member.thread_scratch(1), cacheSize);
                Kokkos::View<double*,MemorySpace> cache_grad(team_member.thread_scratch(1), cacheSize);
                basis_.EvaluateDerivatives(cache_eval.data(), cache_grad.data(), maxOrder_, point_slice(ptInd));
                double result = 0.0;
                for(int i = 0; i <= maxOrder_; i++) {
                    result += cache_grad(i) * coeffs(i);
                }
                out_slice(ptInd) = result * sens_slice(ptInd);
            }
        };
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(2*cacheSize);
        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
        Kokkos::fence();
    }

    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& points,
        StridedMatrix<const double, MemorySpace> const& sens,
        StridedMatrix<double, MemorySpace> out) override {
        auto point_slice = Kokkos::subview(points, 0, Kokkos::ALL());
        auto sens_slice = Kokkos::subview(sens, 0, Kokkos::ALL());
        unsigned int numPts = points.extent(1);
        unsigned int cacheSize = maxOrder_ + 1;

        auto functor = KOKKOS_CLASS_LAMBDA(typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){
                // Get a pointer to the shared memory that Kokkos set up for this team
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                basis_.EvaluateAll(cache.data(), maxOrder_, point_slice(ptInd));
                for(int i = 0; i <= maxOrder_; i++)
                    out(i, ptInd) = cache(i) * sens_slice(ptInd);
            }
        };
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);
        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
        Kokkos::fence();
    }

    void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& points,
        StridedVector<double, MemorySpace> out) override {
        auto point_slice = Kokkos::subview(points, 0, Kokkos::ALL());
        StridedVector<const double, MemorySpace> coeffs = this->savedCoeffs;
        unsigned int numPts = points.extent(1);
        unsigned int cacheSize = maxOrder_ + 1;

        auto functor = KOKKOS_CLASS_LAMBDA(typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){

                // Get a pointer to the shared memory that Kokkos set up for this team
                Kokkos::View<double*,MemorySpace> cache_eval(team_member.thread_scratch(1), cacheSize);
                Kokkos::View<double*,MemorySpace> cache_grad(team_member.thread_scratch(1), cacheSize);
                basis_.EvaluateDerivatives(cache_eval.data(), cache_grad.data(), maxOrder_, point_slice(ptInd));
                double result = 0.0;
                for(int i = 0; i <= maxOrder_; i++) {
                    result += cache_grad(i) * coeffs(i);
                }
                out(ptInd) = log(result);
            }
        };
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(2*cacheSize);
        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
        Kokkos::fence();
    }

    void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& points,
        StridedMatrix<double, MemorySpace> out) override {
        auto point_slice = Kokkos::subview(points, 0, Kokkos::ALL());
        auto out_slice = Kokkos::subview(out, 0, Kokkos::ALL());
        StridedVector<const double, MemorySpace> coeffs = this->savedCoeffs;
        unsigned int numPts = points.extent(1);
        unsigned int numCoeffs = this->numCoeffs;
        unsigned int cacheSize = numCoeffs*3;

        auto functor = KOKKOS_CLASS_LAMBDA(typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){

                // Get a pointer to the shared memory that Kokkos set up for this team
                Kokkos::View<double*,MemorySpace> cache_eval(team_member.thread_scratch(1), numCoeffs);
                Kokkos::View<double*,MemorySpace> cache_grad(team_member.thread_scratch(1), numCoeffs);
                Kokkos::View<double*,MemorySpace> cache_hess(team_member.thread_scratch(1), numCoeffs);
                basis_.EvaluateSecondDerivatives(cache_eval.data(), cache_grad.data(), cache_hess.data(), maxOrder_, point_slice(ptInd));
                double df = 0.0;
                double d2f = 0.0;
                for(int i = 0; i <= maxOrder_; i++) {
                    df += cache_grad(i) * coeffs(i);
                    d2f += cache_hess(i) * coeffs(i);
                }
                out_slice(ptInd) = d2f/df;
            }
        };
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);
        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
        Kokkos::fence();
    }

    void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& points,
        StridedMatrix<double, MemorySpace> out) override {
        auto point_slice = Kokkos::subview(points, 0, Kokkos::ALL());
        StridedVector<const double, MemorySpace> coeffs = this->savedCoeffs;
        unsigned int numPts = points.extent(1);
        unsigned int numCoeffs = this->numCoeffs;
        unsigned int cacheSize = 2*numCoeffs;

        auto functor = KOKKOS_CLASS_LAMBDA(typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){
                auto out_slice = Kokkos::subview(out, Kokkos::ALL(), ptInd);

                // Get a pointer to the shared memory that Kokkos set up for this team
                Kokkos::View<double*,MemorySpace> cache_eval(team_member.thread_scratch(1), numCoeffs);
                Kokkos::View<double*,MemorySpace> cache_grad(team_member.thread_scratch(1), numCoeffs);
                basis_.EvaluateDerivatives(cache_eval.data(), cache_grad.data(), maxOrder_, point_slice(ptInd));
                double df = 0.0;
                for(int i = 0; i <= maxOrder_; i++) {
                    df += cache_grad(i) * coeffs(i);
                    out_slice(i) = cache_grad(i);
                }
                for(int i = 0; i <= maxOrder_; i++) {
                    out_slice(i) /= df;
                }
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
        StridedMatrix<double, MemorySpace> out) override {

        StridedVector<double, MemorySpace> out_slice = Kokkos::subview(out, 0, Kokkos::ALL());
        StridedVector<const double, MemorySpace> r_slice = Kokkos::subview(r, 0, Kokkos::ALL());

        StridedVector<const double, MemorySpace> coeff = this->savedCoeffs;
        const unsigned int numPts = r.extent(1);

        // Figure out how much memory we'll need in the cache
        unsigned int cacheSize = maxOrder_ + 1;

        // Options for root finding
        const double xtol = 1e-6, ytol = 1e-6;
        double xd_guess = 0.; // Initial guess for x_d

        // Define functor if there is a constant worker for all dimensions
        auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();
            int info; // TODO: Handle info?
            if(ptInd<numPts){
                double yd = r_slice(ptInd);
                // Check for NaNs
                if(std::isnan(yd)){
                    out_slice(ptInd) = std::numeric_limits<double>::quiet_NaN();
                    return;
                }
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                SingleUnivariateEvaluator<decltype(coeff)> evaluator {cache.data(), coeff, basis_};
                out_slice(ptInd) = RootFinding::InverseSingleBracket<MemorySpace>(yd, evaluator, xd_guess, xtol, ytol, info);
            }
        };

        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);

        // Paralel loop over each point computing T^{-1}(x,.)(r) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);

        Kokkos::fence();
    }

    std::vector<unsigned int> DiagonalCoeffIndices() const override {
        std::vector<unsigned int> diagIndices(maxOrder_+1);
        std::iota(diagIndices.begin(), diagIndices.end(), 0);
        return diagIndices;
    }

    private:

    template<typename CoeffType>
    class SingleUnivariateEvaluator {
        public:
        SingleUnivariateEvaluator(double* cache, CoeffType coeff, BasisType basis):
            cache_(cache), coeff_(coeff), basisEval_(basis), maxOrderEval_(coeff.extent(0)-1)
        {};

        double operator()(double x) const {
            basisEval_.EvaluateAll(cache_, maxOrderEval_, x);
            double result = 0.0;
            for(int i = 0; i <= maxOrderEval_; i++) {
                result += cache_[i] * coeff_(i);
            }
            return result;
        }

        private:
        double* cache_;
        const unsigned int maxOrderEval_;
        const CoeffType coeff_;
        const BasisType basisEval_;
    };

    const unsigned int maxOrder_;
    const BasisType basis_;
};

}

#endif