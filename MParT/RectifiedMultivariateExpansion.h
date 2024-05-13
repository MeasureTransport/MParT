#ifndef MPART_RECTIFIEDMULTIVARIATEEXPANSION_H
#define MPART_RECTIFIEDMULTIVARIATEEXPANSION_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/Utilities/RootFinding.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/KokkosHelpers.h"

#include <algorithm>
#include <numeric>

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

        using Worker_T = MultivariateExpansionWorker<
            BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous,
                Kokkos::pair<OffdiagEval, DiagEval>, //OffDiag and DiagEval are the same
            Rectifier>, MemorySpace
        >;
        
        RectifiedMultivariateExpansion(Worker_T const& worker_):
                                    ConditionalMapBase<MemorySpace>(worker_.InputSize(), 1, worker_.NumCoeffs()),
                                    worker(worker_)
        {};


        ~RectifiedMultivariateExpansion() = default;


        void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                          StridedMatrix<double, MemorySpace>              output) override
        {
            // Take first dim-1 dimensions of pts and evaluate expansion_off
            // Add that to the evaluation of expansion_diag on pts
            StridedVector<double, MemorySpace> output_slice = Kokkos::subview(output, 0, Kokkos::ALL());
            StridedVector<const double, MemorySpace> coeff = this->savedCoeffs;

            const unsigned int numPts = pts.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = worker.CacheSize();

            // Define functor if there is a constant worker for all dimensions
            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);


                    // Fill in entries in the cache that are dependent on x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::None);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::None);
                    output_slice(ptInd) = worker.Evaluate(cache.data(), coeff);
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

            StridedVector<const double, MemorySpace> coeff = this->savedCoeffs;

            const unsigned int numPts = pts.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = worker.CacheSize();

            // Define functor if there is a constant worker for all dimensions
            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                    StridedVector<double, MemorySpace> grad = Kokkos::subview(output, Kokkos::ALL(), ptInd);

 
                    // Evaluate the expansion

                    // Fill in the entries in the cache
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::Input);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Input);

                    // Evaluate the expansion
                    worker.InputDerivative(cache.data(), coeff, grad);

                    for(unsigned int i=0; i<inDim; ++i) {
                        grad(i) *= sens_slice(ptInd);
                    }

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
            StridedVector<const double, MemorySpace> coeff = this->savedCoeffs;
            StridedVector<const double, MemorySpace> sens_slice = Kokkos::subview(sens, 0, Kokkos::ALL());

            const unsigned int numPts = pts.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = worker.CacheSize();
            unsigned int maxParams = coeff.size();

            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                    auto grad = Kokkos::subview(output, Kokkos::ALL(), ptInd);

                    // Fill in entries in the cache
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::Parameters);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Parameters);

                    // Evaluate the expansion
                    worker.CoeffDerivative(cache.data(), coeff, grad);

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
            StridedVector<const double, MemorySpace> coeff = this->savedCoeffs;
            unsigned int cacheSize = worker.CacheSize();

            // Take logdet of diagonal expansion
            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::None);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Diagonal);
                    // Evaluate the expansion
                    output(ptInd) = Kokkos::log(worker.DiagonalDerivative(cache.data(), coeff, 1));
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

            StridedVector<const double, MemorySpace> coeff = this->savedCoeffs;

            const unsigned int numPts = x1.extent(1);

            // Figure out how much memory we'll need in the cache
            unsigned int cacheSize = worker.CacheSize();

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
                   
                    // Note r = g(x) + f(x,y) --> y = f(x,.)^{-1}(r - g(x))
                    double yd = r_slice(ptInd);

                    // Fill in entries in the cache that are independent on x_d.
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::None);
                    SingleWorkerEvaluator<decltype(pt), decltype(coeff)> evaluator {cache.data(), pt, coeff, worker};
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
            StridedVector<const double, MemorySpace> coeff = this->savedCoeffs;
            unsigned int cacheSize = worker.CacheSize();

            // Take logdet of diagonal expansion
            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    auto out = Kokkos::subview(output, Kokkos::ALL(), ptInd);

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    // Fill in the mixed entries grad_{x,y}d_y
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::MixedInput);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::MixedInput);

                    // Evaluate the expansion for mixed derivative
                    worker.MixedInputDerivative(cache.data(), coeff, out);

                    // Find diagonal derivative d_y
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::Diagonal);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Diagonal);
                    double diag_deriv = worker.DiagonalDerivative(cache.data(), coeff, 1);

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

            
            StridedVector<const double, MemorySpace> coeff = this->savedCoeffs;
            unsigned int numPts = pts.extent(1);
            unsigned int cacheSize = worker.CacheSize();

            // Take logdet of diagonal expansion
            auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

                unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

                if(ptInd<numPts){

                    // Create a subview containing only the current point
                    auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                    auto out = Kokkos::subview(output, Kokkos::ALL(), ptInd);

                    // Get a pointer to the shared memory that Kokkos set up for this team
                    Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                    // Fill in cache with mixed entries grad_c d_y T(x,y; c)
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::MixedCoeff);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::MixedCoeff);

                    // Evaluate the Mixed coeff derivatives
                    worker.MixedCoeffDerivative(cache.data(), coeff, 1, out);

                    // Find diagonal derivative d_y
                    worker.FillCache1(cache.data(), pt, DerivativeFlags::Diagonal);
                    worker.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Diagonal);
                    double diag_deriv = worker.DiagonalDerivative(cache.data(), coeff, 1);
                    for(unsigned int ii=0; ii<out.size(); ++ii) out(ii) /= diag_deriv;
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);

            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);
            Kokkos::fence();
        }

        std::vector<unsigned int> DiagonalCoeffIndices() const
        {
            return worker.NonzeroDiagonalEntries();
        }

    private:
        template<typename PointType, typename CoeffType>
        struct SingleWorkerEvaluator {
            double* cache;
            PointType pt;
            CoeffType coeffs;
            Worker_T worker;

            KOKKOS_FUNCTION SingleWorkerEvaluator(double* cache_, PointType pt_, CoeffType coeffs_, Worker_T worker_):
                cache(cache_), pt(pt_), coeffs(coeffs_), worker(worker_) {}
            KOKKOS_INLINE_FUNCTION double operator()(double x) {
                worker.FillCache2(cache, pt, x, DerivativeFlags::None);
                return worker.Evaluate(cache, coeffs);
            }
        };


        Worker_T worker;
    }; // class RectifiedMultivariateExpansion
}


#endif