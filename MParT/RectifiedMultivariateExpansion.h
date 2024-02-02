#ifndef MPART_RECTIFIEDMULTIVARIATEEXPANSION_H
#define MPART_RECTIFIEDMULTIVARIATEEXPANSION_H

#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/MultivariateExpansionWorker.h"
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
    template<typename MemorySpace, class BasisEvaluatorType, class RectifiedBasisEvaluatorType>
    class RectifiedMultivariateExpansion : public ConditionalMapBase<MemorySpace>
    {
    public:

        RectifiedMultivariateExpansion(MultivariateExpansion<BasisEvaluatorType, MemorySpace> const& expansion_off,
                                       MultivariateExpansion<RectifiedBasisEvaluatorType, MemorySpace> const& expansion_diag):
                                    ConditionalMapBase<MemorySpace>(expansion_off.inputDim, 1, expansion_off.numCoeffs + expansion_diag.numCoeffs),
                                    worker(mset, basis1d)
        {
            // TODO: Check that the inputs are compatible
            // Both MVE have output dim 1
            // MVE_off has the same input dim as MVE_diag - 1
            // MVE_diag has no terms constant in last input
            // Then, ensure all of these have the appropriate coefficient views
        };

        ~RectifiedMultivariateExpansion() = default;


        void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                          StridedMatrix<double, MemorySpace>              output) override
        {
            // Take first dim-1 dimensions of pts and evaluate expansion_off
            // Add that to the evaluation of expansion_diag on pts
            Kokkos::View<double**, MemorySpace> output_off("output_tmp", 1, pts.extent(1));
            StridedMatrix<double, MemorySpace> pts_off = Kokkos::subview(pts, std::make_pair(0:pts.extent(0)-1), Kokkos::ALL());
            expansion_off.Evaluate(pts_off, output_off);
            expansion_diag.Evaluate(pts, output);
            Kokkos::fence();
            Kokkos::parallel_for("RectifiedMultivariateExpansion::EvaluateImpl", pts.extent(1), KOKKOS_LAMBDA(int i){
                output(0, i) += output_off(0, i);
            });
        }

        void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                          StridedMatrix<const double, MemorySpace> const& sens,
                          StridedMatrix<double, MemorySpace>              output) override
        {
            // Take first dim-1 dimensions of pts and take gradient of expansion_off
            // Add that to the gradient of expansion_diag on pts
            Kokkos::View<double**, MemorySpace> output_off("output_tmp", pts.extent(0)-1, pts.extent(1));
            StridedMatrix<double, MemorySpace> pts_off = Kokkos::subview(pts, std::make_pair(0:pts.extent(0)-1), Kokkos::ALL());
            expansion_off.Gradient(pts_off, sens, output_off);
            expansion_diag.Gradient(pts, sens, output);
            Kokkos::fence();
            Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {output.extent(0)-1, output.extent(1)});
            Kokkos::parallel_for("RectifiedMultivariateExpansion::GradientImpl", policy, KOKKOS_LAMBDA(const int i, const int j){
                output(j, i) += output_off(j, i);
            });
        }


        void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                           StridedMatrix<const double, MemorySpace> const& sens,
                           StridedMatrix<double, MemorySpace>              output) override
        {
            // Take gradient of each expansion individually and concatenate
            StridedMatrix<double, MemorySpace> output_off = Kokkos::subview(output, std::make_pair(0,expansion_off.numCoeffs), Kokkos::ALL());
            StridedMatrix<double, MemorySpace> output_diag = Kokkos::subview(output, std::make_pair(expansion_off.numCoeffs,expansion_off.numCoeffs+expansion_diag.numCoeffs), Kokkos::ALL());
            StridedMatrix<double, MemorySpace> pts_off = Kokkos::subview(pts, std::make_pair(0:pts.extent(0)-1), Kokkos::ALL());
            expansion_off.CoeffGrad(pts_off, sens, output_off);
            expansion_diag.CoeffGrad(pts, sens, output_diag);
        }

        void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                StridedVector<double, MemorySpace>              output) override
        {
            // Take logdet of diagonal expansion
            unsigned int numPts = pts.extent(1);
            StridedVector<double, MemorySpace> coeff_diag = Kokkos::subview(this->savedCoeffs, std::make_pair(expansion_off.numCoeffs,expansion_off.numCoeffs+expansion_diag.numCoeffs));
            MultivariateExpansionWorker<RectifiedBasisEvaluatorType, MemorySpace> worker_diag = expansion_diag.worker;
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
            expansion_off.EvaluateImpl(x1, output);
            StridedVector<double, MemorySpace> out_slice = Kokkos::subview(output, 0, Kokkos::ALL());
            StridedVector<const double, MemorySpace> r_slice = Kokkos::subview(r, 0, Kokkos::ALL());
            Kokkos::parallel_for(output.extent(1), KOKKOS_LAMBDA(int i){
                out_slice(i) = r_slice(i) - out_slice(0, i);
            });
            // Need inverse implementation for expansion_diag
            // TODO
        }

        void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                         StridedMatrix<double, MemorySpace>              output) override
        {
            
            unsigned int numPts = pts.extent(1);
            StridedVector<double, MemorySpace> coeff_diag = Kokkos::subview(this->savedCoeffs,
                std::make_pair(expansion_off.numCoeffs,expansion_off.numCoeffs+expansion_diag.numCoeffs)
            );
            MultivariateExpansionWorker<RectifiedBasisEvaluatorType, MemorySpace> worker_diag = expansion_diag.worker;
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
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::MixedInput);
                    worker_diag.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::MixedInput);
                    
                    // Evaluate the expansion
                    output(ptInd) = worker_diag.MixedInputDerivative(cache.data(), coeff_diag, 1);
                    
                    worker_diag.FillCache1(cache.data(), pt, DerivativeFlags::Diagonal);
                    worker_diag.FillCache2(cache.data(), pt, pt(pt.size()-1), DerivativeFlags::Diagonal);
                    output(ptInd) /= worker_diag.DiagonalDerivative(cache.data(), coeff_diag, 1);
                }
            };

            auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);
            
            // Paralel loop over each point computing T(x_1,...,x_D) for that point
            auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
            Kokkos::parallel_for(policy, functor);
        }

        void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                         StridedMatrix<double, MemorySpace>              output) override
        {
            // Take logdetcoeffgrad of diagonal expansion, output to bottom block
            StridedMatrix<double, MemorySpace> output_diag = Kokkos::subview(output,
                std::make_pair(expansion_off.numCoeffs,expansion_off.numCoeffs+expansion_diag.numCoeffs),
                Kokkos::ALL());
            // TODO
        }

    private:

        void MixedInputDerivative(StridedMatrix<const double, MemorySpace> const& pts,  
                                  StridedMatrix<double, MemorySpace>              output) override
        {
        }
        MultivariateExpansion<BasisEvaluatorType, MemorySpace> expansion_off;
        MultivariateExpansion<RectifiedBasisEvaluatorType, MemorySpace> expansion_diag;

    }; // class RectifiedMultivariateExpansion
}


#endif 