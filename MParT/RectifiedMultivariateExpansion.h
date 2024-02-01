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
    template<class BasisEvaluatorType, class RectifiedBasisEvaluatorType, typename MemorySpace>
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
            expansion_diag.LogDeterminant(pts, output);
        }

        void InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                         StridedMatrix<const double, MemorySpace> const& r,  
                         StridedMatrix<double, MemorySpace>              output) override
        {
            // We know x1 should be the same as the input to expansion_off
            // Since we are working with r = g(x) + f(x,y) --> y = f(x,.)^{-1}(r - g(x))
            expansion_off.EvaluateImpl(x1, output);
            Kokkos::parallel_for(output.extent(1), KOKKOS_LAMBDA(int i){
                output(0, i) = r(0, i) - output(0, i);
            });
            // Need inverse implementation for expansion_diag
            expansion_diag.InverseImpl(x1, output, output);
        }

        void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                         StridedMatrix<double, MemorySpace>              output) override
        {
            // Take logdetinputgrad of diagonal expansion
            expansion_diag.LogDeterminantInputGrad(pts, output);
        }

        void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                         StridedMatrix<double, MemorySpace>              output) override
        {
            // Take logdetcoeffgrad of diagonal expansion, output to bottom block
            StridedMatrix<double, MemorySpace> output_diag = Kokkos::subview(output, std::make_pair(expansion_off.numCoeffs,expansion_off.numCoeffs+expansion_diag.numCoeffs), Kokkos::ALL());
            expansion_diag.LogDeterminantCoeffGrad(pts, output_diag);
        }

    private:
        MultivariateExpansion<BasisEvaluatorType, MemorySpace> expansion_off;
        MultivariateExpansion<RectifiedBasisEvaluatorType, MemorySpace> expansion_diag;

    }; // class RectifiedMultivariateExpansion
}


#endif 