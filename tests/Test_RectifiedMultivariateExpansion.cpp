#include <catch2/catch_all.hpp>
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/Sigmoid.h"
#include "MParT/HermiteFunction.h"
#include "MParT/RectifiedMultivariateExpansion.h"
#include "MParT/MultivariateExpansion.h"

using namespace mpart;
using namespace Catch;
using namespace Catch::Matchers;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE("RectifiedMultivariateExpansion, Unrectified", "[RMVE_NoRect]") {
    unsigned int dim = 3;
    unsigned int maxOrder = 2;
    using T = ProbabilistHermite;
    // Create a rectified MVE equivalent to a simple Hermite expansion
    using OffdiagEval_T = BasisEvaluator<BasisHomogeneity::Homogeneous, T>;
    using DiagEval_T = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<T, T>, Identity>;
    using RectExpansion_T = RectifiedMultivariateExpansion<MemorySpace, T, T, Identity>;
    BasisEvaluator<BasisHomogeneity::Homogeneous, T> basis_eval_offdiag;
    BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<T, T>, Identity> basis_eval_diag{dim};
    FixedMultiIndexSet<MemorySpace> fmset_offdiag(dim-1, maxOrder);
    auto limiter = MultiIndexLimiter::NonzeroDiagTotalOrderLimiter(maxOrder);
    FixedMultiIndexSet<MemorySpace> fmset_diag = MultiIndexSet::CreateTotalOrder(dim, maxOrder, limiter).Fix(true);
    MultivariateExpansionWorker<OffdiagEval_T, MemorySpace> worker_off(fmset_offdiag, basis_eval_offdiag);
    MultivariateExpansionWorker<DiagEval_T, MemorySpace> worker_diag(fmset_diag, basis_eval_diag);
    RectExpansion_T rect_mve(worker_off, worker_diag);

    FixedMultiIndexSet<MemorySpace> fmset_mve {dim, maxOrder};
    MultivariateExpansion<OffdiagEval_T, MemorySpace> mve(1, fmset_mve, basis_eval_offdiag);
    SECTION("Initialization") {
        REQUIRE(rect_mve.numCoeffs == mve.numCoeffs);
        REQUIRE(rect_mve.inputDim == dim);
        REQUIRE(rect_mve.outputDim == 1);
    }
    Kokkos::View<double*, MemorySpace> coeffs("Input", mve.numCoeffs);
    Kokkos::deep_copy(coeffs, 0.1);

    mve.WrapCoeffs(coeffs);
    rect_mve.SetCoeffs(coeffs);

    unsigned int numPts = 20;
    Kokkos::View<double**, MemorySpace> points("Input", dim, numPts);
    for(int i = 0; i < numPts; i++) {
        for(int j = 0; j < dim; j++) {
            points(j, i) = 2*double(j-i)/(numPts-1);
        }
    }
    StridedMatrix<double, MemorySpace> eval_ref = mve.Evaluate(points);
    StridedMatrix<double, MemorySpace> eval_rect = rect_mve.Evaluate(points);
    SECTION("Evaluation") {
        REQUIRE(eval_rect.extent(0) == 1);
        REQUIRE(eval_rect.extent(1) == numPts);
        for(int i = 0; i < numPts; i++) {
            CHECK_THAT(eval_rect(0, i), WithinRel(eval_ref(0, i), 1e-14));
        }
    }
    Kokkos::View<double**, MemorySpace> sens ("Sensitivity", 1, numPts);
    for(int i = 0; i < numPts; i++) sens(0,i) = i % 3 ? 0.5 : -0.5;

    SECTION("Gradient") {
        StridedMatrix<double, MemorySpace> grad_ref = mve.Gradient(points, sens);
        StridedMatrix<double, MemorySpace> grad_rect = rect_mve.Gradient(points, sens);
        REQUIRE(grad_rect.extent(0) == dim);
        REQUIRE(grad_rect.extent(1) == numPts);
        for(int i = 0; i < numPts; i++) {
            for(int j = 0; j < dim; j++) {
                CHECK_THAT(grad_rect(j, i), WithinRel(grad_ref(j, i), 1e-14));
            }
        }
    }
    SECTION("CoeffGrad") {
        StridedMatrix<double, MemorySpace> grad_ref = mve.CoeffGrad(points, sens);
        StridedMatrix<double, MemorySpace> grad_rect = rect_mve.CoeffGrad(points, sens);
        REQUIRE(grad_rect.extent(0) == mve.numCoeffs);
        REQUIRE(grad_rect.extent(1) == numPts);
        std::vector<double> grad_ref_vec(mve.numCoeffs);
        std::vector<double> grad_rect_vec(mve.numCoeffs);
        for(int i = 0; i < numPts; i++) {
            for(int j = 0; j < mve.numCoeffs; j++) {
                grad_ref_vec[j] = grad_ref(j, i);
                grad_rect_vec[j] = grad_rect(j, i);
            }
            // sort the gradients to compare (since coeffs are in diff order)
            std::sort(grad_ref_vec.begin(), grad_ref_vec.end());
            std::sort(grad_rect_vec.begin(), grad_rect_vec.end());
            // Compare the sorted gradients
            for(int j = 0; j < mve.numCoeffs; j++) {
                CHECK_THAT(grad_rect_vec[j], WithinRel(grad_ref_vec[j], 1e-14));
            }
        }
    }
    double fd_step = 1e-6;
    std::vector<bool> idx_checked (numPts);
    StridedVector<double, MemorySpace> logdet_rect = rect_mve.LogDeterminant(points);
    SECTION("LogDeterminant") {
        Kokkos::View<double**, MemorySpace> pts_fd ("Perturbed Input", dim, numPts);
        for(int i = 0; i < numPts; i++) {
            for(int j = 0; j < dim; j++) {
                pts_fd(j, i) = points(j, i);
                if(j == dim-1) pts_fd(j, i) += fd_step;
            }
        }
        StridedMatrix<double, MemorySpace> eval_fd = mve.Evaluate(pts_fd);
        int numChecked = 0;
        for(int i = 0; i < numPts; i++) {
            double fd_deriv = (eval_fd(0, i) - eval_ref(0, i))/fd_step;
            if(fd_deriv > 0) { // Since unrectified, derivative may be negative.
                double logdet_ref_i = std::log(fd_deriv);
                CHECK_THAT(logdet_rect(i), WithinRel(logdet_ref_i, 20*fd_step));
                idx_checked[i] = true;
                numChecked++;
            }
        }
        REQUIRE(numChecked > 3); // Require at least 4 points to have a positive derivative
    }
    // Following checks are for self consistency
    SECTION("LogDeterminantInputGrad") {
        StridedMatrix<double, MemorySpace> logdet_rect_grad = rect_mve.LogDeterminantInputGrad(points);
        Kokkos::View<double*, MemorySpace> logdet_rect_fd ("Perturbed LogDet", numPts);

        for(int i = 0; i < numPts; i++) // Perturb each point in first dimension
            points(0,i) += fd_step;

        for(int j = 0; j < dim; j++) {
            rect_mve.LogDeterminantImpl(points, logdet_rect_fd);
            for(int i = 0; i < numPts; i++) {
                double logdet_rect_grad_fd = (logdet_rect_fd(i) - logdet_rect(i))/fd_step;
                if(idx_checked[i]) // Only check points with positive derivative
                    CHECK_THAT(logdet_rect_grad(j, i), WithinRel(logdet_rect_grad_fd, 20*fd_step));
                points(j,i) -= fd_step;
                if(j < dim-1) points(j+1,i) += fd_step;
            }
        }
    }
    SECTION("LogDeterminantCoeffGrad") {
        StridedMatrix<double, MemorySpace> logdet_rect_coeff_grad = rect_mve.LogDeterminantCoeffGrad(points);
        Kokkos::View<double*, MemorySpace> logdet_rect_fd ("Perturbed LogDet", numPts);
        rect_mve.Coeffs()(0) += fd_step;

        for(int j = 0; j < mve.numCoeffs; j++) {
            rect_mve.LogDeterminantImpl(points, logdet_rect_fd);
            for(int i = 0; i < numPts; i++) {
                double logdet_rect_coeff_grad_fd = (logdet_rect_fd(i) - logdet_rect(i))/fd_step;
                if(idx_checked[i]) // Only check points with positive derivative
                    CHECK_THAT(logdet_rect_coeff_grad(j, i), WithinRel(logdet_rect_coeff_grad_fd, 20*fd_step));
                rect_mve.Coeffs()(j) -= fd_step;
                if(j < mve.numCoeffs-1) rect_mve.Coeffs()(j+1) += fd_step;
            }
        }
    }
    // Cannot test Inverse since the unrectified expansion may not be invertible
}

TEMPLATE_TEST_CASE("Sigmoid RectifiedMultivariateExpansion","[sigmoid_rmve]", SigmoidTypeSpace::Logistic) {
    unsigned int maxOrder = 4;
    unsigned int dim = 3;
    FixedMultiIndexSet<MemorySpace> fmset_offdiag(dim-1, maxOrder);
    auto limiter = MultiIndexLimiter::NonzeroDiagTotalOrderLimiter(maxOrder);
    MultiIndexSet mset_diag = MultiIndexSet::CreateTotalOrder(dim, maxOrder, limiter);
    FixedMultiIndexSet<MemorySpace> fmset_diag = mset_diag.Fix(true);

    const int num_sigmoids = 3;
    const int order = num_sigmoids+1+2;
    const int param_length = 2 + num_sigmoids*(num_sigmoids+1)/2;
    Kokkos::View<double*, MemorySpace> center("Sigmoid Center", param_length);
    Kokkos::View<double*, MemorySpace> width("Sigmoid Width", param_length);
    Kokkos::View<double*, MemorySpace> weight("Sigmoid Weight", param_length);
    double edge_bound = 3.;
    center(0) = -edge_bound; width(0) = 2*edge_bound/10; weight(0) = 1.;
    center(1) =  edge_bound; width(1) = 2*edge_bound/10; weight(1) = 1.;
    int param_idx = 2;
    for(int curr_order = 1; curr_order <= num_sigmoids; curr_order++) {
        for(int i = 0; i < curr_order; i++) {
            center(param_idx) = 4*(-(curr_order-1)/2 + i);
            width(param_idx) = 1/((double)i+1);
            weight(param_idx) = 1./curr_order;
            param_idx++;
        }
    }
    using Sigmoid_T = Sigmoid1d<MemorySpace,TestType>;
    using OffdiagEval_T = BasisEvaluator<BasisHomogeneity::Homogeneous, HermiteFunction>;
    using DiagEval_T = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<HermiteFunction, Sigmoid_T>, SoftPlus>;
    using RectExpansion_T = RectifiedMultivariateExpansion<MemorySpace, HermiteFunction, Sigmoid_T, SoftPlus>;

    Sigmoid_T basis_diag (center, width, weight);
    HermiteFunction basis_offdiag;
    OffdiagEval_T basis_eval_offdiag {basis_offdiag};
    DiagEval_T basis_eval_diag {dim, basis_offdiag, basis_diag};
    MultivariateExpansionWorker<OffdiagEval_T, MemorySpace> worker_off(fmset_offdiag, basis_eval_offdiag);
    MultivariateExpansionWorker<DiagEval_T, MemorySpace> worker_diag(fmset_diag, basis_eval_diag);
    RectExpansion_T expansion(worker_off, worker_diag);

}