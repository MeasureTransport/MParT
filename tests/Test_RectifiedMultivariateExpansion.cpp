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
    // Cannot test Inverse since the unrectified expansion is generally not be invertible
}

using TestPair1 = std::pair<HermiteFunction, SigmoidTypeSpace::Logistic>;
TEMPLATE_TEST_CASE("Single Sigmoid RectifiedMultivariateExpansion","[single_sigmoid_rmve]", TestPair1) {
    using OffdiagBasis = typename TestType::first_type;
    using SigmoidShape = typename TestType::second_type;
    using Sigmoid_T = Sigmoid1d<MemorySpace,SigmoidShape>;
    using OffdiagEval_T = BasisEvaluator<BasisHomogeneity::Homogeneous,OffdiagBasis>;
    using DiagEval_T = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<OffdiagBasis, Sigmoid_T>, SoftPlus>;
    using RectExpansion_T = RectifiedMultivariateExpansion<MemorySpace, OffdiagBasis, Sigmoid_T, SoftPlus>;
    unsigned int dim = 2; // one offdiag dim and one diag dim
    // Setup sigmoid
    Kokkos::View<double*,MemorySpace> center("Sigmoid Center", 3);
    Kokkos::View<double*,MemorySpace> width("Sigmoid Width", 3);
    Kokkos::View<double*,MemorySpace> weight("Sigmoid Weight", 3);
    center(0) = -1; width(0) = 1.0; weight(0) = 1.; // Left Edge Term
    center(1) =  1; width(1) = 1.0; weight(1) = 1.; // Right Edge Term
    center(2) =  0; width(2) = 1.0; weight(2) = 1.; // Center Term
    Sigmoid_T basis_diag (center, width, weight);
    // Setup basis evals
    HermiteFunction basis_offdiag;
    OffdiagEval_T basis_eval_offdiag {basis_offdiag};
    DiagEval_T basis_eval_diag {dim, basis_offdiag, basis_diag};
    // Setup multi index sets
    unsigned int maxOrder = 4;
    FixedMultiIndexSet<MemorySpace> fmset_offdiag (dim-1, maxOrder);
    unsigned int sigmoid_order = 4; // const, linear, left ET, right ET, sigmoid
    auto limiter = MultiIndexLimiter::NonzeroDiagTotalOrderLimiter(sigmoid_order);
    MultiIndexSet mset_diag = MultiIndexSet::CreateTotalOrder(dim, sigmoid_order, limiter);
    FixedMultiIndexSet<MemorySpace> fmset_diag = mset_diag.Fix(true);
    // Setup expansion
    MultivariateExpansionWorker<OffdiagEval_T, MemorySpace> worker_off(fmset_offdiag, basis_eval_offdiag);
    MultivariateExpansionWorker<DiagEval_T, MemorySpace> worker_diag(fmset_diag, basis_eval_diag);
    RectExpansion_T expansion(worker_off, worker_diag);

    // Initialize Points and Coeffs
    Kokkos::View<double*, MemorySpace> coeffs("Input", expansion.numCoeffs);
    Kokkos::deep_copy(coeffs, 0.0); // Initialize to 0
    expansion.WrapCoeffs(coeffs);
    int numPts = 20;
    Kokkos::View<double**, MemorySpace> points("Input", dim, numPts+1);
    for(int i = 0; i < numPts; i++) {
        double h = double(numPts-1)/2;
        double grid = 3*double(i-h)/h;
        points(0, i) = -grid;
        points(1, i) =  grid;
    }
    points(0, numPts) = 0; points(1, numPts) = 1.; // Point to get constant of proportionality
    // Store the independent evaluations for each order of diag
    Kokkos::View<double**, MemorySpace> diag_eval("DiagEval", sigmoid_order, numPts);
    // First, check the diagonal shape against reference
    double prop_constant;
    double edge_term_inv_1 = std::log(std::exp(1)-1)+1; // Where edge terms should equal 1
    for(unsigned int order = 1; order <= sigmoid_order; order++) {
        // Find index of coefficient for multiindex {0, order}
        int idx = fmset_diag.MultiToIndex({0u, order});
        REQUIRE(idx >= 0);
        unsigned int coeff_idx = fmset_offdiag.Size() + idx;
        coeffs(coeff_idx) = 1.0; // Set the order term to 1
        StridedMatrix<double, MemorySpace> eval = expansion.Evaluate(points);
        // Due to how MVE_worker handles zeros, constant here can change between orders
        // Generally should be 1 for constant, and then fixed for other orders
        prop_constant = std::abs(eval(0, numPts)); // Normalize r(x)s(y) when r is constant
        for(int i = 0; i < numPts; i++) {
            double y = points(1, i);
            double ref;
            if(order == 1) ref = y;
            if(order == 2) ref = -SoftPlus::Evaluate(-(y+1));
            if(order == 3) ref =  SoftPlus::Evaluate( (y-1));
            if(order == 4) ref =  SigmoidShape::Evaluate(y);
            double rect = eval(0, i)/prop_constant;
            diag_eval(order-1, i) = rect;
            REQUIRE_THAT(rect, WithinRel(ref, 1e-14));
        }
        // Setup the calibration point depending on next order
        if(order == 1) points(1, numPts) = -edge_term_inv_1; // left side (LET)
        if(order == 2) points(1, numPts) =  edge_term_inv_1; // right side (RET)
        if(order == 3) points(1, numPts) =  100; // Far right side (Sigmoid)
        coeffs(coeff_idx) = 0.0; // Set the previous coeff to 0
    }

    // Now check that setting only offdiag terms gives correct result
    // Store the independent evaluations for each order of offdiag
    Kokkos::View<double**, MemorySpace> offdiag_eval ("OffdiagEval", maxOrder+1, numPts);
    for(unsigned int offdiag_check_order = 0; offdiag_check_order <= maxOrder; offdiag_check_order++) {
        int offdiag_idx = fmset_offdiag.MultiToIndex({offdiag_check_order});
        REQUIRE(offdiag_idx >= 0);
        coeffs(offdiag_idx) = 1.0; // Add the offdiag term back
        StridedMatrix<double, MemorySpace> eval = expansion.Evaluate(points);
        for(int i = 0; i < numPts; i++) {
            double x = points(0, i);
            double ref = basis_offdiag.Evaluate(offdiag_check_order, x);
            double rect = eval(0, i);
            CHECK_THAT(rect, WithinRel(ref, 1e-14));
            offdiag_eval(offdiag_check_order, i) = rect;
        }
        coeffs(offdiag_idx) = 0.0; // Set the offdiag term to 0
    }
    // Now check all the rectified diag terms together using stored evals
    SECTION("Rectified Evaluation") {
        for(unsigned int coeff_idx = fmset_offdiag.Size(); coeff_idx < expansion.numCoeffs; coeff_idx++) {
            coeffs(coeff_idx) = 1.0; // Set the corresponding midx term to 1
            std::vector<unsigned int> multi = fmset_diag.IndexToMulti(coeff_idx - fmset_offdiag.Size());
            unsigned int m0 = multi[0], m1 = multi[1];
            StridedMatrix<double, MemorySpace> eval = expansion.Evaluate(points);
            for(int i = 0; i < numPts; i++) {
                // Get cached reference values
                double ref_offdiag = offdiag_eval(m0, i);
                double ref_diag = diag_eval(m1-1, i);
                // Combine the two
                double ref = SoftPlus::Evaluate(ref_offdiag)*ref_diag;
                double rect = eval(0, i);
                CHECK_THAT(rect, WithinRel(ref, 1e-14));
            }
            coeffs(coeff_idx) = 0.0; // Set the previous coeff to 0
        }
    }
}

// Test the gradient and inverse methods
TEMPLATE_TEST_CASE("Multiple Sigmoid RectifiedMultivariateExpansion","[multi_sigmoid_rmve]", TestPair1) {
    using OffdiagBasis = typename TestType::first_type;
    using SigmoidShape = typename TestType::second_type;
    using Sigmoid_T = Sigmoid1d<MemorySpace,SigmoidShape>;
    using OffdiagEval_T = BasisEvaluator<BasisHomogeneity::Homogeneous,OffdiagBasis>;
    using DiagEval_T = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<OffdiagBasis, Sigmoid_T>, SoftPlus>;
    using RectExpansion_T = RectifiedMultivariateExpansion<MemorySpace, OffdiagBasis, Sigmoid_T, SoftPlus>;
    unsigned int maxOrder = 4;
    unsigned int dim = 3;
    FixedMultiIndexSet<MemorySpace> fmset_offdiag(dim-1, maxOrder);
    auto limiter = MultiIndexLimiter::NonzeroDiagTotalOrderLimiter(maxOrder);
    MultiIndexSet mset_diag = MultiIndexSet::CreateTotalOrder(dim, maxOrder, limiter);
    FixedMultiIndexSet<MemorySpace> fmset_diag = mset_diag.Fix(true);

    // Set up sigmoids
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
    // Set up evaluators, workers, and expansion
    Sigmoid_T basis_diag (center, width, weight);
    OffdiagBasis basis_offdiag;
    OffdiagEval_T basis_eval_offdiag {basis_offdiag};
    DiagEval_T basis_eval_diag {dim, basis_offdiag, basis_diag};
    MultivariateExpansionWorker<OffdiagEval_T, MemorySpace> worker_off(fmset_offdiag, basis_eval_offdiag);
    MultivariateExpansionWorker<DiagEval_T, MemorySpace> worker_diag(fmset_diag, basis_eval_diag);
    RectExpansion_T expansion(worker_off, worker_diag);
    // Setup coeffs
    Kokkos::View<double*, MemorySpace> coeffs("Input", expansion.numCoeffs);
    for(int c = 0; c < expansion.numCoeffs; c++) coeffs(c) = 1.-double(c)/expansion.numCoeffs;
    expansion.WrapCoeffs(coeffs);
    // Setup points
    unsigned int numPts = 20;
    Kokkos::View<double**, MemorySpace> points("Input", dim, numPts);
    for(int i = 0; i < numPts; i++) {
        for(int j = 0; j < dim; j++) {
            points(j, i) = double(j+i-j*i)/(numPts-1);
        }
    }
    // Setup sensitivities
    Kokkos::View<double**, MemorySpace> sens ("Sensitivity", 1, numPts);
    for(int i = 0; i < numPts; i++) {
        sens(0, i) = i % 3 ? 0.5 : -0.5;
    }
    // Setup eval
    StridedMatrix<double, MemorySpace> eval = expansion.Evaluate(points);
    double fd_step = 1e-6;
    SECTION("Gradient") {
        StridedMatrix<double, MemorySpace> grad_rect = expansion.Gradient(points, sens);
        Kokkos::View<double**, MemorySpace> eval_fd ("EvaluateImpl storage", 1, numPts);
        REQUIRE(grad_rect.extent(0) == dim);
        REQUIRE(grad_rect.extent(1) == numPts);
        for(int i = 0; i < numPts; i++) {
            points(0, i) += fd_step;
        }
        for(int j = 0; j < dim; j++) {
            expansion.EvaluateImpl(points, eval_fd);
            for(int i = 0; i < numPts; i++) {
                double fd_deriv = (eval_fd(0, i) - eval(0, i))/fd_step;
                REQUIRE_THAT(grad_rect(j, i), WithinRel(fd_deriv*sens(0,i), 20*fd_step));
                points(j, i) -= fd_step;
                if(j < dim-1) points(j+1, i) += fd_step;
            }
        }
    }
    SECTION("CoeffGrad") {
        StridedMatrix<double, MemorySpace> grad_rect = expansion.CoeffGrad(points, sens);
        Kokkos::View<double**, MemorySpace> eval_fd ("EvaluateImpl storage", 1, numPts);
        REQUIRE(grad_rect.extent(0) == expansion.numCoeffs);
        REQUIRE(grad_rect.extent(1) == numPts);
        coeffs(0) += fd_step;
        for(int j = 0; j < expansion.numCoeffs; j++) {
            expansion.EvaluateImpl(points, eval_fd);
            for(int i = 0; i < numPts; i++) {
                double fd_deriv = (eval_fd(0, i) - eval(0, i))/fd_step;
                if(abs(fd_deriv) > fd_step)
                    CHECK_THAT(grad_rect(j, i), WithinRel(fd_deriv*sens(0,i), 20*fd_step));
                else
                    CHECK_THAT(grad_rect(j, i), WithinAbs(fd_deriv*sens(0,i), 1e-9));
            }
            coeffs(j) -= fd_step;
            if(j < expansion.numCoeffs-1) coeffs(j+1) += fd_step;
        }
    }
    StridedVector<double, MemorySpace> logdet_rect = expansion.LogDeterminant(points);
    SECTION("LogDeterminant") {
        Kokkos::View<double**, MemorySpace> pts_fd ("Perturbed Input", dim, numPts);
        for(int i = 0; i < numPts; i++) {
            for(int j = 0; j < dim; j++) {
                pts_fd(j, i) = points(j, i);
                if(j == dim-1) pts_fd(j, i) += fd_step;
            }
        }
        StridedMatrix<double, MemorySpace> eval_fd = expansion.Evaluate(pts_fd);
        for(int i = 0; i < numPts; i++) {
            double fd_deriv = (eval_fd(0, i) - eval(0, i))/fd_step;
            double logdet_ref_i = std::log(fd_deriv);
            CHECK_THAT(logdet_rect(i), WithinRel(logdet_ref_i, 20*fd_step));
        }
    }
    SECTION("LogDeterminantInputGrad") {
        StridedMatrix<double, MemorySpace> logdet_rect_grad = expansion.LogDeterminantInputGrad(points);
        Kokkos::View<double*, MemorySpace> logdet_rect_fd ("Perturbed LogDet", numPts);
        for(int i = 0; i < numPts; i++) // Perturb each point in first dimension
            points(0, i) += fd_step;
        for(int j = 0; j < dim; j++) {
            expansion.LogDeterminantImpl(points, logdet_rect_fd);
            for(int i = 0; i < numPts; i++) {
                double logdet_rect_grad_fd = (logdet_rect_fd(i) - logdet_rect(i))/fd_step;
                CHECK_THAT(logdet_rect_grad(j, i), WithinRel(logdet_rect_grad_fd, 20*fd_step));
                points(j, i) -= fd_step;
                if(j < dim-1) points(j+1, i) += fd_step;
            }
        }
    }
    SECTION("LogDeterminantCoeffGrad") {
        StridedMatrix<double, MemorySpace> logdet_rect_coeff_grad = expansion.LogDeterminantCoeffGrad(points);
        Kokkos::View<double*, MemorySpace> logdet_rect_fd ("Perturbed LogDet", numPts);
        coeffs(0) += fd_step;
        for(int j = 0; j < expansion.numCoeffs; j++) {
            expansion.LogDeterminantImpl(points, logdet_rect_fd);
            for(int i = 0; i < numPts; i++) {
                double logdet_rect_coeff_grad_fd = (logdet_rect_fd(i) - logdet_rect(i))/fd_step;
                CHECK_THAT(logdet_rect_coeff_grad(j, i), WithinRel(logdet_rect_coeff_grad_fd, 20*fd_step));
            }
            coeffs(j) -= fd_step;
            if(j < expansion.numCoeffs-1) coeffs(j+1) += fd_step;
        }
    }
    SECTION("Inverse") {
        auto subpts_1 = Kokkos::subview(points, std::pair<int,int>(0,dim-1), Kokkos::ALL());
        auto subpts_ref = Kokkos::subview(points, std::pair<int,int>(dim-1,dim), Kokkos::ALL());
        StridedMatrix<double, MemorySpace> inv_eval = expansion.Inverse(subpts_1, eval);
        double tol = 1e-6; // From the xtol,ytol in RMVE
        for(int i = 0; i < numPts; i++) {
            CHECK_THAT(inv_eval(0, i), WithinAbs(subpts_ref(0, i), tol));
        }
    }
}