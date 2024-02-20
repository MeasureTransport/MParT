#ifndef MPART_MONOTONECOMPONENT_H
#define MPART_MONOTONECOMPONENT_H

#if defined(MPART_HAS_CEREAL)
#include <cereal/types/polymorphic.hpp>
#include "MParT/Utilities/Serialization.h"
#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/binary.hpp>
#endif // MPART_HAS_CEREAL

#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexSet.h"

#include "MParT/ConditionalMapBase.h"
#include "MParT/DerivativeFlags.h"
#include "MParT/MonotoneIntegrand.h"
#include "MParT/MultivariateExpansion.h"

#include "MParT/Utilities/Miscellaneous.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/KokkosHelpers.h"
#include "MParT/Utilities/RootFinding.h"


#include <Eigen/Core>

#include <Kokkos_Core.hpp>

namespace mpart{

/**
@brief Defines a function \f$T:R^N\rightarrow R\f$ such that \f$\partial T / \partial x_N >0\f$ is strictly positive.

@details
The function \f$T\f$ is based on another (generally non-monotone) function \f$f : R^N\rightarrow R\f$, a strictly positve function
\f$g : R\rightarrow R_{>0}\f$, and a non-negative scalar \f$\epsilon\f$.   Together, these functions define the monotone component $T$ through

\f[
T(x_1, x_2, ..., x_D) = f(x_1,x_2,..., x_{D-1}, 0) + \int_0^{x_D}  g\left( \partial_D f(x_1,x_2,..., x_{D-1}, t) + \epsilon \right) dt
\f]

@tparam ExpansionType A class defining the function \f$f\f$.  It must satisfy the cached parameterization concept.
@tparam PosFuncType A class defining the function \f$g\f$.  This class must have `Evaluate` and `Derivative` functions accepting a double and returning a double.  The MParT::SoftPlus and MParT::Exp classes in PositiveBijectors.h are examples of classes defining this interface.
@tparam QuadratureType A class defining the integration scheme used to approximate \f$\int_0^{x_N}  g\left( \frac{\partial f}{\partial x_d}(x_1,x_2,..., x_{N-1}, t) \right) dt\f$.  The type must have a function `Integrate(f,lb,ub)` that accepts a functor `f`, a double lower bound `lb`, a double upper bound `ub`, and returns a double with an estimate of the integral.   The MParT::AdaptiveSimpson and MParT::RecursiveQuadrature classes provide this interface.
*/
template<class ExpansionType, class PosFuncType, class QuadratureType, typename MemorySpace>
class MonotoneComponent : public ConditionalMapBase<MemorySpace>
{

public:

    /** @brief Construct a monotone component with a specific expansion and quadrature type.
        @details
        @param expansion The expansion used to define the function \f$f\f$.
        @param quad The quadrature rule used to approximate \f$\int_0^{x_D}  g\left( \partial_D f(x_1,x_2,..., x_{D-1}, t) \right) dt\f$
        @param useContDeriv A flag to specify whether the analytic derivative of \f$T(x_1, x_2, ..., x_D)\f$ should be used by default, or if the derivative of the discretized integral should be used.  If "true", the analytic or "continuous" derivative will be used.  If "false", the derivative of the numerically approximated integral will be used.
        @param nugget The value of the non negative quantity \f$\epsilon\f$ defining the minimum slope of the transport map.
        @verbatim embed:rst
          See the :ref:`diag_deriv_section` mathematical background for more details.
        @endverbatim
    */
    MonotoneComponent(ExpansionType  const& expansion,
                      QuadratureType const& quad,
                      bool useContDeriv=true,
                      double nugget=0.0) : ConditionalMapBase<MemorySpace>(expansion.InputSize(), 1, expansion.NumCoeffs()),
                                                    expansion_(expansion),
                                                    quad_(quad),
                                                    dim_(expansion.InputSize()),
                                                    useContDeriv_(useContDeriv),
                                                    nugget_(nugget){};

    

    MonotoneComponent(ExpansionType  const& expansion,
                      QuadratureType const& quad,
                      bool useContDeriv,
                      double nugget,
                      Kokkos::View<const double*, MemorySpace> coeffsIn) : ConditionalMapBase<MemorySpace>(expansion.InputSize(), 1, expansion.NumCoeffs(), coeffsIn),
                                                    expansion_(expansion),
                                                    quad_(quad),
                                                    dim_(expansion.InputSize()),
                                                    useContDeriv_(useContDeriv),
                                                    nugget_(nugget){};

    virtual std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> GetBaseFunction() override{return std::make_shared<MultivariateExpansion<typename ExpansionType::BasisType, typename ExpansionType::KokkosSpace>>(1,expansion_);};

    /** Override the ConditionalMapBase Evaluate function. */
    void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<double, MemorySpace>              output) override
    {
        StridedVector<double,MemorySpace> outputSlice = Kokkos::subview(output, 0, Kokkos::ALL());
        EvaluateImpl(pts, this->savedCoeffs, outputSlice);
    }

    void InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                     StridedMatrix<const double, MemorySpace> const& r,
                     StridedMatrix<double, MemorySpace>              output) override
    {
        auto rSlice = Kokkos::subview(r,0,Kokkos::ALL());
        auto outputSlice = Kokkos::subview(output, 0, Kokkos::ALL());
        InverseImpl(x1, rSlice, this->savedCoeffs, outputSlice);
    }

    void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                            StridedVector<double,MemorySpace>               output) override
    {
        // First, get the diagonal derivative
        if(useContDeriv_){
            ContinuousDerivative(pts, this->savedCoeffs, output);
        }else{
            Kokkos::View<double*,MemorySpace> evals("Evaluations", pts.extent(1));
            DiscreteDerivative(pts, this->savedCoeffs, evals, output);
        }

        // Now take the log
        auto policy = Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space>(0,output.extent(0));
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA (unsigned int i) {
            if(output(i)<=0){
                output(i) = -std::numeric_limits<double>::infinity();
            }else{
                output(i) = std::log(output(i));
            }
        });
    }

    bool isGradFunctionInputValid(int sensRows, int sensCols, int ptsRows, int ptsCols, int outputRows, int outputCols, int expectedOutputRows) {
        bool isSensRowsValid = sensRows==this->outputDim;
        bool isInputColsValid = sensCols==ptsCols;
        bool isPtsValid = ptsRows==this->inputDim;
        bool isOutputRowsValid = outputRows==expectedOutputRows;
        bool isOutputColsValid = outputCols==ptsCols;
        return isSensRowsValid && isInputColsValid && isPtsValid && isOutputRowsValid && isOutputColsValid;
    }

    void checkGradFunctionInput(const std::string method, int sensRows, int sensCols, int ptsRows, int ptsCols, int outputRows, int outputCols, int expectedOutputRows) {
        bool isInputValid = isGradFunctionInputValid(sensRows, sensCols, ptsRows, ptsCols, outputRows, outputCols, expectedOutputRows);
        if(!isInputValid) {
            std::stringstream ss;
            ss << method << ": Invalid dimensions of input args." <<
            "sens: (" << sensRows << "," << sensCols << "), expected: " << this->outputDim << ", " << ptsCols << "), "
            << "pts: (" << ptsRows << "," << ptsCols << "), expected: (" << this->inputDim << "," << ptsCols << "), "
            << "output: (" << outputRows << "," << outputCols << "), expected: (" << expectedOutputRows << "," << ptsCols << ")";
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }
    }

    void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<const double, MemorySpace> const& sens,
                      StridedMatrix<double, MemorySpace>              output) override
    {
        checkGradFunctionInput("Gradient", sens.extent(0), sens.extent(1), pts.extent(0), pts.extent(1), output.extent(0), output.extent(1), this->inputDim);

        Kokkos::View<double*,MemorySpace> evals("Map output", pts.extent(1));

        InputJacobian(pts, this->savedCoeffs, evals, output);

        // Scale each column by the sensitivity
        auto policy = Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space>(0,pts.extent(1));
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA (unsigned int ptInd) {
            for(unsigned int i=0; i<this->inputDim; ++i)
                output(i,ptInd) *= sens(0,ptInd);
        });
    }

    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                       StridedMatrix<const double, MemorySpace> const& sens,
                       StridedMatrix<double, MemorySpace>              output) override
    {
        checkGradFunctionInput("CoeffGradImpl", sens.extent(0), sens.extent(1), pts.extent(0), pts.extent(1), output.extent(0), output.extent(1), this->numCoeffs);

        Kokkos::View<double*,MemorySpace> evals("Map output", pts.extent(1));

        CoeffJacobian(pts, this->savedCoeffs, evals, output);

        // Scale each column by the sensitivity
        auto policy = Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space>(0,pts.extent(1));
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA (unsigned int ptInd) {
            for(unsigned int i=0; i<this->numCoeffs; ++i)
                output(i,ptInd) *= sens(0,ptInd);
        });
    }


    void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                     StridedMatrix<double, MemorySpace>              output) override
    {
        Kokkos::View<double*,MemorySpace> derivs("Diagonal Derivative", pts.extent(1));

        // First, get the diagonal derivative
        if(useContDeriv_){
            ContinuousMixedJacobian(pts,this->savedCoeffs, output);
            ContinuousDerivative(pts, this->savedCoeffs, derivs);
        }else{
            Kokkos::View<double*,MemorySpace> evals("Evaluations", pts.extent(1));
            DiscreteMixedJacobian(pts,this->savedCoeffs, output);
            DiscreteDerivative(pts, this->savedCoeffs, evals, derivs);
        }

        // Now take the log
        auto policy = Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space>(0,pts.extent(1));
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA (unsigned int ptInd) {
            for(unsigned int i=0; i<this->numCoeffs; ++i)
                output(i,ptInd) *= 1.0/derivs(ptInd);
        });
    }

    void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                     StridedMatrix<double, MemorySpace>              output) override
    {
        Kokkos::View<double*,MemorySpace> derivs("Diagonal Derivative", pts.extent(1));

        // First, get the diagonal derivative
        if(useContDeriv_){
            ContinuousMixedInputJacobian(pts,this->savedCoeffs, output);
            ContinuousDerivative(pts, this->savedCoeffs, derivs);
        }else{
            // Kokkos::View<double*,MemorySpace> evals("Evaluations", pts.extent(1));
            // DiscreteMixedJacobian(pts,this->savedCoeffs, output);
            // DiscreteDerivative(pts, this->savedCoeffs, evals, derivs);
            std::stringstream msg;
            msg << "Discrete derivative version is not implemented yet (To Do)";
            throw std::invalid_argument(msg.str());
        }

        // Now take the log
        auto policy = Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space>(0,pts.extent(1));
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA (unsigned int ptInd) {
            for(unsigned int i=0; i<this->dim_; ++i)
                output(i,ptInd) *= 1.0/derivs(ptInd);
        });
    }

    /**
     * @brief Support calling EvaluateImpl with non-const views.
     */
    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space, class... OtherTraits>
    void EvaluateImpl(Kokkos::View<double**, OtherTraits...>  const& pts,
                      StridedVector<double,MemorySpace> const& coeffs,
                      StridedVector<double,MemorySpace>              output)
    {
        Kokkos::View<const double**, OtherTraits...> constPts = pts;
        StridedVector<const double, MemorySpace> constCoeffs = coeffs;

        EvaluateImpl<ExecutionSpace,OtherTraits...>(constPts,constCoeffs,output);
    }

    /**
       @brief Evaluates the monotone function \f$T(x_1,\ldots,x_D)\f$ at multiple points.

     * @param[in] pts A \f$D\times N\f$ array containing the \f$N\f$ points in \f$\mathbb{R}^D\f$ where we want to evaluate the monotone component.  Each column is a point.
     * @param[in] coeffs The coefficients in the expansion defining \f$f\f$.  The length of this array must be the same as the number of terms in the multiindex set passed to the constructor.
     * @param[out] output Kokkos::View<double*> An array containing the evaluattions \f$T(x^{(i)}_1,\ldots,x^{(i)}_D)\f$ for each \f$i\in\{0,\ldots,N\}\f$.
     */
    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space, class... OtherTraits>
    void EvaluateImpl(Kokkos::View<const double**, OtherTraits...>   const& pts,
                      StridedVector<const double,MemorySpace>        const& coeffs,
                      StridedVector<double,MemorySpace>                     output)
    {
        const unsigned int numPts = pts.extent(1);
        if(output.extent(0)!=numPts) {
            std::stringstream ss;
            ss << "EvaluateImpl: output has incorrect number of columns. "
            << "Expected: " << pts.extent(1) << ", got " << output.extent(0);
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }

        // Ask the expansion how much memory it would like for its one-point cache
        const unsigned int cacheSize = expansion_.CacheSize();
        quad_.SetDim(1);
        const unsigned int workspaceSize = quad_.WorkspaceSize();

        auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){
                // Create a subview containing only the current point
                auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

                // Get a pointer to the shared memory that Kokkos set up for this team
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                Kokkos::View<double*,MemorySpace> workspace(team_member.thread_scratch(1), workspaceSize);

                // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
                expansion_.FillCache1(cache.data(), pt, DerivativeFlags::None);
                output(ptInd) = EvaluateSingle(cache.data(), workspace.data(), pt, pt(dim_-1), coeffs, quad_, expansion_);
            }
        };


        // Create a policy with enough scratch memory to cache the polynomial evaluations
        unsigned int cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize+workspaceSize);
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes,functor);

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, functor);

    }


    // template< typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    // void InverseImpl(StridedMatrix<double, MemorySpace> const& xs,
    //                  StridedVector<double, MemorySpace> const& ys,
    //                  StridedVector<double, MemorySpace> const& coeffs,
    //                  StridedVector<double, MemorySpace>        output,
    //                  std::map<std::string, std::string>        options=std::map<std::string,std::string>())
    // {
    //     StridedMatrix<const double, MemorySpace> constXs = xs;
    //     StridedVector<const double, MemorySpace> constYs = ys;
    //     StridedVector<const double, MemorySpace> constCoeffs = coeffs;

    //     InverseImpl<ExecutionSpace>(constXs,constYs,constCoeffs,output,options);
    // }

    /**
     @brief Evaluates the inverse of the diagonal of the monotone component.
     @details This function solve the nonlinear equation \f$y_D = T(x_1,\ldots,x_D)\f$ for \f$x_D\f$ given \f$N\f$ different
              values of \f$y_D^{(1)},\ldots,y_D^{(N)}\f$.  It is possible to specify either a single \f$x_{1:D-1}\f$ sample or \f$N\f$ different values
              \f$x_{1:D-1}^{(1)},\ldots,x_{1:D-1}^{(N)}\f$.  If only a single point is given, then this function returns a vector
              containing \f$x_D^{(i)}=T^{-1}(x_1,\ldots,x_{D-1}, y^{(i)}_D)\f$.    When \f$N\f$N points are given, this function returns
              a vector containing \f$x_D^{(i)}=T^{-1}(x_1^{(i)},\ldots,x_{D-1}^{(i)}, y^{(i)}_D)\f$.

     @param xs A \f$D\times N_1\f$ array containing \f$N_1\f$ \f$x_{1:D-1}\f$ points.  Note that this matrix must have either 1 or N columns and at least \f$D\f$ rows.  The \f$D\f$ row will serve as an initial guess for \f$x_D\f$ during the inversion.
     @param ys A length \f$N\f$ array containing \f$N\f$ values of \f$y_D\f$ for use in the solve.
     @param coeffs The coefficients in the expansion defining \f$f\f$.  The length of this array must be the same as the number of terms in the multiindex set passed to the constructor.
     @param output An array for storing the computed values of \f$y_D^{(i)}\f$.  Memory for this array must be preallocated before calling this function.
     @param options A map containing options for the method (e.g., converge criteria, step sizes).   Available options are "Method" (must be "Bracket"), "xtol" (any nonnegative float), and "ytol" (any nonnegative float).
     */
    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    void InverseImpl(StridedMatrix<const double, MemorySpace> const& xs,
                     StridedVector<const double, MemorySpace> const& ys,
                     StridedVector<const double, MemorySpace> const& coeffs,
                     StridedVector<double, MemorySpace>              output,
                 std::map<std::string, std::string>                  options=std::map<std::string,std::string>())
    {
        // Extract the method from the options map
        std::string method;
        if(options.count("Method")){
            method = options["Method"];
        }else{
            method = "Bracket";
        }

        // Check to make sure the method is valid
        if(method!="Bracket"){
            std::stringstream msg;
            msg << "Invalid method given to MonotoneComponent::Inverse.  Given \"" << method << "\", but valid options are [\"Bisect\"].";
            throw std::invalid_argument(msg.str());
        }

        // Extract the xtol and ytol option if they exist
        double xtol, ytol;
        if(options.count("xtol")){
            xtol = std::stod(options["xtol"]);
            if(xtol<0){
                std::stringstream msg;
                msg << "Invalid tolerance \"xtol\" given to MonotoneComponent::Inverse.  Value must be non-negative, but given " << xtol;
                throw std::invalid_argument(msg.str());
            }
        }else{
            xtol = 1e-6;
        }
        if(options.count("ytol")){
            ytol = std::stod(options["ytol"]);
            if(ytol<0){
                std::stringstream msg;
                msg << "Invalid tolerance \"ytol\" given to MonotoneComponent::Inverse.  Value must be non-negative, but given " << ytol;
                throw std::invalid_argument(msg.str());
            }
        }else{
            ytol = 1e-6;
        }

        if((ytol<=std::numeric_limits<double>::epsilon())&&(xtol<=std::numeric_limits<double>::epsilon())){
            std::stringstream msg;
            msg << "Invalid tolerances given to MonotoneComponent::Inverse.  Either \"xtol\" or \"ytol\" must be nonzero, but given values are " << xtol << " and " << ytol;
            throw std::invalid_argument(msg.str());
        }



        // Set up the cache for each point
        const unsigned int numPts = ys.extent(0);
        const unsigned int numXs = xs.extent(1); // The number of input points

        // Check to make sure the output and and input have the right sizes.
        if((numXs!=1)&&(numXs!=numPts)){
            std::stringstream msg;
            msg << "Invalid argument sizes given to MonotoneComponent::Inverse. The number of x points is " << numXs << ", but the number of y points is " << numPts << ".  If the number of xs is not 1 then it must match the number of ys.";
            throw std::invalid_argument(msg.str());
        }
        if(output.extent(0)!=numPts){
            std::stringstream msg;
            msg << "Invalid argument sizes given to MonotoneComponent::Inverse.  The output array has size " << output.extent(0) << " but there are N=" << numPts << " to invert.";
            throw std::invalid_argument(msg.str());
        }

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = expansion_.CacheSize();
        quad_.SetDim(1);
        const unsigned int workspaceSize = quad_.WorkspaceSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        unsigned int cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize+workspaceSize);

        auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();
            int info;

            if(ptInd<numPts){
                unsigned int xInd = ptInd;
                if(numXs==1)
                    xInd = 0;

                // Create a subview containing x_{1:d-1}
                auto pt = Kokkos::subview(xs, Kokkos::ALL(), xInd);
                
                // Check for NaNs.  If found, set output to nan and return
                for(unsigned int ii=0; ii<pt.size(); ++ii){
                    if(std::isnan(pt(ii))){
                        output(ptInd) = std::numeric_limits<double>::quiet_NaN();
                        return;
                    }
                }

                // Fill in the cache with everything that doesn't depend on x_d
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                expansion_.FillCache1(cache.data(), pt, DerivativeFlags::None);

                // Compute the inverse
                Kokkos::View<double*,MemorySpace> workspace(team_member.thread_scratch(1), workspaceSize);
                auto eval = SingleEvaluator<decltype(pt),decltype(coeffs)>(workspace.data(), cache.data(), pt, coeffs, quad_, expansion_, nugget_);
                output(ptInd) = RootFinding::InverseSingleBracket<MemorySpace>(ys(ptInd), eval, pt(pt.extent(0)-1), xtol, ytol, info);
            }
        };

        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);

        // Paralel loop over each point computing x_D = T^{-1}(x_1,...,x_{D-1},y_D) for that point
        Kokkos::parallel_for(policy, functor);
    }


    /**
       @brief Approximates the "continuous derivative" \f$\frac{\partial T}{\partial x_D}\f$ derived from the exact integral form of the transport map.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives.
        @param pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param coeffs The ceofficients in an expansion for \f$f\f$.
        @return Kokkos::View<double*> The values of \f$\frac{\partial T}{\partial x_D}\f$ at each point.

        @see DiscreteDerivative
     */
    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    Kokkos::View<double*,MemorySpace>  ContinuousDerivative(StridedMatrix<const double, MemorySpace> const& pts,
                                                            StridedVector<const double, MemorySpace> const& coeffs)
    {
        const unsigned int numPts = pts.extent(1);
        Kokkos::View<double*,MemorySpace> derivs("Monotone Component Derivatives", numPts);

        ContinuousDerivative<ExecutionSpace>(pts,coeffs, derivs);

        return derivs;
    }
    // template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    // Kokkos::View<double*,MemorySpace>  ContinuousDerivative(StridedMatrix<double, MemorySpace> const& pts,
    //                                                         StridedVector<double, MemorySpace> const& coeffs)
    // {
    //     StridedMatrix<const double, MemorySpace> pts2 = pts;
    //     StridedVector<const double, MemorySpace> coeffs2 = coeffs;
    //     return ContinuousDerivative<ExecutionSpace>(pts2,coeffs2);
    // }



    /**
        @brief Approximates the "continuous derivative" \f$\frac{\partial T}{\partial x_D}\f$ derived from the exact integral form of the transport map.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives.
        @param[in] pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param[in] coeffs The ceofficients in an expansion for \f$f\f$.
        @param[in,out] evals  The values of map component itself \f$T\f$ at each point.
        @param[in,out] derivs The values of \f$ \frac{\partial T}{\partial x_D}\f$ at each point.

        @see DiscreteDerivative
     */
    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    void ContinuousDerivative(StridedMatrix<const double, MemorySpace> const& pts,
                              StridedVector<const double, MemorySpace> const& coeffs,
                              StridedVector<double, MemorySpace>              derivs)
    {
        const unsigned int numPts = pts.extent(1);
        const unsigned int dim = pts.extent(0);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = expansion_.CacheSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);

        auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            // The index of the for loop
            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){
                // Create a subview containing only the current point
                auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

                // Evaluate the orthgonal polynomials in each direction (except the last) for all possible orders
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                // Precompute anything that does not depend on x_d.  The DerivativeFlags::None arguments specifies that we won't want to derivative wrt to x_i for i<d
                expansion_.FillCache1(cache.data(), pt, DerivativeFlags::None);

                // Fill in parts of the cache that depend on x_d.  Tell the expansion we're going to want first derivatives wrt x_d
                expansion_.FillCache2(cache.data(), pt, pt(dim-1), DerivativeFlags::Diagonal);

                // Compute \partial_d f
                derivs(ptInd) = expansion_.DiagonalDerivative(cache.data(), coeffs,1);

                // Compute g(df/dx)
                derivs(ptInd) = PosFuncType::Evaluate(derivs(ptInd));
            }
        };


        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
    }

    // template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    // void ContinuousDerivative(StridedMatrix<double, MemorySpace> const& pts,
    //                           StridedVector<double, MemorySpace> const& coeffs,
    //                           StridedVector<double, MemorySpace>        derivs)
    // {
    //     StridedMatrix<const double, MemorySpace> pts2 = pts;
    //     StridedVector<const double, MemorySpace> coeffs2 = coeffs;
    //     ContinuousDerivative<ExecutionSpace>(pts2,coeffs2, derivs);
    // }

    /**
    @brief Approximates the "discrete derivative" of the quadrature-based approximation \f$\tilde{T}\f$.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives.
        @param[in] pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param[in] coeffs The ceofficients in an expansion for \f$f\f$.
        @returns Kokkos::View<double*> The values of \f$ \frac{\partial \tilde{T}}{\partial x_D}\f$ at each point.

        @see ContinuousDerivative
    */
    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    Kokkos::View<double*, MemorySpace>  DiscreteDerivative(StridedMatrix<const double, MemorySpace> const& pts,
                                                           StridedVector<const double, MemorySpace> const& coeffs)
    {
        const unsigned int numPts = pts.extent(1);
        Kokkos::View<double*,MemorySpace> evals("Component Evaluations", numPts);
        Kokkos::View<double*,MemorySpace> derivs("Component Derivatives", numPts);

        DiscreteDerivative<ExecutionSpace>(pts,coeffs, evals, derivs);

        return derivs;
    }

    // template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    // Kokkos::View<double*, MemorySpace>  DiscreteDerivative(StridedMatrix<double, MemorySpace> const& pts,
    //                                                        StridedVector<double, MemorySpace>  const& coeffs)
    // {
    //     StridedMatrix<const double, MemorySpace> pts2 = pts;
    //     StridedVector<const double, MemorySpace> coeffs2 = coeffs;
    //     return DiscreteDerivative<ExecutionSpace>(pts2,coeffs2);
    // }


    /**
       @brief Approximates the "discrete derivative" of the quadrature-based approximation \f$\tilde{T}\f$.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives.
        @param[in] pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param[in] coeffs The ceofficients in an expansion for \f$f\f$.
        @param[out] evals  The values of the map component itself \f$\tilde{T}\f$ at each point.
        @param[out] derivs Kokkos::View<double*> The values of \f$ \frac{\partial \tilde{T}}{\partial x_D}\f$ at each point.

        @see ContinuousDerivative
     */
    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    void  DiscreteDerivative(StridedMatrix<const double, MemorySpace> const& pts,
                             StridedVector<const double, MemorySpace> const& coeffs,
                             StridedVector<double, MemorySpace>              evals,
                             StridedVector<double, MemorySpace>              derivs)
    {
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);

        Kokkos::View<double*,MemorySpace> output("ExpansionOutput", numPts);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = expansion_.CacheSize();

        quad_.SetDim(2);
        const unsigned int workspaceSize = quad_.WorkspaceSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize+workspaceSize+2);

        auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){
                // Create a subview containing only the current point
                auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

                // Get a pointer to the shared memory Kokkos is managing for the cache
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                Kokkos::View<double*,MemorySpace> workspace(team_member.thread_scratch(1), workspaceSize);
                Kokkos::View<double*,MemorySpace> both(team_member.thread_scratch(1), 2);

                // Fill in the cache with anything that doesn't depend on x_d
                expansion_.FillCache1(cache.data(), pt, DerivativeFlags::None);

                // Create the integrand g( \partial_D f(x_1,...,x_{D-1},t))
                MonotoneIntegrand<ExpansionType, PosFuncType, decltype(pt), decltype(coeffs), MemorySpace> integrand(cache.data(), expansion_, pt, coeffs, DerivativeFlags::Diagonal, nugget_);

                // Compute \int_0^x g( \partial_D f(x_1,...,x_{D-1},t)) dt
                quad_.Integrate(workspace.data(), integrand, 0, 1, both.data());
                evals(ptInd) = both(0);
                derivs(ptInd) = both(1);

                // Add f(x_1,x_2,...,x_{d-1},0) to the evaluation output
                expansion_.FillCache2(cache.data(), pt, 0.0, DerivativeFlags::None);
                evals(ptInd) += expansion_.Evaluate(cache.data(), coeffs);
            }
        };

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
    }

    // template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    // void  DiscreteDerivative(StridedMatrix<double, MemorySpace> const& pts,
    //                          StridedVector<double, MemorySpace> const& coeffs,
    //                          StridedVector<double, MemorySpace>        evals,
    //                          StridedVector<double, MemorySpace>        derivs)
    // {
    //     StridedMatrix<const double, MemorySpace> pts2 = pts;
    //     StridedVector<const double, MemorySpace> coeffs2 = coeffs;
    //     DiscreteDerivative(pts2, coeffs2, evals, derivs);
    // }

    bool isJacobianInputValid(int jacRows, int jacCols, int evalRows, int expectJacRows, int expectJacCols, int expectEvalRows) {
        bool isJacRowsCorrect = jacRows == expectJacRows;
        bool isJacColsCorrect = jacCols == expectJacCols;
        bool isEvalRowsCorrect = evalRows == expectEvalRows;
        return isJacRowsCorrect && isJacColsCorrect && isEvalRowsCorrect;
    }

    void checkJacobianInput(const std::string method, int jacRows, int jacCols, int evalRows, int expectJacRows, int expectJacCols, int expectEvalRows) {
        bool isInputValid = isJacobianInputValid(jacRows, jacCols, evalRows, expectJacRows, expectJacCols, expectEvalRows);
        if(!isInputValid) {
            std::stringstream ss;
            ss << method << ": Incorrect input arg sizes. "
               << "jacobian: (" << jacRows << "," << jacCols << "), expected: (" << expectJacRows << "," << expectJacCols << "), ";
            if(expectEvalRows > 0)
                ss << "evaluations: (" << evalRows << "), expected: (" << expectEvalRows << ")";
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }
    }

    /** @brief Returns the gradient of the map with respect to the parameters \f$\mathbf{w}\f$ at multiple points.

        @details
        Consider \f$N\f$ points \f$\{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(N)}\}\f$ and let
        \f$y_d^{(i)} = T_d(\mathbf{x}^{(i)}; \mathbf{w})\f$.   This function computes \f$\nabla_{\mathbf{w}} y_d^{(i)}\f$
        for each output \f$y_d^{(i)}\f$.

        @param[in] pts A \f$D\times N\f$ matrix containing the points \f$x^{(1)},\ldots,x^{(N)}\f$.  Each column is a point.
        @param[in] coeffs A vector of coefficients defining the function \f$f(\mathbf{x}; \mathbf{w})\f$.
        @param[out] evaluations A vector containing the \f$N\f$ predictions \f$y_d^{(i)}\f$.  The vector must be preallocated and have \f$N\f$ components when passed to this function.  An error will occur if this vector is not the correct size.
        @param[out] jacobian A matrix containing the \f$M\times N\f$ Jacobian matrix, where \f$M\f$ is the length of the parameter vector \f$\mathbf{w}\f$.  This matrix must be sized correctly or an error will occur.

        @see CoeffGradient
    */
    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    void CoeffJacobian(StridedMatrix<const double, MemorySpace>  const& pts,
                       StridedVector<const double, MemorySpace>  const& coeffs,
                       StridedVector<double, MemorySpace>               evaluations,
                       StridedMatrix<double, MemorySpace>               jacobian)
    {
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);

        checkJacobianInput("CoeffJacobian", jacobian.extent(0), jacobian.extent(1), evaluations.extent(0), numTerms, numPts, numPts);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = expansion_.CacheSize();
        quad_.SetDim(numTerms+1);
        const unsigned int workspaceSize = quad_.WorkspaceSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize+workspaceSize+numTerms+1);

        auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){
                // Create a subview containing only the current point
                auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                auto jacView = Kokkos::subview(jacobian, Kokkos::ALL(), ptInd);

                // Get a pointer to the shared memory that Kokkos has set up for the cache
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                Kokkos::View<double*,MemorySpace> workspace(team_member.thread_scratch(1), workspaceSize);
                Kokkos::View<double*,MemorySpace> integral(team_member.thread_scratch(1), numTerms+1);

                // Fill in the cache with anything that doesn't depend on x_d
                expansion_.FillCache1(cache.data(), pt, DerivativeFlags::None);

                // Create the integrand g( \partial_D f(x_1,...,x_{D-1},t))
                MonotoneIntegrand<ExpansionType, PosFuncType, decltype(pt),decltype(coeffs), MemorySpace> integrand(cache.data(), expansion_, pt, coeffs, DerivativeFlags::Parameters, nugget_);

                // Compute \int_0^x g( \partial_D f(x_1,...,x_{D-1},t)) dt as well as the gradient of this term wrt the coefficients of f
                quad_.Integrate(workspace.data(), integrand, 0, 1, integral.data());

                evaluations(ptInd) = integral(0);

                expansion_.FillCache2(cache.data(), pt,  0.0, DerivativeFlags::None);
                evaluations(ptInd) += expansion_.CoeffDerivative(cache.data(), coeffs, jacView);

                // Add the Integral to the coefficient gradient
                for(unsigned int termInd=0; termInd<numTerms; ++termInd)
                    jacView(termInd) += integral(termInd+1);
            }

        };

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
    }

    /** @brief Returns the gradient of the map with respect to the input \f$x_{1:d}\f$ at multiple points.

        @details
        Consider \f$N\f$ points \f$\{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(N)}\}\f$ and let
        \f$y_d^{(i)} = T_d(\mathbf{x}^{(i)}; \mathbf{w})\f$.   This function computes \f$\nabla_{\mathbf{x}} y_d^{(i)}\f$
        for each output \f$y_d^{(i)}\f$.

        @param[in] pts A \f$D\times N\f$ matrix containing the points \f$x^{(1)},\ldots,x^{(N)}\f$.  Each column is a point.
        @param[in] coeffs A vector of coefficients defining the function \f$f(\mathbf{x}; \mathbf{w})\f$.
        @param[out] evaluations A vector containing the \f$N\f$ predictions \f$y_d^{(i)}\f$.  The vector must be preallocated and have \f$N\f$ components when passed to this function.  An error will occur if this vector is not the correct size.
        @param[out] jacobian A matrix containing the \f$d\times N\f$ Jacobian matrix, where \f$d\f$ is the length of the input vector \f$\mathbf{x}\f$.  This matrix must be sized correctly or an error will occur.

        @see CoeffGradient
    */
    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    void InputJacobian(StridedMatrix<const double, MemorySpace>  const& pts,
                       StridedVector<const double, MemorySpace>  const& coeffs,
                       StridedVector<double, MemorySpace>               evaluations,
                       StridedMatrix<double, MemorySpace>               jacobian)
    {
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);

        checkJacobianInput("InputJacobian",jacobian.extent(0), jacobian.extent(1), evaluations.extent(0), dim_, numPts, numPts);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = expansion_.CacheSize();
        quad_.SetDim(dim_+1);
        const unsigned int workspaceSize = quad_.WorkspaceSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize+workspaceSize+dim_+1);

        auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){
                // Create a subview containing only the current point
                auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                auto jacView = Kokkos::subview(jacobian, Kokkos::ALL(), ptInd);

                // Get a pointer to the shared memory that Kokkos has set up for the cache
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                Kokkos::View<double*,MemorySpace> workspace(team_member.thread_scratch(1), workspaceSize);
                Kokkos::View<double*,MemorySpace> integral(team_member.thread_scratch(1), dim_+1);

                // Fill in the cache with anything that doesn't depend on x_d
                expansion_.FillCache1(cache.data(), pt, DerivativeFlags::Input);

                // Create the integrand g( \partial_D f(x_1,...,x_{D-1},t))
                MonotoneIntegrand<ExpansionType, PosFuncType, decltype(pt),decltype(coeffs), MemorySpace> integrand(cache.data(), expansion_, pt, coeffs, DerivativeFlags::Input, nugget_);

                // Compute \int_0^x g( \partial_D f(x_1,...,x_{D-1},t)) dt as well as the gradient of this term wrt the map input
                quad_.Integrate(workspace.data(), integrand, 0, 1, integral.data());

                evaluations(ptInd) = integral(0);

                expansion_.FillCache2(cache.data(), pt,  0.0, DerivativeFlags::Input);
                evaluations(ptInd) += expansion_.InputDerivative(cache.data(), coeffs, jacView);

                // Add the Integral to the coefficient gradient
                for(unsigned int d=0; d<dim_-1; ++d){
                    jacView(d) += integral(d+1);
                }

                jacView(dim_-1) = integral(dim_);

            }

        };

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
    }

    void checkMixedJacobianInput(const std::string method, int jacRows, int jacCols, int expectJacRows, int expectJacCols) {
        checkJacobianInput(method, jacRows, jacCols, 0, expectJacRows, expectJacCols, 0);
    }

    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    void ContinuousMixedJacobian(StridedMatrix<const double, MemorySpace> const& pts,
                                 StridedVector<const double, MemorySpace> const& coeffs,
                                 StridedMatrix<double, MemorySpace>              jacobian)
    {
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);
        const unsigned int dim = pts.extent(0);

        checkMixedJacobianInput("ContinuousMixedJacobian", jacobian.extent(0), jacobian.extent(1), numTerms, numPts);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = expansion_.CacheSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);

        auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            // The index of the for loop
            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){
                // Create a subview containing only the current point
                auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                auto jacView = Kokkos::subview(jacobian, Kokkos::ALL(), ptInd);

                // Evaluate the orthgonal polynomials in each direction (except the last) for all possible orders
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                // Precompute anything that does not depend on x_d.  The DerivativeFlags::None arguments specifies that we won't want to derivative wrt to x_i for i<d
                expansion_.FillCache1(cache.data(), pt, DerivativeFlags::None);

                // Fill in parts of the cache that depend on x_d.  Tell the expansion we're going to want first derivatives wrt x_d
                expansion_.FillCache2(cache.data(), pt, pt(dim-1), DerivativeFlags::Diagonal);

                // Compute \partial_d f
                double df = expansion_.MixedCoeffDerivative(cache.data(), coeffs, 1, jacView);
                double dgdf = PosFuncType::Derivative(df);

                // Scale the jacobian by dg(df)
                for(unsigned int i=0; i<numTerms; ++i)
                    jacView(i) *= dgdf;
            }

        };

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
    }

    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    void ContinuousMixedInputJacobian(StridedMatrix<const double, MemorySpace> const& pts,
                                 StridedVector<const double, MemorySpace> const& coeffs,
                                 StridedMatrix<double, MemorySpace>              jacobian)
    {
        const unsigned int numPts = pts.extent(1);
        const unsigned int dim = pts.extent(0);

        checkMixedJacobianInput("ContinuousMixedInputJacobian", jacobian.extent(0), jacobian.extent(1), dim, numPts);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = expansion_.CacheSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize);

        auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            // The index of the for loop
            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){
                // Create a subview containing only the current point
                auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                auto jacView = Kokkos::subview(jacobian, Kokkos::ALL(), ptInd);

                // Evaluate the orthgonal polynomials in each direction (except the last) for all possible orders
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);

                // Precompute anything that does not depend on x_d.  The DerivativeFlags::None arguments specifies that we won't want to derivative wrt to x_i for i<d
                expansion_.FillCache1(cache.data(), pt, DerivativeFlags::MixedInput);

                // Fill in parts of the cache that depend on x_d.  Tell the expansion we're going to want first derivatives wrt x_d
                expansion_.FillCache2(cache.data(), pt, pt(dim-1), DerivativeFlags::MixedInput);

                // Compute \partial_d f
                double df = expansion_.MixedInputDerivative(cache.data(), coeffs, jacView);
                double dgdf = PosFuncType::Derivative(df);


                // Scale the jacobian by dg(df)
                for(unsigned int i=0; i<dim; ++i)
                    jacView(i) *= dgdf;
            }

        };

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
    }

    template<typename ExecutionSpace=typename MemoryToExecution<MemorySpace>::Space>
    void DiscreteMixedJacobian(StridedMatrix<const double, MemorySpace> const& pts,
                               StridedVector<const double, MemorySpace> const& coeffs,
                               StridedMatrix<double, MemorySpace>              jacobian)
    {
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);

        checkMixedJacobianInput("DiscreteMixedJacobian", jacobian.extent(0), jacobian.extent(1), numTerms, numPts);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = expansion_.CacheSize();
        quad_.SetDim(numTerms+1);
        const unsigned int workspaceSize = quad_.WorkspaceSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        auto cacheBytes = Kokkos::View<double*,MemorySpace>::shmem_size(cacheSize+workspaceSize+2*numTerms+1);

        auto functor = KOKKOS_CLASS_LAMBDA (typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member) {

            unsigned int ptInd = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

            if(ptInd<numPts){
                // Create a subview containing only the current point
                auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
                auto jacView = Kokkos::subview(jacobian, Kokkos::ALL(), ptInd);

                // Get a pointer to the shared memory that Kokkos has set up for the cache
                Kokkos::View<double*,MemorySpace> cache(team_member.thread_scratch(1), cacheSize);
                Kokkos::View<double*,MemorySpace> workspace(team_member.thread_scratch(1), workspaceSize);
                Kokkos::View<double*,MemorySpace> integral(team_member.thread_scratch(1), numTerms+1);

                // Fill in the cache with anything that doesn't depend on x_d
                expansion_.FillCache1(cache.data(), pt, DerivativeFlags::None);

                // Create the integrand g( \partial_D f(x_1,...,x_{D-1},t))
                Kokkos::View<double*,MemorySpace> integrandWork(team_member.thread_scratch(1), numTerms);
                MonotoneIntegrand<ExpansionType, PosFuncType,  decltype(pt), decltype(coeffs), MemorySpace> integrand(cache.data(), expansion_, pt, coeffs, DerivativeFlags::Mixed, nugget_, integrandWork);

                // Compute \int_0^x g( \partial_D f(x_1,...,x_{D-1},t)) dt as well as the gradient of this term wrt the coefficients of f
                quad_.Integrate(workspace.data(), integrand, 0, 1, integral.data());

                // Add the Integral to the coefficient gradient
                for(unsigned int termInd=0; termInd<numTerms; ++termInd)
                    jacView(termInd) += integral(termInd+1);
            }
        };


        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        auto policy = GetCachedRangePolicy<ExecutionSpace>(numPts, cacheBytes, functor);
        Kokkos::parallel_for(policy, functor);
    }

    /**
     @brief Evaluates the monotone component at a single point using an existing cache.

     @tparam PointType The type of point used (subview, etc...)
     @param cache Memory used by the MultivariateExpansion to cache evaluations
     @param workspace  Memory used by the quadrature routine to store evaluations
     @param pt
     @param coeffs
     @return double
     */
    template<typename PointType, typename CoeffsType>
    KOKKOS_FUNCTION static double EvaluateSingle(double*                  cache,
                                                 double*                  workspace,
                                                 PointType         const& pt,
                                                 double                   xd,
                                                 CoeffsType        const& coeffs,
                                                 QuadratureType    const& quad,
                                                 ExpansionType     const& expansion,
                                                 double                   nugget=0.0)
    {
        double output = 0.0;
        // Compute the integral \int_0^1 g( \partial_D f(x_1,...,x_{D-1},t*x_d)) dt
        MonotoneIntegrand<ExpansionType, PosFuncType, PointType, CoeffsType, MemorySpace> integrand(cache,
                                                                                       expansion,
                                                                                       pt,
                                                                                       xd,
                                                                                       coeffs,
                                                                                       DerivativeFlags::None,
                                                                                       nugget);
        quad.Integrate(workspace, integrand, 0, 1, &output);

        expansion.FillCache2(cache, pt, 0.0, DerivativeFlags::None);
        output += expansion.Evaluate(cache, coeffs);
        
        return output;
    }

    /** Give access to the underlying FixedMultiIndexSet
     * @return The FixedMultiIndexSet
     */
    FixedMultiIndexSet<MemorySpace> GetMultiIndexSet() const {
        return expansion_.GetMultiIndexSet();
    }

    /**
     * @brief Indices of coeffs that correspond to terms non-constant in \f$x_d\f$
     * 
     * @return std::vector<unsigned int> 
     */
    std::vector<unsigned int> DiagonalCoeffIndices() const {
        return expansion_.NonzeroDiagonalEntries();
    }

#if defined(MPART_HAS_CEREAL)
    // Define a serialize or save/load pair as you normally would
    template <class Archive>
    void save( Archive & ar ) const
    {   
        ar( cereal::base_class<ConditionalMapBase<MemorySpace>>( this )); 
        ar( expansion_, quad_, useContDeriv_, nugget_);
        ar( this->savedCoeffs );
    }

    template <class Archive>
    static void load_and_construct( Archive & ar, cereal::construct<MonotoneComponent<ExpansionType, PosFuncType,QuadratureType,MemorySpace>> & construct )
    {   
        ExpansionType expansion;
        QuadratureType quad;
        bool useContDeriv;
        double nugget;
        ar(expansion, quad, useContDeriv, nugget);

        Kokkos::View<double*, MemorySpace> coeffs;
        ar( coeffs );

        if(coeffs.size() == expansion.NumCoeffs()){
            construct( expansion, quad, useContDeriv, nugget, coeffs);
        }else{
            construct( expansion, quad, useContDeriv, nugget);
        }
    }

#endif // MPART_HAS_CEREAL

private:

    ExpansionType expansion_;
    QuadratureType quad_;
    unsigned int dim_;
    bool useContDeriv_;
    double nugget_;


    template<typename PointType, typename CoeffType>
    struct SingleEvaluator {
        double* workspace;
        double* cache;
        PointType pt;
        CoeffType coeffs;
        QuadratureType quad;
        ExpansionType expansion;
        double nugget;

        SingleEvaluator(double* workspace_, double* cache_, PointType pt_, CoeffType coeffs_, QuadratureType quad_, ExpansionType expansion_, double nugget_):
            workspace(workspace_), cache(cache_), pt(pt_), coeffs(coeffs_), quad(quad_), expansion(expansion_), nugget(nugget_) {};
        double operator()(double x) {
            return EvaluateSingle(cache, workspace, pt, x, coeffs, quad, expansion, nugget);
        }
    };
};

} // namespace mpart

#if defined(MPART_HAS_CEREAL)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory1)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory2)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory3)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory4)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory5)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory6)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory7)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory8)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory9)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory10)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory11)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory12)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory13)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory14)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory15)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory16)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory17)
CEREAL_FORCE_DYNAMIC_INIT(mpartInitMapFactory18)
#endif

#endif