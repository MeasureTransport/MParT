#ifndef MPART_MONOTONECOMPONENT_H
#define MPART_MONOTONECOMPONENT_H

#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexSet.h"

#include "MParT/DerivativeFlags.h"
#include "MParT/MonotoneIntegrand.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>


namespace mpart{


/**
@brief Defines a function \f$T:R^N\rightarrow R\f$ such that \f$\partial T / \partial x_N >0\f$ is strictly positive.

@details 
The function \f$T\f$ is based on another (generally non-monotone) function \f$f : R^N\rightarrow R\f$ and a strictly positve function 
\f$g : R\rightarrow R_{>0}\f$.   Together, these functions define the monotone component $T$ through

$$
T(x_1, x_2, ..., x_D) = f(x_1,x_2,..., x_{D-1}, 0) + \int_0^{x_D}  g\left( \partial_D f(x_1,x_2,..., x_{D-1}, t) \right) dt
$$

@tparam ExpansionType A class defining the function \f$f\f$.  It must satisfy the cached parameterization concept.
@tparam PosFuncType A class defining the function \f$g\f$.  This class must have `Evaluate` and `Derivative` functions accepting a double and returning a double.  The MParT::SoftPlus and MParT::Exp classes in PositiveBijectors.h are examples of classes defining this interface.
@tparam QuadratureType A class defining the integration scheme used to approximate \f$\int_0^{x_N}  g\left( \frac{\partial f}{\partial x_d}(x_1,x_2,..., x_{N-1}, t) \right) dt\f$.  The type must have a function `Integrate(f,lb,ub)` that accepts a functor `f`, a double lower bound `lb`, a double upper bound `ub`, and returns a double with an estimate of the integral.   The MParT::AdaptiveSimpson and MParT::RecursiveQuadrature classes provide this interface. 
*/
template<class ExpansionType, class PosFuncType, class QuadratureType>
class MonotoneComponent
{

public:

    MonotoneComponent(ExpansionType  const& expansion, 
                      QuadratureType const& quad) : _expansion(expansion), _quad(quad), _dim(expansion.InputSize()){};



    /**
       @brief Evaluates the monotone function \f$T(x_1,\ldots,x_D)\f$ at multiple points.
       
     * @param[in] pts A \f$D\times N\f$ array containing the \f$N\f$ points in \f$\mathbb{R}^D\f$ where we want to evaluate the monotone component.  Each column is a point.
     * @param[in] coeffs The coefficients in the expansion defining \f$f\f$.  The length of this array must be the same as the number of terms in the multiindex set passed to the constructor.
     * @return Kokkos::View<double*> An array containing the evaluattions \f$T(x^{(i)}_1,\ldots,x^{(i)}_D)\f$ for each \f$i\in\{0,\ldots,N\}\f$.
     */
    Kokkos::View<double*> Evaluate(Kokkos::View<double**> const& pts, 
                                   Kokkos::View<double*> const& coeffs)
    {   
        const unsigned int numPts = pts.extent(1);
        Kokkos::View<double*> output("Monotone Component Evaluations", numPts);

        Evaluate(pts, coeffs, output);

        return output;
    }

    /**
       @brief Evaluates the monotone function \f$T(x_1,\ldots,x_D)\f$ at multiple points.
       
     * @param[in] pts A \f$D\times N\f$ array containing the \f$N\f$ points in \f$\mathbb{R}^D\f$ where we want to evaluate the monotone component.  Each column is a point.
     * @param[in] coeffs The coefficients in the expansion defining \f$f\f$.  The length of this array must be the same as the number of terms in the multiindex set passed to the constructor.
     * @param[out] output Kokkos::View<double*> An array containing the evaluattions \f$T(x^{(i)}_1,\ldots,x^{(i)}_D)\f$ for each \f$i\in\{0,\ldots,N\}\f$.
     */

    void Evaluate(Kokkos::View<double**> const& pts, 
                  Kokkos::View<double*>  const& coeffs,
                  Kokkos::View<double*>       & output)
    {
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);
        
        assert(output.extent(0)==numPts);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = _expansion.CacheSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {

            unsigned int ptInd = team_member.league_rank();
            
            // Create a subview containing only the current point 
            auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

            // Get a pointer to the shared memory that Kokkos set up for this team
            double* cache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));

            // Fill in entries in the cache that are independent of x_d.  By passing DerivativeFlags::None, we are telling the expansion that no derivatives with wrt x_1,...x_{d-1} will be needed.
            _expansion.FillCache1(cache, pt, DerivativeFlags::None);

            output(ptInd) = EvaluateSingle(cache, pt, pt(_dim-1), coeffs);            
        });
    }


    /**
       @brief Evaluates the inverse of the \f$D^{th}\f$ componeot of the monotone function \f$T(x_1,\ldots,x_D)\f$ at multiple points.
       @param xs A \f$D\times N_1\f$ array containing \f$N_1\f$ \f$x_{1:D-1}\f$ points.  Note that this matrix must have either 1 or N columns and at least \f$D\f$ rows.  The \f$D\f$ row will serve as an initial guess for \f$x_D\f$ during the inversion.
       @param ys A length \f$N\f$ array containing \f$N\f$ values of \f$y_D\f$ for use in the solve.
       @param coeffs The coefficients in the expansion defining \f$f\f$.  The length of this array must be the same as the number of terms in the multiindex set passed to the constructor.
       @param options A map containing options for the method (e.g., converge criteria, step sizes).   Available options are "Method" (must be "Bracket"), "xtol" (any nonnegative float), and "ytol" (any nonnegative float).
       @returns A length \f$N\f$ Kokkos::View<double*> containing \f$y_D^{(i)}\f$.
    */
    Kokkos::View<double*> Inverse(Kokkos::View<double**>       const& xs, 
                                  Kokkos::View<double*>        const& ys, 
                                  Kokkos::View<double*>        const& coeffs,
                                  std::map<std::string, std::string>  options=std::map<std::string,std::string>())
    {   
        const unsigned int numPts = ys.extent(0);
        Kokkos::View<double*> output("Monotone Component Inverse Evaluations", numPts);

        Inverse(xs, ys, coeffs, output, options);

        return output;
    }


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
    void Inverse(Kokkos::View<double**>       const& xs, 
                 Kokkos::View<double*>        const& ys,
                 Kokkos::View<double*>        const& coeffs,
                 Kokkos::View<double*>             & output,
                 std::map<std::string, std::string>  options=std::map<std::string,std::string>())
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

        const unsigned int numTerms = coeffs.extent(0);
        const unsigned int dim = xs.extent(0);

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
        const unsigned int cacheSize = _expansion.CacheSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));

        // Paralel loop over each point computing x_D = T^{-1}(x_1,...,x_{D-1},y_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {

            unsigned int ptInd = team_member.league_rank();
            unsigned int xInd = ptInd;
            if(numXs==1)
                xInd = 0;

            // Create a subview containing x_{1:d-1}
            auto pt = Kokkos::subview(xs, Kokkos::ALL(), xInd);

            // Fill in the cache with everything that doesn't depend on x_d
            double* cache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));
            _expansion.FillCache1(cache, pt, DerivativeFlags::None);

            // Compute the inverse 
            output(ptInd) = InverseSingleBracket(cache, pt, ys(ptInd), coeffs, xtol, ytol);
        });
    }
    


    

    

    /**
       @brief Approximates the "continuous derivative" \f$\frac{\partial T}{\partial x_D}\f$ derived from the exact integral form of the transport map.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives.
        @param pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param coeffs The ceofficients in an expansion for \f$f\f$.
        @return Kokkos::View<double*> The values of \f$\frac{\partial T}{\partial x_D}\f$ at each point.

        @see DiscreteDerivative
     */
    Kokkos::View<double*>  ContinuousDerivative(Kokkos::View<double**> const& pts, 
                                              Kokkos::View<double*>  const& coeffs)
    {   
        const unsigned int numPts = pts.extent(1);
        Kokkos::View<double*> derivs("Monotone Component Derivatives", numPts);
        
        ContinuousDerivative(pts,coeffs, derivs);

        return derivs;
    }

    /**
        @brief Approximates the "continuous derivative" \f$\frac{\partial T}{\partial x_D}\f$ derived from the exact integral form of the transport map.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives.
        @param[in] pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param[in] coeffs The ceofficients in an expansion for \f$f\f$.
        @param[in,out] evals  The values of map component itself \f$T\f$ at each point.
        @param[in,out] derivs The values of \f$ \frac{\partial T}{\partial x_D}\f$ at each point.

        @see DiscreteDerivative
     */
    void ContinuousDerivative(Kokkos::View<double**> const& pts, 
                              Kokkos::View<double*>  const& coeffs,
                              Kokkos::View<double*>       & derivs)
    {   
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);
        const unsigned int dim = pts.extent(0);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = _expansion.CacheSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {
            
            // The index of the for loop
            unsigned int ptInd = team_member.league_rank();

            // Create a subview containing only the current point 
            auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

            // Evaluate the orthgonal polynomials in each direction (except the last) for all possible orders
            double* cache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));

            // Precompute anything that does not depend on x_d.  The DerivativeFlags::None arguments specifies that we won't want to derivative wrt to x_i for i<d
            _expansion.FillCache1(cache, pt, DerivativeFlags::None);

            // Fill in parts of the cache that depend on x_d.  Tell the expansion we're going to want first derivatives wrt x_d
            _expansion.FillCache2(cache, pt, pt(dim-1), DerivativeFlags::Diagonal);

            // Compute \partial_d f
            derivs(ptInd) = _expansion.DiagonalDerivative(cache, coeffs,1);

            // Compute g(df/dx)
            derivs(ptInd) = PosFuncType::Evaluate(derivs(ptInd));
        });
    }

    /**
    @brief Approximates the "discrete derivative" of the quadrature-based approximation \f$\tilde{T}\f$.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives.
        @param[in] pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param[in] coeffs The ceofficients in an expansion for \f$f\f$.
        @returns Kokkos::View<double*> The values of \f$ \frac{\partial \tilde{T}}{\partial x_D}\f$ at each point.

        @see ContinuousDerivative
    */
    Kokkos::View<double*>  DiscreteDerivative(Kokkos::View<double**> const& pts, 
                                              Kokkos::View<double*>  const& coeffs)
    {   
        const unsigned int numPts = pts.extent(1);
        Kokkos::View<double*> evals("Component Evaluations", numPts);
        Kokkos::View<double*> derivs("Component Derivatives", numPts);
        
        DiscreteDerivative(pts,coeffs, evals, derivs);

        return derivs;
    }

    /**
       @brief Approximates the "discrete derivative" of the quadrature-based approximation \f$\tilde{T}\f$.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives.
        @param[in] pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param[in] coeffs The ceofficients in an expansion for \f$f\f$.
        @param[out] evals  The values of the map component itself \f$\tilde{T}\f$ at each point.
        @param[out] derivs Kokkos::View<double*> The values of \f$ \frac{\partial \tilde{T}}{\partial x_D}\f$ at each point.

        @see ContinuousDerivative
     */
    void  DiscreteDerivative(Kokkos::View<double**> const& pts, 
                             Kokkos::View<double*>  const& coeffs,
                             Kokkos::View<double*>       & evals, 
                             Kokkos::View<double*>       & derivs)
    {   
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);
        const unsigned int dim = pts.extent(0);

        assert(coeffs.extent(0)==numTerms);

        Kokkos::View<double*> output("ExpansionOutput", numPts);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = _expansion.CacheSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {
            
            unsigned int ptInd = team_member.league_rank();

            // Create a subview containing only the current point 
            auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);

            // Get a pointer to the shared memory Kokkos is managing for the cache
            double* cache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));
            
            // Fill in the cache with anything that doesn't depend on x_d
            _expansion.FillCache1(cache, pt, DerivativeFlags::None);
            
            // Create the integrand g( \partial_D f(x_1,...,x_{D-1},t))
            MonotoneIntegrand<ExpansionType, PosFuncType, decltype(pt), Kokkos::View<double*>> integrand(cache, _expansion, pt, coeffs, DerivativeFlags::Diagonal);
            
            // Compute \int_0^x g( \partial_D f(x_1,...,x_{D-1},t)) dt
            Eigen::VectorXd both = _quad.Integrate(integrand, 0, 1);
            evals(ptInd) = both(0);
            derivs(ptInd) = both(1);

            // Add f(x_1,x_2,...,x_{d-1},0) to the evaluation output
            _expansion.FillCache2(cache, pt, 0.0, DerivativeFlags::None);
            evals(ptInd) += _expansion.Evaluate(cache, coeffs);
            
        });
    }

    /** @brief Returns the gradient of the map with respect to the parameters \f$\mathbf{w}\f$ at multiple points.

        @details 
        Consider \f$N\f$ points \f$\{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(N)}\}\f$ and let 
        \f$y_d^{(i)} = T_d(\mathbf{x}^{(i)}; \mathbf{w})\f$.   This function computes \f$\nabla_{\mathbf{w}} y_d^{(i)}\f$
        for each output \f$y_d^{(i)}\f$.
        
        @param[in] pts A \f$D\times N\f$ matrix containing the points \f$x^{(1)},\ldots,x^{(N)}\f$.  Each column is a point.
        @param[in] coeffs A vector of coefficients defining the function \f$f(\mathbf{x}; \mathbf{w})\f$.
        @param[out] evaluations A vector containing the \f$N\f$ predictions \f$y_d^{(i)}\f$.  The vector must be preallocated and have \f$N\f$ components when passed to this function.  An assertion will be thrown in this vector is not the correct size.
        @param[out] jacobian A matrix containing the \f$N\times M\f$ Jacobian matrix, where \f$M\f$ is the length of the parameter vector \f$\mathbf{w}\f$.  This matrix must be sized correctly or an assertion will be thrown.
        
        @see CoeffGradient
    */
    void CoeffJacobian(Kokkos::View<double**> const& pts, 
                       Kokkos::View<double*>  const& coeffs,
                       Kokkos::View<double*>       & evaluations,
                       Kokkos::View<double**>      & jacobian)
    {
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);
        const unsigned int dim = pts.extent(0);

        assert(jacobian.extent(0)==numPts);
        assert(jacobian.extent(1)==numTerms);
        assert(evaluations.extent(0)==numPts);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = _expansion.CacheSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));


        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {
            
            unsigned int ptInd = team_member.league_rank();

            // Create a subview containing only the current point 
            auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
            auto jacView = Kokkos::subview(jacobian, ptInd, Kokkos::ALL());

            // Get a pointer to the shared memory that Kokkos has set up for the cache
            double* cache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));

            // Fill in the cache with anything that doesn't depend on x_d
            _expansion.FillCache1(cache, pt, DerivativeFlags::None);
            
            // Create the integrand g( \partial_D f(x_1,...,x_{D-1},t))
            MonotoneIntegrand<ExpansionType, PosFuncType, decltype(pt),Kokkos::View<double*>> integrand(cache, _expansion, pt, coeffs, DerivativeFlags::Parameters);
            
            // Compute \int_0^x g( \partial_D f(x_1,...,x_{D-1},t)) dt as well as the gradient of this term wrt the coefficients of f
            Eigen::VectorXd integral = _quad.Integrate(integrand, 0, 1);
            
            evaluations(ptInd) = integral(0);

            _expansion.FillCache2(cache, pt,  0.0, DerivativeFlags::None);
            evaluations(ptInd) += _expansion.CoeffDerivative(cache, coeffs, jacView);

            // Add the Integral to the coefficient gradient
            for(unsigned int termInd=0; termInd<numTerms; ++termInd)
                jacView(termInd) += integral(termInd+1);
            
        });
    }


    void ContinuousMixedJacobian(Kokkos::View<double**> const& pts, 
                                 Kokkos::View<double*>  const& coeffs,
                                 Kokkos::View<double**>      & jacobian)
    {   
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);
        const unsigned int dim = pts.extent(0);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = _expansion.CacheSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {
            
            // The index of the for loop
            unsigned int ptInd = team_member.league_rank();

            // Create a subview containing only the current point 
            auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
            auto jacView = Kokkos::subview(jacobian, ptInd, Kokkos::ALL());

            // Evaluate the orthgonal polynomials in each direction (except the last) for all possible orders
            double* cache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));

            // Precompute anything that does not depend on x_d.  The DerivativeFlags::None arguments specifies that we won't want to derivative wrt to x_i for i<d
            _expansion.FillCache1(cache, pt, DerivativeFlags::None);

            // Fill in parts of the cache that depend on x_d.  Tell the expansion we're going to want first derivatives wrt x_d
            _expansion.FillCache2(cache, pt, pt(dim-1), DerivativeFlags::Diagonal);

            // Compute \partial_d f
            double df = _expansion.MixedDerivative(cache, coeffs, 1, jacView);
            double dgdf = PosFuncType::Derivative(df);

            // Scale the jacobian by dg(df)
            for(unsigned int i=0; i<numTerms; ++i)
                jacView(i) *= dgdf;

        });
    }

    void DiscreteMixedJacobian(Kokkos::View<double**> const& pts, 
                               Kokkos::View<double*>  const& coeffs,
                               Kokkos::View<double**>      & jacobian)
    {   
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = coeffs.extent(0);
        const unsigned int dim = pts.extent(0);

        assert(jacobian.extent(0)==numPts);
        assert(jacobian.extent(1)==numTerms);

        // Ask the expansion how much memory it would like for it's one-point cache
        const unsigned int cacheSize = _expansion.CacheSize();

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));


        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {
            
            unsigned int ptInd = team_member.league_rank();

            // Create a subview containing only the current point 
            auto pt = Kokkos::subview(pts, Kokkos::ALL(), ptInd);
            auto jacView = Kokkos::subview(jacobian, ptInd, Kokkos::ALL());

            // Get a pointer to the shared memory that Kokkos has set up for the cache
            double* cache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));

            // Fill in the cache with anything that doesn't depend on x_d
            _expansion.FillCache1(cache, pt, DerivativeFlags::None);
            
            // Create the integrand g( \partial_D f(x_1,...,x_{D-1},t))
            MonotoneIntegrand<ExpansionType, PosFuncType, decltype(pt),Kokkos::View<double*>> integrand(cache, _expansion, pt, coeffs, DerivativeFlags::Mixed);
            
            // Compute \int_0^x g( \partial_D f(x_1,...,x_{D-1},t)) dt as well as the gradient of this term wrt the coefficients of f
            Eigen::VectorXd integral = _quad.Integrate(integrand, 0, 1);
            
            // Add the Integral to the coefficient gradient
            for(unsigned int termInd=0; termInd<numTerms; ++termInd)
                jacView(termInd) += integral(termInd+1);
            
        });
    }

private:
    ExpansionType _expansion;
    QuadratureType _quad;
    const unsigned int _dim;


    /**
     @brief Evaluates the monotone component at a single point using an existing cache.

     @tparam PointType The type of point used (subview, etc...)
     @param cache 
     @param pt 
     @param coeffs 
     @return double 
     */
    template<typename PointType, typename CoeffsType>
    double EvaluateSingle(double*                  cache,
                          PointType         const& pt,
                          double                   xd,
                          CoeffsType        const& coeffs)
    {   
        double output = 0.0;
        // Compute the integral \int_0^1 g( \partial_D f(x_1,...,x_{D-1},t*x_d)) dt 
        MonotoneIntegrand<ExpansionType, PosFuncType, PointType, CoeffsType> integrand(cache, 
                                                                           _expansion,
                                                                           pt,
                                                                           xd,
                                                                           coeffs, 
                                                                           DerivativeFlags::None);
        output = _quad.Integrate(integrand, 0, 1)(0);
        
        // Finish filling in the cache for an evaluation of the expansion with x_d=0
        _expansion.FillCache2(cache, pt, 0.0, DerivativeFlags::None);
        output += _expansion.Evaluate(cache, coeffs);   

        return output;
    }


    /**
     @brief Solves \f$y_D = T(x_1,\ldots,x_D)\f$ for \f$x_D\f$ at a single point using the ITP bracketing method.
     @details Uses the [Interpolate Truncate and Project (ITP)](https://en.wikipedia.org/wiki/ITP_method) method to solve 
              \f$y_D = T(x_1,\ldots,x_D)\f$ for \f$x_D\f$.  This method is a bracketing method similar to bisection, but
              with faster convergence.   The worst case performance of ITP, in terms of required evaluations of \f$T\f$,
              is the same as bisection. 

     @param cache Memory set up by Kokkos for this evaluation.  The `_expansion.FillCache1` function must be
                  called before calling this function.
     @param pt Array of length \f$D\f$ containing the fixed values of \f$x_{1:D-1}\f$ and an initial guess for \f$x_D\f$ in the last component.
     @param coeffs An array of coefficients for the expansion.
     @param options 
     @return The value of \f$x_D\f$ solving \f$y_D = T(x_1,\ldots,x_D)\f$

     @tparam PointType The type of the point.  Typically either Kokkos::View<double*> or some subview with a similar 1d signature.
     @tparam CoeffsType The type of the coefficients.  Typically Kokkos::View<double*> or a similarly structured subview.
     */
    template<typename PointType, typename CoeffsType>
    double InverseSingleBracket(double*                             cache,
                               PointType                    const& pt,
                               double                              yd,
                               CoeffsType                   const& coeffs,
                               const double xtol, const double ftol)
    {   
        double stepSize=1.0;
        const unsigned int maxIts = 10000;

        // First, we need to find two points that bound the solution.
        double xlb, xub;
        double ylb, yub;
        double xb, xf; // Bisection point and regula falsi point
        double xc, yc;

        xlb = pt(_dim-1);
        ylb = EvaluateSingle(cache, pt, xlb, coeffs);

        // We actually found an upper bound...
        if(ylb>yd){
            std::swap(ylb,yub);
            std::swap(xlb,xub);

            // Now find a lower bound...
            unsigned int i;
            for(i=0; i<maxIts; ++i){ // Could just be while(true), but want to avoid infinite loop
                xlb = xub-stepSize;
                ylb = EvaluateSingle(cache, pt, xlb, coeffs);
                if(ylb>yd){
                    std::swap(ylb,yub);
                    std::swap(xlb,xub);
                    stepSize *= 2.0;
                }else{
                    break;
                }
            }
            assert(i<maxIts);

        // We have a lower bound...
        }else{
            // Now find an upper bound...
            unsigned int i;
            for(i=0; i<maxIts; ++i){ // Could just be while(true), but want to avoid infinite loop
                xub = xlb+stepSize;
                yub = EvaluateSingle(cache, pt, xub, coeffs);
                if(yub<yd){
                    std::swap(ylb,yub);
                    std::swap(xlb,xub);
                    stepSize *= 2.0;
                }else{
                    break;
                }
            }
            assert(i<10000);
        }

        assert(ylb<yub);
        assert(xlb<xub);

        // Bracketed search 
        const double k1 = 0.1;
        const double k2 = 2.0;
        const double nhalf = std::ceil(std::log2(0.5*(xub-xlb)/xtol));
        const double n0 = 1.0;

        double sigma, delta, rho;
        unsigned int it;
        for(it=0; it<maxIts; ++it){

            xb = 0.5*(xub+xlb); // bisection point
            xf = xlb - (yd-ylb)*(xub-xlb) / (yub-ylb); // regula-falsi point

            sigma = ((xb-xf)>0)?1.0:-1.0; // sign(xb-xf)
            delta = std::min(k1*std::pow((xub-xlb), k2), std::abs(xb-xf));
            
            xf += delta*sigma;
            
            rho = std::min(xtol*std::pow(2.0, nhalf + n0 - it) - 0.5*(xub-xlb), std::abs(xf - xb));
            xc = xb - sigma*rho;

            yc = EvaluateSingle(cache, pt, xc, coeffs);

            if(std::abs(yc-yd)<ftol){
                return xc;
            }else if(yc>yd){
                std::swap(yc,yub);
                std::swap(xc,xub);
            }else{
                std::swap(yc,ylb);
                std::swap(xc,xlb);
            }

            // Check for convergence 
            if(((xub-xlb)<xtol)||((yub-ylb)<ftol))
                break;
        };

        assert(it<maxIts);
        return 0.5*(xub+xlb);
    }



};

} // namespace mpart
#endif 