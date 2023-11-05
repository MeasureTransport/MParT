#ifndef MPART_LINEARIZEDBASIS_H
#define MPART_LINEARIZEDBASIS_H

#include <Kokkos_Core.hpp>


namespace mpart{

/**
 * @brief Basis that is linear outside a given upper and lower bound.
 * @details Implemented to be piecewise-defined mapping \f$F_{ab}[\psi]\f$, where \f$\psi\f$ is a real-valued univariate function. We define
 * \f[ F_{ab}[\psi](x) = \begin{cases}\psi(x) & x\in[a,b]\\\psi(a)+(x-a)\psi^\prime(a) & x<a\\\psi(b) + (x-b)\psi^\prime(b) && x>b \f]
 *
 * @tparam OtherBasis type of basis inside bounds (e.g. OrthogonalPolynomial)
 */
template<typename OtherBasis>
class LinearizedBasis
{
public:

    LinearizedBasis(double     lb=-3,
                    double     ub=3) : lb_(lb),
                                       ub_(ub)
    {
        assert(lb<ub);
    }

    LinearizedBasis(OtherBasis polyBasis,
                    double     lb=-3,
                    double     ub=3) : polyBasis_(polyBasis),
                                       lb_(lb),
                                       ub_(ub)
    {
        assert(lb<ub);
    }

    KOKKOS_INLINE_FUNCTION void EvaluateAll(double*              output,
                                            unsigned int         maxOrder,
                                            double               x) const
    {

        if(x<lb_){
            polyBasis_.EvaluateAll(output, maxOrder, lb_);

            // Now update the values based on the derivative value at the left point
            for(unsigned int i=0; i<=maxOrder; ++i)
                output[i] += polyBasis_.Derivative(i,lb_) * (x-lb_);

        }else if(x>ub_){
            polyBasis_.EvaluateAll(output, maxOrder, ub_);

            // Now update the values based on the derivative value at the right point
            for(unsigned int i=0; i<=maxOrder; ++i)
                output[i] += polyBasis_.Derivative(i,ub_) * (x-ub_);

        }else{
            polyBasis_.EvaluateAll(output, maxOrder, x);
        }
    }

   KOKKOS_INLINE_FUNCTION  void EvaluateDerivatives(double*              vals,
                                                    double*              derivs,
                                                    unsigned int         maxOrder,
                                                    double               x) const
    {
        if(x<lb_){

            // Evaluate the underlying basis at the left linearization point
            polyBasis_.EvaluateDerivatives(vals, derivs, maxOrder, lb_);

            // Now update the basis values based on the derivative values at the left point
            for(unsigned int i=0; i<=maxOrder; ++i)
                vals[i] += derivs[i]*(x-lb_);

        }else if(x>ub_){

            // Evaluate the underlying basis at the right linearization point
            polyBasis_.EvaluateDerivatives(vals, derivs, maxOrder, ub_);

            // Now update the basis values based on the derivative values at the right point
            for(unsigned int i=0; i<=maxOrder; ++i)
                vals[i] += derivs[i]*(x-ub_);

        }else{
            polyBasis_.EvaluateDerivatives(vals, derivs, maxOrder, x);
        }
    }


    KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(double*              vals,
                                   double*              derivs,
                                   double*              derivs2,
                                   unsigned int         maxOrder,
                                   double               x) const
    {
        if(x<lb_){
            EvaluateDerivatives(vals, derivs, maxOrder, x);
            for(unsigned int i=0; i<=maxOrder; ++i)
                derivs2[i] = 0.0;

        }else if(x>ub_){
            EvaluateDerivatives(vals, derivs, maxOrder, x);
            for(unsigned int i=0; i<=maxOrder; ++i)
                derivs2[i] = 0.0;

        }else{
            polyBasis_.EvaluateSecondDerivatives(vals, derivs, derivs2, maxOrder, x);
        }
    }


    KOKKOS_INLINE_FUNCTION double Evaluate(unsigned int const order,
                                           double const x) const
    {
        if(x<lb_){
            return polyBasis_.Evaluate(order,lb_) + polyBasis_.Derivative(order,lb_)*(x-lb_);
        }else if(x>ub_){
            return polyBasis_.Evaluate(order,ub_) + polyBasis_.Derivative(order,ub_)*(x-ub_);
        }else{
            return polyBasis_.Evaluate(order, x);
        }
    }

    KOKKOS_INLINE_FUNCTION double Derivative(unsigned int const order,
                                             double const x) const
    {
        if(x<lb_){
            return polyBasis_.Derivative(order,lb_);
        }else if(x>ub_){
            return polyBasis_.Derivative(order,ub_);
        }else{
            return polyBasis_.Derivative(order, x);
        }
    }

    KOKKOS_INLINE_FUNCTION double SecondDerivative(unsigned int const order,
                            double const x) const
    {
        if(x<lb_){
            return 0.0;
        }else if(x>ub_){
            return 0.0;
        }else{
            return polyBasis_.SecondDerivative(order, x);
        }
    }

#if defined(MPART_HAS_CEREAL)

    template <class Archive>
    void serialize( Archive & ar )
    {   
        ar( polyBasis_, lb_, ub_ );
    }
#endif 

private:
    OtherBasis polyBasis_;

    double lb_; //<- Left linearization point
    double ub_; //<- Right linearization point

}; // class LinearizedBasis

}

#endif