#ifndef ORTHOGONALPOLYNOMIAL_H
#define ORTHOGONALPOLYNOMIAL_H

#include <cmath>

namespace mpart{

/*
p_{k}(x) = (a_k x + b_k) p_{k-1}(x) - c_k p_{k-2}(x)
*/
template<class Mixer>
class OrthogonalPolynomial : public Mixer
{
public:

    /* Evaluates all polynomials up to a specified order. */
    void EvaluateAll(double*              output,
                     unsigned int         maxOrder,
                     double               x) const
    {
        output[0] = this->phi0(x);

        if(maxOrder>0)
            output[1] = this->phi1(x);

        for(unsigned int order=2; order<=maxOrder; ++order)
            output[order] = (this->ak(order)*x + this->bk(order))*output[order-1] - this->ck(order)*output[order-2];
    }

    /** Evaluates the derivative of every polynomial in this family up to degree maxOrder (inclusive).
        The results are stored in the memory pointed to by the derivs pointer.
    */
    void EvaluateDerivatives(double*      derivs,
                             unsigned int maxDegree,
                             double       x) const
    {
        double oldVal=0;
        double oldOldVal=0;
        double currVal;
        currVal = this->phi0(x);
        derivs[0] = 0.0;

        if(maxDegree>0){
            oldVal = currVal;
            currVal = this->phi1(x);
            derivs[1] = this->phi1_deriv(x);
        }

        // Evaluate the polynomials and their derivatives using the three term recurrence
        double ak, bk, ck;
        for(unsigned int order=2; order<=maxDegree; ++order){
            oldOldVal = oldVal;
            oldVal = currVal;

            ak = this->ak(order);
            bk = this->bk(order);
            ck = this->ck(order);
            currVal = (ak*x + bk)*oldVal - ck*oldOldVal;
            derivs[order] = ak*oldVal + (ak*x + bk)*derivs[order-1] - ck*derivs[order-2];
        }
    }

    /** Evaluates the value and derivative of every polynomial in this family up to degree maxOrder (inclusive).
        The results are stored in the memory pointed to by the derivs pointer.
    */
    void EvaluateDerivatives(double*      vals,
                           double*      derivs,
                           unsigned int maxOrder,
                           double       x) const
    {
        vals[0] = this->phi0(x);
        derivs[0] = 0.0;

        if(maxOrder>0){
            vals[1] = this->phi1(x);
            derivs[1] = this->phi1_deriv(x);
        }

        // Evaluate the polynomials and their derivatives using the three term recurrence
        double ak, bk, ck;
        for(unsigned int order=2; order<=maxOrder; ++order){
            ak = this->ak(order);
            bk = this->bk(order);
            ck = this->ck(order);
            vals[order] = (ak*x + bk)*vals[order-1] - ck*vals[order-2];
            derivs[order] = ak*vals[order-1] + (ak*x + bk)*derivs[order-1] - ck*derivs[order-2];
        }
    }

    void EvaluateSecondDerivatives(double*      vals,
                                   double*      derivs,
                                   double*      secondDerivs,
                                   unsigned int maxOrder,
                                   double       x) const
    {
        vals[0] = this->phi0(x);
        derivs[0] = 0.0;
        secondDerivs[0] = 0.0;

        if(maxOrder>0){
            vals[1] = this->phi1(x);
            derivs[1] = this->phi1_deriv(x);
            secondDerivs[1] = 0.0;
        }

        // Evaluate the polynomials and their derivatives using the three term recurrence
        double ak, bk, ck;
        for(unsigned int order=2; order<=maxOrder; ++order){
            ak = this->ak(order);
            bk = this->bk(order);
            ck = this->ck(order);
            vals[order] = (ak*x + bk)*vals[order-1] - ck*vals[order-2];
            derivs[order] = ak*vals[order-1] + (ak*x + bk)*derivs[order-1] - ck*derivs[order-2];
            secondDerivs[order] = ak*derivs[order-1] + ak*derivs[order-1] + (ak*x+bk)*secondDerivs[order-1] - ck*secondDerivs[order-2];
        }
    }



    double Evaluate(unsigned int const order,
                    double const x) const
    {
        if(order==0){
            return this->phi0(x);
        }else if(order==1){
            return this->phi1(x);
        }else{

            // "Downward" Clenshaw algorithm  http://mathworld.wolfram.com/ClenshawRecurrenceFormula.html
            double yk2 = 0.0;
            double yk1 = 0.0;
            double yk = 1.0;
            double alpha, beta;

            for( int k=order-1; k>=0; k-- ) {
                yk2 = yk1;
                yk1 = yk;

                alpha = this->ak(k+1)*x + this->bk(k+1);
                beta = -this->ck(k+2);
                yk = alpha*yk1 + beta*yk2;
            }

            beta = -this->ck(2);
            return yk1*this->phi1(x) + beta * this->phi0(x)*yk2;
        }
    }

    double Derivative(unsigned int const order,
                      double const x) const
    {
        if(order==0){
            return 0.0;
        }else if(order==1){
            return this->phi1_deriv(x);
        }else{
            double lag2_val;
            double lag1_val = this->phi0(x);
            double next_val = this->phi1(x);
            double lag2_deriv;
            double lag1_deriv = 0.0;
            double next_deriv =  this->phi1_deriv(x);

            // Evaluate the polynomials and their derivatives using the three term recurrence
            double ak, bk, ck;
            for(unsigned int i=2; i<=order; ++i){
                ak = this->ak(i);
                bk = this->bk(i);
                ck = this->ck(i);

                lag2_val = lag1_val;
                lag1_val = next_val;
                next_val = (ak*x + bk)*lag1_val - ck*lag2_val;


                lag2_deriv = lag1_deriv;
                lag1_deriv = next_deriv;
                next_deriv = ak*lag1_val + (ak*x + bk)*lag1_deriv - ck*lag2_deriv;
            }

            return next_deriv;
        }
    }

    double SecondDerivative(unsigned int const order,
                      double const x) const
    {
        if(order<=1){
            return 0.0;
        }else{
            double lag2_val;
            double lag1_val = this->phi0(x);
            double next_val = this->phi1(x);

            double lag2_deriv;
            double lag1_deriv = 0.0;
            double next_deriv = this->phi1_deriv(x);

            double lag2_deriv2;
            double lag1_deriv2 = 0.0;
            double next_deriv2 = 0.0;

            // Evaluate the polynomials and their derivatives using the three term recurrence
            double ak, bk, ck;
            for(unsigned int i=2; i<=order; ++i){
                ak = this->ak(i);
                bk = this->bk(i);
                ck = this->ck(i);

                lag2_val = lag1_val;
                lag1_val = next_val;
                next_val = (ak*x + bk)*lag1_val- ck*lag2_val;

                lag2_deriv = lag1_deriv;
                lag1_deriv = next_deriv;
                next_deriv = ak*lag1_val + (ak*x + bk)*lag1_deriv - ck*lag2_deriv;

                lag2_deriv2 = lag1_deriv2;
                lag1_deriv2 = next_deriv2;
                next_deriv2 = ak*lag1_deriv + ak*lag1_deriv + (ak*x + bk)*lag1_deriv2 - ck*lag2_deriv2;
            }

            return next_deriv2;
        }
    }
};


class ProbabilistHermiteMixer{
public:

    double Normalization(unsigned int polyOrder) const {return sqrt(2.0*M_PI) * std::tgamma(polyOrder+1); }

protected:

    double ak(unsigned int k) const {return 1.0;}
    double bk(unsigned int k) const {return 0.0;}
    double ck(unsigned int k) const {return k-1.0;}
    double phi0(double x) const {return 1.0;}
    double phi1(double x) const {return x;}
    double phi1_deriv(double x) const{return 1.0;};
};

typedef OrthogonalPolynomial<ProbabilistHermiteMixer> ProbabilistHermite;


class PhysicistHermiteMixer{
public:

    double Normalization(unsigned int polyOrder) const {return sqrt(M_PI) * pow(2.0, static_cast<double>(polyOrder)) * std::tgamma(polyOrder+1); }

protected:

    double ak(unsigned int k) const {return 2.0;}
    double bk(unsigned int k) const {return 0.0;}
    double ck(unsigned int k) const {return 2.0*(k-1.0);}
    double phi0(double x) const {return 1.0;}
    double phi1(double x) const {return 2.0*x;}
    double phi1_deriv(double x) const {return 2.0;};
};

typedef OrthogonalPolynomial<PhysicistHermiteMixer> PhysicistHermite;


} // namespace mpart

#endif